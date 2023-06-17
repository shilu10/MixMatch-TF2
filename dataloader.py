import time, os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as ans 
from tqdm import tqdm 
import shutil 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import Model
from tensorflow.keras import layers 
from tensorflow.keras.layers import *
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import *
from datetime import datetime
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense, Input, UpSampling2D, Conv2DTranspose, Conv2D, add, Add,\
                    Lambda, Concatenate, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, \
                    Add, LayerNormalization, Activation, LeakyReLU, SeparableConv2D, Softmax, MaxPooling2D


class DataLoader:
    """
        Class, will be useful for creating the BYOL dataset or dataset for the DownStream task
            like classification or segmentation.
        Methods:
            __download_data(scope: private)
            __normalize(scope: private)
            __preprocess_img(scope: private)
             __get_valdata(scope: private)
            get_dataset(scope: public)
            __create_tf_dataset(scope: public)

        Property:
            dname(dtype: str)        : dataset name(supports cifar10, cifar100).
            n_val(type: int)         : Number of validation data needed, this will be created by splitting the testing
                                       data.
            resize_shape(dtype: int) : Resize shape, bcoz pretrained models, might have a different required shape.
            normalize(dtype: bool)   : bool value, whether to normalize the data or not.
            n_labeled(dtype: int)    : number of training samples needed to be labeled.
    """

    def __init__(self, dname="cifar10", n_val=5000, normalize=True, n_labelled_samples=100):
        assert dname in ["cifar10", 'cifar100', "svhn"], "supported datasets are cifar10, cifar100,svhn"
        assert n_val <= 10_000, "ValueError: nval value should be <= 10_000"

        self.__n_labelled_samples = n_labelled_samples
        train_data, test_data = self.__download_data(dname)
        self.__train_X, self.__train_y = train_data
        self.__dtest_X, self.__dtest_y = test_data

        self.__get_unlabeled_data()
        self.__get_valdata(n_val)
        self.__normalize() if normalize else None


    def __len__(self):
        return self.__train_X.shape[0] + self.__dtest_X.shape[0]

    def __repr__(self):
        return f"Training Samples: {self.__train_X.shape[0]}, Testing Samples: {self.__dtest_X.shape[0]}"

    def __download_data(self, dname):
        """
            Downloads the data from the tensorflow website using the tensorflw.keras.load_data() method.
            Params:
                dname(type: Str): dataset name, it just supports two dataset cifar10 or cifar100
            Return(type(np.ndarray, np.ndarray))
                returns the training data and testing data
        """
        if dname == "cifar10":
            train_data, test_data = tf.keras.datasets.cifar10.load_data()

        if dname == "cifar100":
            train_data, test_data = tf.keras.datasets.cifar100.load_data()

        if dname == "svhn":
            dataset = tfds.load(name='svhn_cropped')
            train_data = dataset['train']
            test_data = dataset['test']

        return train_data, test_data

    def __normalize(self):
        """
            this method, will used to normalize the inputs.
        """
        self.__train_X = self.__train_X / 255.0
        self.__dtest_X = self.__dtest_X / 255.0


    def __preprocess_tf_dataset(self, tf_ds, batch_size, transform=False, subset="unlablled"):
        try:
            tf_ds = tf_ds.shuffle(1024, seed=42)
            tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
            if transform:
                if subset == 'unlabelled':
                    tf_ds = tf_ds.map(lambda x: self.__augment(x, is_label=False),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

                else:
                    tf_ds = tf_ds.map(lambda x, y: self.__augment(x, y),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)


            tf_ds = tf_ds.prefetch(tf.data.experimental.AUTOTUNE)
            return tf_ds

        except Exception as err:
            return err

    def get_dataset(self, batch_size, subset="unlabelled",
                                        transform=False, k_augmentation=1):
        """
            this method, will gives the byol dataset, which is nothing
            but a tf.data.Dataset object.
            Params:
                batch_size(dtype: int)   : Batch Size.
                subset(dtype: str) : which type of dataset needed

            return(type: tf.data.Dataset)
                returns the tf.data.Dataset for intended dataset_type,
                by preprocessing and converting the np data.
        """
        try:
            if subset == "unlabelled":
                tf_ds = tf.data.Dataset.from_tensor_slices((self.__unlabelled_X))
                res = []

                for _ in range(k_augmentation):
                    inner_res = self.__preprocess_tf_dataset(
                                            tf_ds=tf_ds,
                                            batch_size=batch_size,
                                            transform=transform,
                                            subset=subset
                                        )
                    res.append(inner_res)

                return tf.data.Dataset.zip(tuple(res))

            if subset == "labelled":
                tf_ds = tf.data.Dataset.from_tensor_slices((self.__labelled_X, self.__labelled_y))
                tf_ds = self.__preprocess_tf_dataset(
                                        tf_ds=tf_ds,
                                        batch_size=batch_size,
                                        transform=transform,
                                        subset=subset
                                    )
                return tf_ds

            if subset == "val":
                tf_ds = tf.data.Dataset.from_tensor_slices((self.__val_X, self.__val_y))
                tf_ds = self.__preprocess_tf_dataset(
                                        tf_ds=tf_ds,
                                        batch_size=batch_size,
                                        transform=transform,
                                        subset=subset
                                    )
                return tf_ds

            if subset == "test":
                tf_ds = tf.data.Dataset.from_tensor_slices((self.__test_X, self.__test_y))
                tf_ds = self.__preprocess_tf_dataset(
                                        tf_ds=tf_ds,
                                        batch_size=batch_size,
                                        transform=transform,
                                        subset=subset
                                    )
                return tf_ds

        except Exception as err:
            return err

    def __get_valdata(self, nval):
        """
            this method is used to create a validation data by randomly sampling from the testing data.
            Params:
                nval(dtype: Int); Number of validation data needed, rest of test_X.shape[0] - nval, will be
                                  testing data size.
            returns(type; np.ndarray, np.ndarray):
                returns the testing and validation dataset.
        """
        try:
            ind_arr = np.arange(10_000)
            val_inds = np.random.choice(ind_arr, nval, replace=False)
            test_inds = [i for i in ind_arr if not i in val_inds]

            self.__test_X, self.__test_y = self.__dtest_X[test_inds], self.__dtest_y[test_inds]
            self.__val_X, self.__val_y = self.__dtest_X[val_inds], self.__dtest_y[val_inds]

        except Exception as err:
            raise err

    def __get_unlabeled_data(self):
        try:
            ind_arr = np.arange(40_000)
            labelled_inds = np.random.choice(
                                            ind_arr,
                                            self.__n_labelled_samples,
                                            replace=False
                                        )
            unlabelled_inds = [i for i in ind_arr if not i in labelled_inds]
            self.__labelled_X = self.__train_X[labelled_inds]
            self.__labelled_y = self.__train_y[labelled_inds]

            self.__unlabelled_X = self.__train_X[unlabelled_inds]
            self.__unlabelled_y = self.__train_y[unlabelled_inds]

        except Exception as err:
            return err

    @tf.function
    def __augment(self, x, label=None, is_label=True):
        try:
            # random left right flipping
            x = tf.image.random_flip_left_right(x)
            # random pad and crop
            x = tf.pad(x, paddings=[(0, 0), (4, 4), (4, 4), (0, 0)], mode='REFLECT')
            x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x)
            if not is_label:
                return x
            else:
                return x, label

        except Exception as err:
            return err

