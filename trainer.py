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



def train_step(labelled_batch, unlabelled_batch, model, ema_model, optimizer,
                   xe_loss_tracker, l2_loss_tracker, total_loss_tracker, metric_tracker, **kwargs):
    
    T = kwargs.get('T')
    K = kwargs.get('K')
    beta = kwargs.get('beta')
    ema_decay_rate = kwargs.get('ema_decay_rate')
    weight_decay_rate = kwargs.get('weight_decay_rate')
    lambda_u = kwargs.get('lambda_u')
    

    train_X, train_y = labelled_batch
    train_U = unlabelled_batch
    batch_size = train_X.shape[0]
    
    with tf.GradientTape() as tape:
        # running mixmatch to get a combined training dataset.(unlabeled and labeled)
        XU, XUy = mixmatch(model, train_X, train_y, train_U, T, K, beta)
        logits = [model(XU[0])]
        for batch in XU[1:]:
            logits.append(model(batch))

        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = tf.concat(logits[1:], axis=0)

        # compute loss
        xe_loss, l2u_loss = semi_loss(XUy[: batch_size], logits_x, XUy[batch_size: ], logits_u)
        total_loss = xe_loss + lambda_u * l2u_loss
    
    model_params = model.trainable_weights 
    grads = tape.gradient(total_loss, model_params)
    optimizer.apply_gradients(zip(grads, model_params))
    
    # update the weights of both models.
    ema_weight_update(model, ema_model, ema_decay_rate)
    weight_decay(model, weight_decay_rate)
    
    metric_obj_func = tf.keras.metrics.CategoricalAccuracy()
    acc = metric_obj_func(train_y, model(train_X))
    xe_loss_tracker.update_state(xe_loss)
    l2_loss_tracker.update_state(l2u_loss)
    total_loss_tracker.update_state(total_loss)
    metric_tracker.update_state(acc)
    
    return {
        "accuracy": metric_tracker.result(),
        'xe_loss': xe_loss_tracker.result(),
        "l2_loss": l2_loss_tracker.result(),
        "total_loss": total_loss_tracker.result()
    }
    

def test_step(val_batch, model, metric_tracker, loss_tracker):
    X, y = val_batch
    batch_size = X.shape[0]
    # cal the loss with logits and y
    logits = model(X, training=False)
    loss_obj_function = tf.keras.losses.SparseCrossentropy()
    loss_val = loss_obj_function(y, logits)
    loss_tracker.update_state(loss_val)
    
    # cal the acc with logits and y
    acc_obj_function = tf.keras.metrics.SparseAccuracy()
    acc_val = acc_obj_function(y, logits)
    metric_tracker.update_state(acc_val)
    
    return {
        'accuracy': metric_tracker.result(),
        'loss': loss_tracker.result()
    }


def train(labelled_ds, unlabelled_ds, val_ds, epochs, **kwargs):
    # loss and metrics trackers
    xe_loss_tracker = tf.keras.metrics.Mean()
    l2_loss_tracker = tf.keras.metrics.Mean()
    total_loss_tracker = tf.keras.metrics.Mean()
    train_acc_tracker = tf.keras.metrics.Mean()
    val_loss_tracker = tf.keras.metrics.Mean()
    val_acc_tracker = tf.keras.metrics.Mean()
    
    # arguments
    K = kwargs.get("K")
    beta = kwargs.get('beta')
    T = kwargs.get("T")
    ema_decay_rate = kwargs.get('ema_decay_rate')
    weight_decay_rate = kwargs.get('weight_decay_rate')
    learning_rate = kwargs.get("learning_rate")
    lambda_u = kwargs.get("lambda_u")
    n_classes = kwargs.get("n_classes")
    ckpt_dir = kwargs.get('ckpt_dir')
    log_path = kwargs.get('log_path')
    
    # models and optimizer
    model = WideResNet(n_classes, depth=28, width=2)
    ema_model = WideResNet(n_classes, depth=28, width=2)
    ema_model.set_weights(model.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # checkpoints
    model_ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(model_ckpt, f'{ckpt_dir}/model', max_to_keep=3)
    # for ema model
    ema_ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=ema_model)
    ema_manager = tf.train.CheckpointManager(ema_ckpt, f'{ckpt_dir}/ema', max_to_keep=3)
    
    # summary writers
    train_writer = tf.summary.create_file_writer(f'{log_path}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/validation')
    
    model_ckpt.restore(manager.latest_checkpoint)
    ema_ckpt.restore(ema_manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
        
    for epoch in range(epochs):
        print(f'Epoch; {epoch}')
        for unlabelled_batch in tqdm(unlabelled_ds):
            model_ckpt.step.assign_add(1)
            ema_ckpt.step.assign_add(1)
            for i, labelled_batch in enumerate(labelled_ds):
                if i == 1:
                    break
                accuracy, xe_loss, l2_loss, total_loss = train_step(labelled_batch,
                                                            unlabelled_batch,
                                                            model,
                                                            ema_model,
                                                            optimizer,
                                                            xe_loss_tracker,
                                                            l2_loss_tracker, 
                                                            total_loss_tracker,
                                                            train_acc_tracker,
                                                            K=K,
                                                            T=T, 
                                                            beta=beta,
                                                            ema_decay_rate=ema_decay_rate,
                                                            weight_decay_rate=weight_decay_rate,
                                                            lambda_u=lambda_u
                                                            )
        
        for val_batch in val_ds:
            val_accuracy, val_loss = test_step(
                                        val_batch,
                                        model,
                                        val_acc_tracker,
                                        val_loss_tracker
                                    )
        with train_writer.as_default():
            tf.summary.scalar('xe_loss', xe_loss, step=epoch)
            tf.summary.scalar('l2u_loss', l2_loss, step=epoch)
            tf.summary.scalar('total_loss', total_loss, step=epoch)
            tf.summary.scalar('accuracy', accuracy, step=epoch)
        
        with val_writer.as_default():
            tf.summary.scalar('xe_loss', val_loss, step=epcoh)
            tf.summary.scalar('val_accuracy', val_accuracy, step=epoch)   
            
        if epoch % 10 == 0:
            model_save_path = manager.save(checkpoint_number=int(model_ckpt.step))
            ema_save_path = ema_manager.save(checkpoint_number=int(ema_ckpt.step))
            print(f'Saved model checkpoint for epoch {int(model_ckpt.step)} @ {model_save_path}')
            print(f'Saved ema checkpoint for epoch {int(ema_ckpt.step)} @ {ema_save_path}')
            
        print(f"train_loss: {total_loss}, xe_loss: {xe_loss}, l2_loss: {l2_loss}, train_accuracy: {accuracy}")
        print(f"val_loss: {val_loss}, val_accuracy: {val_accuracy}")
    
    for writer in [train_writer, val_writer]:
        writer.flush()