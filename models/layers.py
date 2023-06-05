import tensorflow as tf
from tensorflow import keras 



class Residual3x3Unit(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out, stride, droprate=0., activate_before_residual=False):
        super(Residual3x3Unit, self).__init__()
        self.bn_0 = BatchNormalization(momentum=0.999)
        self.relu_0 = LeakyReLU(alpha=0.1)
        self.conv_0 =Conv2D(channels_out, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn_1 = BatchNormalization(momentum=0.999)
        self.relu_1 = LeakyReLU(alpha=0.1)
        self.conv_1 = Conv2D(channels_out, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.downsample = channels_in != channels_out
        self.shortcut = Conv2D(channels_out, kernel_size=1, strides=stride, use_bias=False)
        self.activate_before_residual = activate_before_residual
        self.dropout = Dropout(rate=droprate)
        self.droprate = droprate

    @tf.function
    def call(self, x, training=True):
        if self.downsample and self.activate_before_residual:
            x = self.relu_0(self.bn_0(x, training=training))
        elif not self.downsample:
            out = self.relu_0(self.bn_0(x, training=training))
        out = self.relu_1(self.bn_1(self.conv_0(x if self.downsample else out), training=training))
        if self.droprate > 0.:
            out = self.dropout(out)
        out = self.conv_1(out)
        return out + (self.shortcut(x) if self.downsample else x)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_units, channels_in, channels_out, unit, stride, droprate=0., activate_before_residual=False):
        super(ResidualBlock, self).__init__()
        self.units = self._build_unit(n_units, unit, channels_in, channels_out, stride, droprate, activate_before_residual)

    def _build_unit(self, n_units, unit, channels_in, channels_out, stride, droprate, activate_before_residual):
        units = []
        for i in range(n_units):
            units.append(unit(channels_in if i == 0 else channels_out, 
                        channels_out, stride if i == 0 else 1, droprate, activate_before_residual))
        return units

    @tf.function
    def call(self, x, training=True):
        for unit in self.units:
            x = unit(x, training=training)
        return x

