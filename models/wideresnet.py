from layers import ResidualBlock


class WideResNet(tf.keras.Model):
    def __init__(self, num_classes, depth=28, width=2, droprate=0., input_shape=(None, 32, 32, 3), **kwargs):
        super(WideResNet, self).__init__(input_shape, **kwargs)
        assert (depth - 4) % 6 == 0
        N = int((depth - 4) / 6)
        channels = [16, 16 * width, 32 * width, 64 * width]

        self.conv_0 = tf.keras.layers.Conv2D(channels[0], kernel_size=3, strides=1, padding='same', use_bias=False)
        self.block_0 = ResidualBlock(N, channels[0], channels[1], Residual3x3Unit, 1, droprate, True)
        self.block_1 = ResidualBlock(N, channels[1], channels[2], Residual3x3Unit, 2, droprate)
        self.block_2 = ResidualBlock(N, channels[2], channels[3], Residual3x3Unit, 2, droprate)
        self.bn_0 = BatchNormalization(momentum=0.999)
        self.relu_0 = LeakyReLU(alpha=0.1)
        self.avg_pool = AveragePooling2D((8, 8), (1, 1))
        self.flatten = Flatten()
        self.dense = Dense(num_classes)

    @tf.function
    def call(self, inputs, training=True):
        x = inputs
        x = self.conv_0(x)
        x = self.block_0(x, training=training)
        x = self.block_1(x, training=training)
        x = self.block_2(x, training=training)
        x = self.relu_0(self.bn_0(x, training=training))
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
