import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, add

from training.model.convblock import Conv2Block, Conv3Block
from training.model.fcnblock import FCNBlock

class FCN (Model):
    def __init__(self, num_output_channels: int, name="FCN", **kwargs):
        super(FCN, self).__init__(name=name, **kwargs)
        self.num_output_channels = num_output_channels

    def build(self, input_shape):
        ### Encoder
        self.conv_2_block_1 = Conv2Block(64, 3, 1, "same", "conv_2_block_1")
        self.conv_2_block_2 = Conv2Block(128, 3, 1, "same", "conv_2_block_2")
        self.conv_3_block_3 = Conv3Block(256, 3, 1, "same", "conv_3_block_3")
        self.conv_3_block_4 = Conv3Block(512, 3, 1, "same", "conv_3_block_4")
        self.conv_3_block_5 = Conv3Block(512, 3, 1, "same", "conv_3_block_5")

        ### Decoder
        self.fcn_block = FCNBlock(num_output_channels=self.num_output_channels, name="FCNBlock")
        self.conv_f4 = Conv2D(filters=self.num_output_channels, kernel_size=1, padding="same",
                              activation=None, name="conv_f4")
        self.conv_f3 = Conv2D(filters=self.num_output_channels, kernel_size=1, padding="same",
                              activation=None, name="conv_f3")

        self.conv_transpose_f4 = Conv2DTranspose(filters=self.num_output_channels, kernel_size=4, strides=2,
                                                 use_bias=False, padding='same', activation='relu', name="conv_transpose_f4")

        self.conv_transpose_f3 = Conv2DTranspose(filters=self.num_output_channels, kernel_size=16, strides=8,
                                                 padding='same', activation="sigmoid", name="conv_transpose_f3")

        super(FCN, self).build(input_shape)

    def call(self, input_tensor):
        x = input_tensor
        f1 = self.conv_2_block_1(x)
        f2 = self.conv_2_block_2(f1)
        f3 = self.conv_3_block_3(f2)
        f4 = self.conv_3_block_4(f3)
        f5 = self.conv_3_block_5(f4)

        conv_f4 = self.conv_f4(f4)
        fcn_transpose_1 = self.fcn_block(f5)

        merge_1 = add([conv_f4, fcn_transpose_1])
        fcn_transpose_2 = self.conv_transpose_f4(merge_1)

        conv_f3 = self.conv_f3(f3)
        merge_2 = add([conv_f3, fcn_transpose_2])

        output = self.conv_transpose_f3(merge_2)
        return output

    def get_config(self):
        config = super(FCN, self).get_config()
        config.update({"num_output_channels": self.num_output_channels})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
