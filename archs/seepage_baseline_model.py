import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, SeparableConv2D, Concatenate, DepthwiseConv2D, Dense, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute
from tensorflow.keras.models import Model, model_from_json

def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    #x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(x)
    #x = tfa.layers.GroupNormalization(groups=filters, axis= channel_axis)(x)
    #if(activation == None):
     #   return x
    #x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation(activation, name=name)(x)
    return x


def conv2d_bn(x, filters, num_row, num_col):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    return x

def decoder_block(inputs, skip, filters):
    x = tf.keras.layers.UpSampling2D((2, 2),interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip])
    output= conv2d_bn(x, filters, 3, 3)
    return output

#['conv1_conv', 'conv2_block3_1_conv', 'conv3_block4_1_conv', 'conv4_block6_1_conv']


def Baseline_Normal(input_filters, height, width, n_channels):
    #inputs = Input((height, width, n_channels), name = "input_image")
    filters = input_filters
    model_input = Input(shape=(height, width, n_channels))
    """ Pretrained resnet"""
    tf.keras.backend.clear_session()
    #base_model = tf.keras.applications.MobileNetV2(input_tensor=model_input, include_top=False, weights="imagenet",  alpha=1.3)
    base_model = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_tensor=model_input, pooling=max)

    #resnet50 = keras.applications.ResNet50(
     #   weights="imagenet", include_top=False, input_tensor=model_input
    #)
    print("Number of layers in the base model: ", len(base_model.layers))


    base_model.trainable = True
    
    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
    for i, layer in enumerate(base_model.layers[:-48]):
        layer.trainable = False


    """ Encoder """
    s11 = base_model.get_layer("input_1").output               ## (256 x 256)
    s21 = base_model.get_layer("conv1_conv").output    ## (128 x 128)
    s31 = base_model.get_layer("conv2_block3_1_conv").output   ## (64 x 64)
    s41 = base_model.get_layer("conv3_block4_1_conv").output    ## (32 x 32)

    
    """ Bridge """
    b11 = base_model.get_layer("conv4_block6_1_conv").output   ## (16 x 16)
    #b11 = conv2d_bn(b11, filters*16, 3, 3)
    

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*8)                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*4)                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*2)                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters*1)                          ## (256 x 256)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="Baseline_Normal")

    return model



def main():

# Define the model

    model = Baseline_Normal(32, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main()