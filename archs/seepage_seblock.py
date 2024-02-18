import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, DepthwiseConv2D, Dense, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute
from tensorflow.keras.models import Model, model_from_json


def spatial_squeeze_excite_block(input):
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(input)

    x = multiply([input, se])
    return x

def squeeze_excite_block(input, ratio=8):

    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)


    se = GlobalMaxPooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def channel_spatial_squeeze_excite(input, ratio=8):
    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = add([cse, sse])
    return x



def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='gelu', name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters,(num_row,num_col),padding='same',kernel_regularizer=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(x)
    x = tfa.layers.GroupNormalization(groups=filters, axis= channel_axis)(x)
    if(activation == None):
        return x

    #x = tfa.activations.gelu(x, approximate=True)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return x


def conv2d_bn(x, filters, num_row, num_col):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #x = Conv2D(filters,(num_row,num_col),padding='same',kernel_regularizer=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(x)
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    x = initial_conv2d_bn(x, filters, num_row, num_col)

    return x

def decoder_block(inputs, skip, filters, idx=None):
    x = tf.keras.layers.UpSampling2D((2, 2),interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip])
    x = squeeze_excite_block(x)
    x = depthwise_inception_module(x, filters, 3, 3, 'decoder_conv_'+idx)
    x = spatial_squeeze_excite_block(x)
    x = depthwise_inception_module(x, filters, 3, 3,'decoder_conv_'+idx+idx)
    return x


def depthwise_inception_module(x, filters, num_row,  num_col, name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj = filters//8, filters//8, filters//2, filters//8, filters//4, filters//8

    shortcut = depthwise_separable(x, filters_1x1+filters_3x3+filters_5x5+filters_pool_proj, 3, 3)
    conv_1x1 = initial_conv2d_bn(x, filters_1x1, 1, 1, padding='same', activation='selu')

    #conv_3x3 = initial_conv2d_bn(x, filters_3x3, 1, 1, padding='same', activation='selu')
    conv_3x3 = depthwise_separable(x, filters_3x3, 3, 3)
    conv_3x3 = depthwise_separable(conv_3x3, filters_3x3, 3, 3)

    #conv_5x5 = initial_conv2d_bn(x, filters_5x5, 1, 1, padding='same', activation='selu')
    conv_5x5 = depthwise_separable(x, filters_5x5, 5, 5)
    conv_5x5 = depthwise_separable(conv_5x5, filters_5x5, 5, 5)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = depthwise_separable(pool_proj, filters_pool_proj, 3, 3)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    output = tfa.layers.GroupNormalization(groups=filters, axis=channel_axis)(output)
    output = add([shortcut, output])
    
    output = tf.keras.layers.LeakyReLU(alpha=0.02)(output)
    output = tfa.layers.GroupNormalization(groups=filters//4, axis=channel_axis)(output)
    return output


def depthwise_separable(x, filters, num_row, num_col):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = SeparableConv2D(filters, (num_row, num_col), padding='same',kernel_regularizer=None,kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(x)
    x = tfa.layers.GroupNormalization(groups=filters, axis= channel_axis)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return x


def Depthwise_Seepage_Inception_SE(input_filters, height, width, n_channels):
     #inputs = Input((height, width, n_channels), name = "input_image")
    filters = input_filters
    model_input = Input(shape=(height, width, n_channels))
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    """ Pretrained resnet"""
    tf.keras.backend.clear_session()
    base_model = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_tensor=model_input, pooling=max)
    print("Number of layers in the base model: ", len(base_model.layers))
 

    base_model.trainable = True
    
    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
    for i, layer in enumerate(base_model.layers[:-48]):
        layer.trainable = False

    """ Encoder """
    s11 = squeeze_excite_block(base_model.get_layer("input_1").output)     ## (256 x 256)
    s11 = depthwise_inception_module(s11, filters, 3, 3)           
    s11 = spatial_squeeze_excite_block(s11)
    s11 = depthwise_inception_module(s11, filters, 3, 3, 'encoder_1')
               

    s21 = (base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    f2 = s21.shape[channel_axis]
    s21 = squeeze_excite_block(s21)
    s21 = depthwise_inception_module(s21, f2//2, 3, 3) 
    s21 = spatial_squeeze_excite_block(s21)
    s21 = depthwise_inception_module(s21, f2//2, 3, 3, 'encoder_2')

    s31 = (base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    f3 = s31.shape[channel_axis]
    s31 = squeeze_excite_block(s31)
    s31 = depthwise_inception_module(s31, f3//2, 3, 3)
    s31 = spatial_squeeze_excite_block(s31)
    s31 = depthwise_inception_module(s31, f3//2, 3, 3, 'encoder_3')

    s41 = (base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)
    f4 = s41.shape[channel_axis]
    s41 = squeeze_excite_block(s41)
    s41 = depthwise_inception_module(s41, f4//2, 3, 3)
    s41 = spatial_squeeze_excite_block(s41)
    s41 = depthwise_inception_module(s41, f4//2, 3, 3, 'encoder_4')
    """ Bridge """
    b11 = (base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    f5 = b11.shape[channel_axis]
    b11 = squeeze_excite_block(b11)
    b11 = depthwise_inception_module(b11, f5//2, 3, 3)
    b11= spatial_squeeze_excite_block(b11)
    b11 = depthwise_inception_module(b11, f5//2, 3, 3, 'bottleneck')
 

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*4, '1')                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*2, '2')                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*1, '3')                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters//2, '4')                          ## (256 x 256)
    d41 = initial_conv2d_bn(d41, filters//2, 3,3)
    

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="Depthwise_Seepage_Inception_SE")

    return model



def main():

# Define the model

    model = Depthwise_Seepage_Inception_SE(32, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main() 