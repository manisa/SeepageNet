import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, Concatenate, DepthwiseConv2D, Dense, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute, Add, Concatenate
from tensorflow.keras.models import Model, model_from_json



def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer=tf.keras.initializers.HeNormal(seed=2023),
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer=tf.keras.initializers.HeNormal(seed=2023),
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature
    
    #avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    avg_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(cbam_feature)
    #assert avg_pool.shape[-1] == 1
    #max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(cbam_feature)
    #assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    #assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat) 
    assert cbam_feature.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])


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
    x = channel_attention(x)
    x = depthwise_inception_module(x, filters, 3, 3, 'decoder_conv_'+idx)
    x = spatial_attention(x)
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


def Depthwise_Seepage_Inception_CBAM(input_filters, height, width, n_channels):
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
    s11 = channel_attention(base_model.get_layer("input_1").output)     ## (256 x 256)
    s11 = depthwise_inception_module(s11, filters, 3, 3)           
    s11 = spatial_attention(s11)
    s11 = depthwise_inception_module(s11, filters, 3, 3, 'encoder_1')
               

    s21 = (base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    f2 = s21.shape[channel_axis]
    s21 = channel_attention(s21)
    s21 = depthwise_inception_module(s21, f2//2, 3, 3) 
    s21 = spatial_attention(s21)
    s21 = depthwise_inception_module(s21, f2//2, 3, 3, 'encoder_2')

    s31 = (base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    f3 = s31.shape[channel_axis]
    s31 = channel_attention(s31)
    s31 = depthwise_inception_module(s31, f3//2, 3, 3)
    s31 = spatial_attention(s31)
    s31 = depthwise_inception_module(s31, f3//2, 3, 3, 'encoder_3')

    s41 = (base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)
    f4 = s41.shape[channel_axis]
    s41 = channel_attention(s41)
    s41 = depthwise_inception_module(s41, f4//2, 3, 3)
    s41 = spatial_attention(s41)
    s41 = depthwise_inception_module(s41, f4//2, 3, 3, 'encoder_4')
    """ Bridge """
    b11 = (base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    f5 = b11.shape[channel_axis]
    b11 = channel_attention(b11)
    b11 = depthwise_inception_module(b11, f5//2, 3, 3)
    b11= spatial_attention(b11)
    b11 = depthwise_inception_module(b11, f5//2, 3, 3, 'bottleneck')
 

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*4, '1')                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*2, '2')                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*1, '3')                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters//2, '4')                          ## (256 x 256)
    d41 = initial_conv2d_bn(d41, filters//2, 3,3)
    

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="Depthwise_Seepage_Inception_CBAM")

    return model



def main():

# Define the model

    model = Depthwise_Seepage_Inception_CBAM(32, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main() 