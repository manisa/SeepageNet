#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Flatten,Lambda, Conv2DTranspose,AlphaDropout,  Dropout, SeparableConv2D,Concatenate, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute, LocallyConnected2D
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import MultiHeadAttention, Add, LayerNormalization, Flatten



class PCALayer(tf.keras.layers.Layer):
    def __init__(self, n_components, **kwargs):
        super(PCALayer, self).__init__(**kwargs)
        self.n_components = n_components

    def build(self, input_shape):
        self.shape = input_shape
        self.input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.n_components), dtype="float32",
                                      initializer='glorot_uniform',
                                      trainable=False)

    

    def call(self, x):
        # Flatten the input tensor
        #x = tf.linalg.normalize(x,axis=-1)
        #print(x.shape)
        # assumption is that the feature vector is normalized
        #x = tf.math.l2_normalize(x, axis=-1)
        variance = 0.1
        batch_size = tf.shape(x)[0]
        flattened = tf.reshape(x, [batch_size, -1, self.input_dim])
        
        # Compute the mean and subtract it from the input tensor
        mean = tf.reduce_mean(flattened, axis=1, keepdims=True)
        centered = flattened - mean
        

        # Compute the covariance matrix
        cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flattened)[1] - 1, tf.float32)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = tf.linalg.eigh(cov)



        # Sort the eigenvectors based on the eigenvalues
        idx = tf.argsort(eigenvalues, axis=-1, direction='DESCENDING')
        top_eigenvectors = tf.gather(eigenvectors, idx, batch_dims=1, axis=-1)
        top_eigenvectors = top_eigenvectors[:, :, :self.n_components]

        # Compute the explained variance and cumulative variance ratios
        #explained_variance_ratio = tf.math.divide(eigenvalues, tf.reduce_sum(eigenvalues))
        #cumulative_variance_ratio = tf.cumsum(explained_variance_ratio)
        #tf.print(cumulative_variance_ratio, [cumulative_variance_ratio], "This is cumulative cumulative_variance_ratio")


        # Compute the number of components to retain based on the desired variance threshold
        #num_components = tf.math.ceil(tf.reduce_sum(tf.cast(cumulative_variance_ratio >= variance, tf.float32)))
        #num_components = tf.cast(num_components, tf.int32)

        #tf.print("Num of components selected: ", num_components)

    
        #top_eigenvectors = top_eigenvectors[:, :, :self.n_components]

        # Transpose the eigenvectors to match the input shape
        top_eigenvectors = tf.transpose(top_eigenvectors, perm=[0, 1, 2])

        #print("Number of components:", tf.get_static_value(num_components))
    
        # Project centered data onto top principal components
        projected = tf.matmul(centered, top_eigenvectors)

        # Reshape projected data and return as output
        output_shape = tf.concat([tf.shape(x)[:-1], [self.n_components]], axis=0)
        #output = tf.reshape(projected, shape=(-1, *self.output_shape))
        output = tf.reshape(projected, output_shape)
        return output


    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.n_components,)

    def get_config(self):
        config = super(PCALayer, self).get_config()
        config.update({'n_components': self.n_components})
        return config



def attention_through_filters(inputs, name=None):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    #se = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal')(inputs)
    #inputs_1 = initial_conv2d_bn(inputs, filters, num_row, num_col)
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False)(inputs)
    
    inputs_1 = multiply([inputs, se])
    filters = inputs_1.shape[channel_axis]
    se_shape = (1, 1, filters)

    # Use global average pooling to reduce the spatial dimensions
    x_1 = GlobalAveragePooling2D()(inputs_1)
    x_1 = Reshape(se_shape)(x_1)

    
    x_2 = GlobalMaxPooling2D()(inputs_1)
    x_2 = Reshape(se_shape)(x_2)

    
    
    x_1 = multiply([inputs_1, x_1])
    x_2 = multiply([inputs_1, x_2])
    
    x = add([x_1, x_2])
    ca = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=False)(x)
    
    x = multiply([x, ca], name=name) # long shot giving attention

    return x


def spatial_pooling_block(inputs, ratio=4, name=None):
    
    #inputs = iterLBlock(inputs, filters)
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    
    
    se_shape = (1, 1, filters)
    

    spp_1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(inputs)
    
    spp_2 = MaxPooling2D(pool_size=(4,4), strides=(1,1), padding='same')(inputs)    
    
    spp_3 = MaxPooling2D(pool_size=(8,8), strides=(1,1), padding='same')(inputs)
    
    feature = Concatenate()([spp_1,spp_2, spp_3])
    feature = Conv2D(filters,(1, 1),padding='same',kernel_regularizer=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(feature)    
    feature = tfa.layers.GroupNormalization(groups=filters, axis= channel_axis)(feature)
    feature = Activation('sigmoid')(feature)
    x = multiply([inputs, feature])
    x = add([inputs,x], name=name)
    return x

def attention_block(input_tensor, name=None):

    spatial_attention = attention_through_filters(input_tensor)
    channel_attention = spatial_pooling_block(spatial_attention)
    

    # Output the channel-spatial attention block
    output_tensor = add([channel_attention, input_tensor])
    return output_tensor



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
    x = attention_through_filters(x, 'decoder_att_filters_'+idx)
    x = depthwise_inception_module(x, filters, 3, 3, 'decoder_conv_'+idx)
    x = spatial_pooling_block(x, 'decoder_spatial_att_'+idx)
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
    output = tfa.layers.GroupNormalization(groups=filters, axis=channel_axis)(output)
    return output


def depthwise_separable(x, filters, num_row, num_col):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = SeparableConv2D(filters, (num_row, num_col), padding='same',kernel_regularizer=None,kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(x)
    x = tfa.layers.GroupNormalization(groups=filters, axis= channel_axis)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return x


def Depthwise_Seepage_Inception_PCA(input_filters, height, width, n_channels):
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
    s11 = attention_through_filters(base_model.get_layer("input_1").output, name='encoder_att_filters_1')     ## (256 x 256)
    s11 = depthwise_inception_module(s11, filters, 3, 3)           
    s11 = spatial_pooling_block(s11)
    s11 = depthwise_inception_module(s11, filters, 3, 3, 'encoder_1')
    print(s11.shape)
               

    s21 = (base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    f2 = s21.shape[channel_axis]
    pc2 = PCALayer(f2)(s21)
    s21 = attention_through_filters(s21, 'encoder_att_filters_2')
    s21 = depthwise_inception_module(s21, f2, 3, 3) 
    s21 = spatial_pooling_block(s21, 'encoder_spatial_att_1')
    s21 = concatenate([s21, pc2])
    s21 = depthwise_inception_module(s21, f2//2, 3, 3, 'encoder_2')

    #s21 = tf.keras.layers.SpatialDropout2D(rate = 0.01)(s21)

    print(s21.shape)

    s31 = (base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    f3 = s31.shape[channel_axis]
    pc3 = PCALayer(f3)(s31)
    s31 = attention_through_filters(s31, 'encoder_att_filters_3')
    s31 = depthwise_inception_module(s31, f3, 3, 3)
    s31 = spatial_pooling_block(s31, 'encoder_spatial_att_2')
    s31 = concatenate([s31, pc3])
    s31 = depthwise_inception_module(s31, f3//2, 3, 3, 'encoder_3')
    #s31 = tf.keras.layers.SpatialDropout2D(rate = 0.01)(s31)
    print(s31.shape)

    s41 = (base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)
    f4 = s41.shape[channel_axis]
    pc4 = PCALayer(f4)(s41)
    s41 = attention_through_filters(s41, 'encoder_att_filters_4')
    s41 = depthwise_inception_module(s41, f4, 3, 3)
    s41 = spatial_pooling_block(s41, 'encoder_spatial_att_3')
    s41 = concatenate([s41, pc4])
    s41 = depthwise_inception_module(s41, f4//2, 3, 3, 'encoder_4')
    #s41 = tf.keras.layers.SpatialDropout2D(rate = 0.01)(s41)
    print(s41.shape)
    
    """ Bridge """
    b11 = (base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    f5 = b11.shape[channel_axis]
    bpc1 = PCALayer(f5)(b11)
    b11 = attention_through_filters(b11, 'bottleneck_att_filters')
    b11 = depthwise_inception_module(b11, f5, 3, 3)
    b11= spatial_pooling_block(b11, 'bottleneck_spatial_att')
    b11 = concatenate([b11, bpc1])
    b11 = depthwise_inception_module(b11, f5//2, 3, 3, 'bottleneck')
    #b11 = tf.keras.layers.SpatialDropout2D(rate = 0.01)(b11)
    print(b11.shape)
    print("DECODERS SHAPE")

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*4, '1')                         ## (32 x 32)
    print(d11.shape)
    d21 = decoder_block(d11, s31, filters*2, '2')                         ## (64 x 64)
    print(d21.shape)
    d31 = decoder_block(d21, s21, filters*1, '3')                         ## (128 x 128)
    print(d31.shape)
    d41 = decoder_block(d31, s11, filters//1, '4')                          ## (256 x 256)
    print(d41.shape)
    d41 = initial_conv2d_bn(d41, filters//2, 3,3)
    print(d41.shape)
    

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="Depthwise_Seepage_Inception_PCA")

    return model



def main():

# Define the model

    model = Depthwise_Seepage_Inception_PCA(32, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main() 