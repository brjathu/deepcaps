from keras import backend as K
from keras import layers, models, optimizers
from keras.layers import Layer
from keras.layers import Input, Conv2D, Activation, Dense, Dropout, Lambda, Reshape, Concatenate
from keras.layers import BatchNormalization, MaxPooling2D, Flatten, Conv1D, Deconvolution2D, Conv2DTranspose
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from keras.layers.convolutional import UpSampling2D
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from capslayers import Conv2DCaps, ConvCapsuleLayer3D, CapsuleLayer, CapsToScalars, Mask_CID, ConvertToCaps, FlattenCaps


# To limit the GPU usage
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)

def DeepCapsNet(input_shape, n_class, routings):
    # assemble encoder
    x = Input(shape=input_shape)
    l = x

    l = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding="same")(l)  # common conv layer
    l = BatchNormalization()(l)
    l = ConvertToCaps()(l)

    l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])

    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])

    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])
    l1 = l

    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = ConvCapsuleLayer3D(kernel_size=3, num_capsule=32, num_atoms=8, strides=1, padding='same', routings=3)(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])
    l2 = l


    la = FlattenCaps()(l2)
    lb = FlattenCaps()(l1)
    l = layers.Concatenate(axis=-2)([la, lb])

#     l = Dropout(0.4)(l)
    digits_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=32, routings=routings, channels=0, name='digit_caps')(l)

    l = CapsToScalars(name='capsnet')(digits_caps)

    m_capsnet = models.Model(inputs=x, outputs=l, name='capsnet_model')

    y = Input(shape=(n_class,))

    masked_by_y = Mask_CID()([digits_caps, y])  
    masked = Mask_CID()(digits_caps)

    # Decoder Network
    decoder = models.Sequential(name='decoder')
    decoder.add(Dense(input_dim=32, activation="relu", output_dim=8 * 8 * 16))
    decoder.add(Reshape((8, 8, 16)))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Deconvolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    decoder.add(Deconvolution2D(32, 3, 3, subsample=(2, 2), border_mode='same'))
    decoder.add(Deconvolution2D(16, 3, 3, subsample=(2, 2), border_mode='same'))
    decoder.add(Deconvolution2D(8, 3, 3, subsample=(2, 2), border_mode='same'))
    decoder.add(Deconvolution2D(3, 3, 3, subsample=(1, 1), border_mode='same'))
    decoder.add(Activation("relu"))
    decoder.add(Reshape(target_shape=(64, 64, 3), name='out_recon'))

    train_model = models.Model([x, y], [m_capsnet.output, decoder(masked_by_y)])
    eval_model = models.Model(x, [m_capsnet.output, decoder(masked)])
    train_model.summary()

    return train_model, eval_model





def BaseCapsNet(input_shape, n_class, routings):
    # assemble encoder
    x = Input(shape=input_shape)
    l = x

    l = Conv2D(256, (9, 9), strides=(2, 2), activation='relu', padding="same")(l) 
    l = BatchNormalization()(l)
    l = Conv2D(256, (9, 9), strides=(2, 2), activation='relu', padding="same")(l) 
    l = BatchNormalization()(l)
    l = ConvertToCaps()(l)

    l = Conv2DCaps(16, 6, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)

    digits_caps = CapsuleLayer(num_capsule=10, dim_capsule=8, routings=routings, channels=0, name='digit_caps')(l)
    l = CapsToScalars(name='capsnet')(digits_caps)

    m_capsnet = Model(inputs=x, outputs=l, name='capsnet_model')
    y = layers.Input(shape=(n_class,))

    masked_by_y = Mask()([digits_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digits_caps)

    # Decoder Network
    decoder = models.Sequential(name='decoder')
    decoder.add(Dense(input_dim=80, activation="relu", output_dim=8 * 8 * 16))
    decoder.add(Reshape((8, 8, 16)))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(layers.Deconvolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    decoder.add(layers.Deconvolution2D(32, 3, 3, subsample=(2, 2), border_mode='same'))
    decoder.add(layers.Deconvolution2D(16, 3, 3, subsample=(2, 2), border_mode='same'))
    decoder.add(layers.Deconvolution2D(3, 3, 3, subsample=(1, 1), border_mode='same'))
    decoder.add(Activation("relu"))
    decoder.add(layers.Reshape(target_shape=(32, 32, 3), name='out_recon'))

    train_model = models.Model([x, y], [m_capsnet.output, decoder(masked_by_y)])
    eval_model = models.Model(x, [m_capsnet.output, decoder(masked)])
    train_model.summary()

    return train_model, eval_model


