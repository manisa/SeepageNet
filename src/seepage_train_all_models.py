import os
import argparse
import random
import time
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow_addons as tfa
import sys; sys.path.insert(0, '..')
from lib.metrics import jaccard, dice_coef, bce_dice_loss
from lib.dataloader import DataGen


from archs.seepage_nopca_arch import Depthwise_Seepage_Inception
from archs.seepage_seblock import Depthwise_Seepage_Inception_SE
from archs.seepage_cbamblock import Depthwise_Seepage_Inception_CBAM
from archs.seepage_baseline_model import Baseline_Normal
from archs.seepage_pca_arch import Depthwise_Seepage_Inception_PCA



import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json


print('TensorFlow version: {version}'.format(version=tf.__version__))
print('Keras version: {version}'.format(version=tf.keras.__version__))
print('Eager mode enabled: {mode}'.format(mode=tf.executing_eagerly()))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))



def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_text_to_file(filepath, line, model_name):
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a') as file:
        if file_exists:
            file.write('\n')  # Add a newline before appending new lines
            file.write('Training time for ' + str(model_name) + ' = ' + str(line) + ' Hours ' +  '\n')


def compile_and_train_model(config, valid_ids, train_test_ids):
    
    filename = "models_train_time.txt"
    save_train_time = os.path.join(config.all_models_path, filename)
    print(save_train_time)
    print('Loading dataset...')
    #train_ids, test_ids = train_test_split(train_test_ids, test_size=config.test_perc, random_state=config.seed)
    train_ids = train_test_ids
    for arch in config.model_list:
        config.model_type = arch
        print(config.model_type)
        print('Creating directories to store model and results...')
        
        model_path =  config.model_type + "_" + config.loss_function 
        config.model_path = os.path.join(config.all_models_path, model_path)
        create_dir(config.model_path)
        print(config)

        tf.keras.backend.clear_session()

        if config.model_type == "Baseline_Normal_Neg":
            model = Baseline_Normal(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

        elif config.model_type == "Depthwise_Seepage_Inception_Neg":
            model = Depthwise_Seepage_Inception(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

        elif config.model_type == "Depthwise_Seepage_Inception_SE_Neg":
            model = Depthwise_Seepage_Inception_SE(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

        elif config.model_type == "Depthwise_Seepage_Inception_CBAM_Neg":
            model = Depthwise_Seepage_Inception_CBAM(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

        elif config.model_type == "Depthwise_Seepage_Inception_PCA_Concatenate_Neg":
            model = Depthwise_Seepage_Inception_PCA(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)



        # Defining optimizer
        if config.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, beta_1=config.beta1, beta_2=config.beta2, epsilon=1e-08)

        # Defining loss functions
        if config.loss_function == "bce":
            model_loss = tf.keras.losses.BinaryCrossentropy(
                                            from_logits=True,   
                                            label_smoothing=0.0,
                                            axis=-1,
                                            reduction="auto",
                                            name="binary_crossentropy")
        elif config.loss_function == "dice_loss":
            model_loss = dice_loss
        elif config.loss_function == "bce_dice_loss":
            model_loss = bce_dice_loss


        # define metrics
        metrics =[dice_coef, jaccard]
        model.compile(loss=model_loss, optimizer=optimizer, metrics=metrics)
        print(f'model created and compiled for model {config.model_type}')
        print(model.summary())


        csv_path = config.model_path + "/metrics_" + config.model_type + ".csv"
        
        print("Loading data generator...")
        image_size = (config.img_height, config.img_width)
        train_generator = DataGen(train_ids, config.train_path, img_height=config.img_height, img_width=config.img_width, batch_size=config.batch_size)
        valid_generator = DataGen(valid_ids, config.train_path, img_height=config.img_height, img_width=config.img_width, batch_size=config.batch_size)


        steps_per_epoch = len(train_ids) // config.batch_size
        val_steps_per_epoch = len(valid_ids) // config.batch_size

        print(f' Training and Validation steps are {steps_per_epoch}, {val_steps_per_epoch}')
        
        #lr_sched = tf.keras.callbacks.LearningRateScheduler(step_decay)
        
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.06, patience=6, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0.0000000000000001),
                            tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', patience=8, mode='max', verbose=1),
                            tf.keras.callbacks.CSVLogger(csv_path),
                            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config.model_path, 'best_model.h5'), monitor='val_dice_coef', mode='max', save_best_only=True, save_weights_only=False, verbose=1)]

       
        print(f'Fitting model {config.model_type}...')



        start_time = time.time()
        history = model.fit(train_generator, validation_data=valid_generator,
                                        steps_per_epoch=steps_per_epoch, 
                                        validation_steps=val_steps_per_epoch,
                                        epochs=config.num_epochs, callbacks=callbacks, shuffle=True,  verbose=2)
        end_time = time.time()
        training_time = (end_time - start_time) / 3600

        print(f'Training time for config.model_type {training_time}')
        save_text_to_file(save_train_time, training_time, config.model_type)
        
        tf.keras.backend.clear_session()
        print(f'==================Model {config.model_type} training completed in {training_time} hours====================')
        
    

def main(config):
    # Create directories
    create_dir(config.all_models_path)
    create_dir(config.all_results_path)
    create_dir(config.all_graphs_path)

    image_filenames = sorted(glob(os.path.join(config.train_path, "images/*")))
    random.Random(2023).shuffle(image_filenames)
    
    # Split train and valid data using train_test_split
    train_ids, valid_ids = train_test_split(image_filenames, test_size=config.valid_perc, random_state=config.seed)

    print(len(valid_ids), len(train_ids))

    compile_and_train_model(config, valid_ids, train_ids)


if __name__ == '__main__': 
    # model hyper-parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--input_filters', type=int, default=32)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=8e-4) 
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_function', type=str, default='bce_dice_loss')
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)      # momentum2 in Adam    

    parser.add_argument('--model_list', type=str, default=['Depthwise_Seepage_Inception_PCA_Concatenate_Neg'])
    parser.add_argument('--all_models_path', type=str, default='../models/')
    parser.add_argument('--all_graphs_path', type=str, default='../models/bce_dice_loss_loss_graphs')
    parser.add_argument('--all_results_path', type=str, default='../results/')

    parser.add_argument('--train_path', type=str, default='../datasets/seepage_augmented_6_28_23_11241/')
    parser.add_argument('--valid_perc', type=float, default=0.28)
    parser.add_argument('--test_perc', type=float, default=0.0)
    

    parser.add_argument('--seed', type=int, default=6464)
    config = parser.parse_args()
    main(config)
