# Import packages
import csv
import datetime
import itertools
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import tarfile

from IPython.display import display
from os import listdir
from os.path import isfile, join, exists
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_score, recall_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def split_datasets(train_size=0.80, val_size=0.15, test_size=0.05):
    # Create Train, Test, and Validation datasets
    print()
    if os.path.isdir('lung_image_sets') is False:
        print('Extracting "lung_image_sets"')
        my_tar = tarfile.open('lung_image_sets.tar')
        my_tar.extractall()
        my_tar.close()
        print('"lung_image_sets" has been extracted')
    else:
        print('"lung_image_sets" exists')

    src_dir = 'lung_image_sets'
    out_dir = 'split_image_sets'
    class_dir = ['train','test','val']
    child_dir = os.listdir(src_dir)

    # Split datasets
    try:
        os.mkdir(out_dir)
        for cls in class_dir:
            os.mkdir(out_dir + '/' + cls)
            for cld in child_dir:
                os.mkdir(out_dir + '/' + cls + '/' + cld)

        for cld in child_dir:
            mypath = src_dir + '/' + cld + '/' 
            all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            num_files = len(all_files)
            train = int(num_files  * train_size)
            val = int(num_files  * val_size)
            test = int(num_files * test_size)

            random.shuffle(all_files)
            trainlist = all_files[0:train]
            vallist = all_files[train:train+val]
            testlist = all_files[train+val:train+test+val]

            print('current class:', cld)
            print('Total images:', len(all_files))

            for f in trainlist:
                srcfile = mypath + f
                destfile = out_dir + '/train/' + cld + '/' + f
                shutil.move(srcfile, destfile)
            print('Training:', len(trainlist))

            for f in vallist:
                srcfile = mypath + f
                destfile = out_dir + '/val/' + cld + '/' + f
                shutil.move(srcfile, destfile)
            print('Validation:', len(vallist))

            for f in testlist:
                srcfile = mypath + f
                destfile = out_dir + '/test/' + cld + '/' + f
                shutil.move(srcfile, destfile)
            print('Testing:', len(testlist))

            print('-'*24)

        print('Complete!')

    except:
        print('image path already exists')
        pass

def apply_generator():
    print()
    train_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) .flow_from_directory(directory=TRAIN_DATA_DIR,
                     target_size=(IMG_WIDTH, IMG_HEIGHT),
                     classes=['lung_aca', 'lung_n', 'lung_scc'],
                     batch_size=BATCH_SIZE,
                     shuffle=True)

    validation_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) .flow_from_directory(directory=VALIDATION_DATA_DIR,
                         target_size=(IMG_WIDTH, IMG_HEIGHT),
                         classes=['lung_aca', 'lung_n', 'lung_scc'],
                         batch_size=BATCH_SIZE,
                         shuffle=True)
    
    return train_generator, validation_generator

def create_model():
    base_model = MobileNetV2(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    
    for layer in base_model.layers[:]:
        layer.trainable = False

    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)

    return Model(inputs=input, outputs=predictions)

def train_model(model, train_generator, validation_generator):
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(0.001),
                    metrics=['accuracy'])
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto', restore_best_weights=True)
    
    history = model.fit(train_generator,
                    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE),
                    epochs=100,
                    callbacks=[monitor],
                    validation_data=validation_generator,
                    validation_steps=math.ceil(float(VALIDATION_SAMPLES) / BATCH_SIZE)
                    )

    model.save('model_cancer500mobilenetV2.h5')

def evaluate_model(model):
    test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) .flow_from_directory(directory=TEST_DATA_DIR,
                         target_size=(IMG_WIDTH, IMG_HEIGHT),
                         classes=['lung_aca', 'lung_n', 'lung_scc'],
                         batch_size=BATCH_SIZE,
                         shuffle=False)

    predictions = model.predict(x=test_generator, verbose=1)

    y_true = test_generator.classes
    y_pred = np.argmax(predictions, axis=-1)

    print(confusion_matrix(y_true, y_pred))
    
    report = classification_report(y_true, y_pred)
    print(report)


if __name__ == "__main__":
    TRAIN_SIZE = float(input("Enter training size %: "))
    VAL_SIZE = float(input("Enter validation size %: "))
    TEST_SIZE = float(input("Enter test size %: "))
    
    TRAIN_DATA_DIR = 'split_image_sets/train/'
    VALIDATION_DATA_DIR = 'split_image_sets/val/'
    TEST_DATA_DIR = 'split_image_sets/test/'
    
    TRAIN_SAMPLES = int(5000  * TRAIN_SIZE)
    VALIDATION_SAMPLES = int(5000  * VAL_SIZE)
    TEST_SAMPLES = int(5000  * TEST_SIZE)
    
    NUM_CLASSES = 3
    IMG_WIDTH, IMG_HEIGHT = 224, 224
    BATCH_SIZE = 64

    split_datasets(TRAIN_SIZE, VAL_SIZE, TEST_SIZE)
    train_generator, validation_generator = apply_generator()
    new_model = create_model()
    train_model(new_model, train_generator, validation_generator)
    trained_model = load_model('model_cancer500mobilenetV2.h5')
    evaluate_model(trained_model)