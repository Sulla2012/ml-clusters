print('Importing modules')
import pickle as pk
import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt
from tensorflow.keras import datasets, layers, models
#import tensorflow_datasets as tfds
#from astropy.nddata import block_reduce
#from astropy.convolution import Gaussian2DKernel, convolve
#import pandas as pd
#from sklearn.metrics import confusion_matrix
from tensorflow.keras import regularizers
import os, sys
import pathlib


def make_model(model_type = 'simple_cnn', input_shape = (342,342,5), degree = 5):

    if model_type == 'simple_cnn':
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2))
    elif model_type == 'resnet':
        model = tf.keras.applications.ResNet50(include_top=True,
        weights= None,
        input_shape = input_shape,
        classes = 2)
    elif model_type == 'pretrained':
        model = tf.keras.applications.ResNet50(include_top=False,
        weights= 'imagenet',
        input_shape = input_shape_rgb,
        classes = 2)
        
    elif model_type == 'inception_v2':
        base_model = tf.keras.applications.InceptionResNetV2(include_top = False,
                input_shape = input_shape,
                weights= 'imagenet'
                )
        base_model.trainable = False


    elif model_type == 'hyper':
        def train_test_model(hparams):
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(hparams[HP_DROPOUT]))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(hparams[HP_NUM_UNITS], activation='relu'))
            model.add(layers.Dense(2))
            model.compile(
              optimizer=hparams[HP_OPTIMIZER],
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
          )

            model.fit(train_dataset, epochs=10) 
            _, accuracy = model.evaluate(test_dataset)
            return accuracy


    elif model_type == 'test':
        model = models.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', 
               input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(2)])

    elif model_type == 'deep_few_filters':
        model = models.Sequential()
        #model.add(layers.experimental.preprocessing.Rescaling(1./all_max, input_shape=input_shape))
        model.add(layers.Conv2D(2, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        j = 0
        for j in range(degree):
            model.add(layers.Conv2D(4, (3, 3), activation='relu'))
            model.add(layers.ZeroPadding2D((1,1)))
            #model.add(layers.MaxPooling2D((2, 2)))
            j +=1
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(2))
        
    elif model_type == 'dropout':
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.0001), activation='relu', input_shape=input_shape))
        layers.Dropout(0.5),
        model.add(layers.AveragePooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
        layers.Dropout(0.5),
        model.add(layers.AveragePooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
        layers.Dropout(0.5),
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2))

    
    elif model_type == 'kosiba':
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
        for i in range(degree):
            model.add(layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        layers.Dropout(0.65)

        model.add(layers.Dense(2))
    model.summary()
    if model_type == 'kosiba':
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
              0.0001,
              decay_steps=348/10*1000,
              decay_rate=1e-6,
              staircase=False)
        opt = tf.keras.optimizers.SGD(
                learning_rate=0.01, momentum=0.9, nesterov = True, name='SGD')
        model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    else:
        model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model


data_dir = str(sys.argv[1])

data_dir =str(data_dir)

print('loading data')
pos_im = pk.load(open(data_dir+'large-redmapper/large-redmapper_w_y.pk', 'rb'))
neg_im = pk.load(open(data_dir+'large-randoms/large-randoms_w_y.pk', 'rb'))


w_y = str(sys.argv[2])

if w_y == 'False':
	print('No y')
	pos_im, neg_im = pos_im[...,:5], neg_im[...,:5]

tot = min(pos_im.shape[0], neg_im.shape[0])
train_size = int(0.7 * tot)
val_size = int(0.15 * tot)
test_size = int(0.15 * tot)
    
train_pos = pos_im[:train_size]
val_pos = pos_im[train_size:train_size + val_size]
test_pos = pos_im[train_size + val_size:]
    
train_neg = neg_im[:train_size]
val_neg = neg_im[train_size:train_size + val_size]
test_neg = neg_im[train_size + val_size:]

input_shape = train_pos.shape[1:]

train_images = np.concatenate((train_pos,train_neg))
val_images = np.concatenate((val_pos,val_neg))
test_images = np.concatenate((test_pos,test_neg))

train_labels = np.array(train_pos.shape[0]*[1] + train_neg.shape[0]*[0])
val_labels = np.array(val_pos.shape[0]*[1] + val_neg.shape[0]*[0])
test_labels = np.array(test_pos.shape[0]*[1] + test_neg.shape[0]*[0])

batch_size = 10

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

model = make_model('kosiba', input_shape = input_shape, degree = 4)


data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360, width_shift_range=4,
    height_shift_range=4,zoom_range=0.3)


checkpoint_path = "/scratch/r/rbond/jorlo/ml-clusters/models/large_redmapper_w_y.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


history = model.fit(data_augmentation.flow(train_images, train_labels), epochs=int(2e3), 
                    validation_data=val_dataset, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
