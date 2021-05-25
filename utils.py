import numpy as np
from numba import jit
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers

@jit(nopython=True)
def normalize_map(imap):
    temp_map = np.zeros(imap.shape)
    for j in range(imap.shape[-1]):
        temp = (imap[...,j]-np.mean(imap[...,j]))/np.std(imap[...,j])
        temp_map[...,j] = temp
                                                        
    return temp_map

#A helper function that stores a bunch of models I've been playing around with. Notable Kosiba is a model following 
#Kosiba 2020 (https://arxiv.org/abs/2006.05998) who had good success training it on x-ray+ optical data
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
        model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
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

class feature_model:
    
    def __init__(self, train, comp, w_y = False):
        self.w_y = w_y
        self.train = clean_inputs(train)
        self.comp = clean_inputs(comp)
       
    def clean_inputs(data_tensor):
        flags = []
        for i in range(data_tensor.shape[0]):
            if np.any(np.isnan(data_tensor[i,...])):
                flags.append(i)
        data_tensor = np.delete(data_tensor, flags, axis = 0)

