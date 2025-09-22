import tensorflow as tf
from tensorflow.keras import optimizers, layers, models, regularizers
import keras_tuner as kt
from tensorboard.plugins.hparams import api as tf_hp

from tensorflow import keras

def get_optimizer():
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.001,
      decay_steps=20000,
      decay_rate=1,
      staircase=False)
    return tf.keras.optimizers.Adam(lr_schedule)

def model_builder(hp):
    model = models.Sequential()
    l1_hp = hp.Int('l1_size', min_value = 3, max_value = 7, step =2)
    model.add(layers.Conv2D(16, l1_hp, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    d1_hp = hp.Float('dropout1', min_value = 0, max_value = 0.5)
    model.add(layers.Dropout(d1_hp))
    l2_hp = hp.Int('l2_size', min_value = 3, max_value = 7, step =2) 
    model.add(layers.Conv2D(32, l2_hp, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    l3_hp = hp.Int('l3_size', min_value = 3, max_value = 7, step =2)
    model.add(layers.Conv2D(64, l3_hp, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    d2_hp = hp.Float('dropout2', min_value = 0, max_value = 0.5)
    model.add(layers.Dropout(d2_hp))
    model.add(layers.Flatten())
    hp_units = hp.Int('dense_size', min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units, activation='relu'))
    model.add(layers.Dense(2))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

    return model

def make_model(model_type = 'simple_cnn', optimizer = None, kernel_regularizer = None, strategy = None, input_shape = (342,342,5), degree = 5):

    '''
    A function which creats a CNN model based off the TF model class from a number of different preset types
    It also intializes the model, which involves setting hyper parameters, regularization and the like.
    This all can, and often is, done explictly in a script but these are common enough that it's nice to be able 
    to just call them.

    Inputs:
        model_type: str, a keywork which selects from the predefined models
        input_shape: array of ints, TF models have to know how big the input is
        degree: int, not used by all models. Sets the depth for some CNN modes with variable depth

    Outputs:
        model: A TF model class with features as specified by the model_type, with initialization performed
    '''

    if strategy is None:
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

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
        with strategy.scope():
            model = models.Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer= kernel_regularizer, input_shape=input_shape),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer = kernel_regularizer),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer = kernel_regularizer),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_regularizer = kernel_regularizer),
            layers.Dense(2)])

    elif model_type == 'DeepShadows':
        model = models.Sequential([
            layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.13), input_shape = input_shape),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'),
            layers.Dropout(rate=0.4),
            layers.Conv2D(filters=2*16, kernel_size=(3,3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.13)),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'),
            layers.Dropout(rate=0.4),
            layers.Conv2D(filters=2*32, kernel_size=(3,3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.13)),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'),
            layers.Dropout(rate=0.4),
            layers.Flatten(),
            layers.Dense(units=1024, activation='relu',kernel_regularizer=regularizers.l2(0.12)),
            layers.Dense(units=1, activation='sigmoid')])



    elif model_type == 'test-no-dist':
        model = models.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer= kernel_regularizer, input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer = kernel_regularizer),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer = kernel_regularizer),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer = kernel_regularizer),
        layers.Dense(2)])


    elif model_type == 'test_hyper': 
        tuner = kt.BayesianOptimization(model_builder,
                     max_trials = 100,
                     objective='val_accuracy', 
                     directory='/scratch/r/rbond/jorlo/ml-clusters/results',
                     project_name='hyper_parameters')

        return tuner

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

    elif model_type == 'DeepShadows':
        model.compile(optimizer=optimizers.Adadelta(0.1),
              loss= 'binary_crossentropy',
              metrics=['accuracy'])
    else:
        if optimizer is None:
            optimizer = get_optimizer()
        model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model



