import pickle as pk
import tensorflow as tf
import numpy as np
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import os, sys, pathlib, argparse, h5py

from models import make_model

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
'''
parser = argparse.ArgumentParser()

parser.add_argument("idir")
parser.add_argument("-y", "--withy", default=True)
parser.add_argument("-n", "--numcut", type=int, default=1000)

args = parser.parse_args()

data_dir = str(args.idir)

data_dir =str(data_dir)

print('loading data')

pos_im_act = []
pos_im_des = []
neg_im = []

try:
    cut = int(args.numcut)
except:
    sys.exit('Error: cut must be convertable into int')

w_y = str(args.withy)

if w_y != 'False' and w_y != 'True':
    sys.exit('Error: with y argument must be True or False')


for directory in os.listdir(data_dir):
    print(directory)
    if directory[:3] == 'act' and (int(directory[4:8]) < cut):
        h5f = h5py.File(data_dir+directory)
        pos_im_act.append(h5f['act'][:])
    elif directory[:3] == 'des' and (int(directory[4:8]) < cut):
        h5f = h5py.File(data_dir+directory)
        pos_im_des.append(h5f['des'][:])
    elif directory[:6]=='random' and (int(directory[7:11]) < cut):
        h5f = h5py.File(data_dir+directory)
        neg_im.append(h5f['random'][:])


pos_im_act = np.vstack(pos_im_act)
pos_im_des = np.vstack(pos_im_des)
neg_im = np.vstack(neg_im)


pos_im = np.concatenate((pos_im_act, pos_im_des))


flags = []
for i in range(pos_im.shape[0]):
        if np.any(np.isnan(pos_im[i,...])):
                flags.append(i)

pos_im = np.delete(pos_im, flags, axis = 0)

flags = []
for i in range(neg_im.shape[0]):
        if np.any(np.isnan(neg_im[i,...])):
                flags.append(i)

neg_im = np.delete(neg_im, flags, axis = 0)


neg_im = neg_im[:len(pos_im)]
pos_im = pos_im[:len(neg_im)]

print(len(pos_im), len(neg_im))





#If not fitting with y, only use the DES data, which is the first 5 channels. 
if w_y == 'False':
    pos_im, neg_im = pos_im[...,:5], neg_im[...,:5]

print('Fitting with y = ', w_y)

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

#model = make_model('simple_cnn', input_shape = input_shape, degree = 4)
model = make_model('test', input_shape = input_shape)

data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360, width_shift_range=4,
    height_shift_range=4,zoom_range=0.3)

if w_y == 'False':
        checkpoint_path = "/scratch/r/rbond/jorlo/ml-clusters/models/redmapper_wo_y.ckpt"

else:
        checkpoint_path = "/scratch/r/rbond/jorlo/ml-clusters/models/redmapper_w_y.ckpt"


checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


history = model.fit(data_augmentation.flow(train_images, train_labels), epochs=int(20),
                    validation_data=val_dataset, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

print('\nTest accuracy: ', test_acc)



