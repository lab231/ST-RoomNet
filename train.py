import os
from xml.etree import ElementTree
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from spatial_transformer import ProjectiveTransformer, AffineTransformer
#from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import *
import random
import scipy.io
from sklearn.utils import shuffle

train_path = 'E:/LSUN2016_surface_relabel/surface_relabel/train/'
val_path = 'E:/LSUN2016_surface_relabel/surface_relabel/val/'

train_data = scipy.io.loadmat('training.mat')
train_data = train_data.get('training')[0]
val_data = scipy.io.loadmat('validation.mat')
val_data =  val_data.get('validation')[0]

print(train_data.shape)

train_data = shuffle(train_data)
batch_size = 48


def data_generator(data, path = train_path, batch_size=32, number_of_batches=None):
 counter = 0
 n_classes = 11

 #training parameters
 train_w , train_h = 400, 400
 while True:
  idx_start = batch_size * counter
  idx_end = batch_size * (counter + 1)
  x_batch = []
  y_seg_batch = []
  y_batch = []
  for file in data[idx_start:idx_end]:
   img_name = list(file)[0][0]
   img = cv2.imread(path+img_name+'.jpg')
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = cv2.resize(img, (train_w , train_h))
   seg = cv2.imread(path+img_name+'.png',0)
   seg = cv2.resize(seg, (train_w , train_h), interpolation= cv2.INTER_NEAREST)
   seg = seg / 51.0
   label = list(file)[2][0][0]
   #print(label)
   if label in [3,4,5,10]:
        seg[seg==2] = 1
   rand_hflip = random.randint(0, 1)
   rand_brightness = random.randint(0, 1)
   
   if (rand_hflip == 1 and path == train_path):
       img = cv2.flip(img, 1)
       seg = cv2.flip(seg, 1)
       seg_temp = np.array(seg, copy= True)
       #flip the right and left walls
       seg[seg_temp==2] = 3
       seg[seg_temp==3] = 2
   if (rand_brightness == 1 and path == train_path):
       val = random.uniform(-1, 1) * 30
       img = img + val
       img = np.clip(img, 0, 255)
       img = np.uint8(img)
    
   img = preprocess_input(img)
   x_batch.append(img) 
   y_seg_batch.append(seg)
  counter += 1
  x_train = np.array(x_batch)
  y_seg_train = np.array(y_seg_batch)
  yield x_train, y_seg_train
  if (counter == number_of_batches):
        counter = 0

ref_img = tf.io.read_file('ref_img2.png')
ref_img = tf.io.decode_png(ref_img)
ref_img = tf.cast(ref_img, tf.float32) / 51.0
ref_img = ref_img[tf.newaxis,...]
ref_img = tf.tile(ref_img, [batch_size,1,1,1])
print(ref_img.shape)

w= np.zeros((768, 8), dtype='float32')

b = np.zeros(8, dtype='float32')
b[0] = 1
b[4] = 1

base_model = ConvNext(include_top=False, weights="imagenet", input_shape= (400,400,3), pooling = 'avg')
theta = Dense(8, weights=[w, b])(base_model.output)
stl = ProjectiveTransformer((400,400)).transform(ref_img, theta)
model = Model(base_model.input, stl)

model.summary()
model.compile(optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay = 0.0001), loss = ['huber_loss'], metrics = ['accuracy'])

model.summary()

filepath="E:/LSUN2016_surface_relabel/surface_relabel/weights_stn/weights-improvement-{epoch:02d}-{loss:.4f}-{val_loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint]
n_train = 4000
n_valid= 394
model.fit(data_generator(train_data, train_path, batch_size, number_of_batches= n_train // batch_size),
            steps_per_epoch=max(1, n_train//batch_size), initial_epoch = 0,
            validation_data= data_generator(val_data, val_path, batch_size, number_of_batches= n_valid // batch_size),
            validation_steps=max(1, n_valid//batch_size),
            epochs=150,
            callbacks=callbacks_list)
