#Import libraries and tools.
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas

from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tqdm import tqdm
import argparse

#Setting up cli code.

parser = argparse.ArgumentParser(
                    prog = 'SpottingDiffusion',
                    description = 'Train the SpottingDiffusion Model')

parser.add_argument('epochs', metavar='epochs', type=int, nargs=1,
                    help='how many epochs to train for.')

parser.add_argument('dropout', metavar='dropout', type=float, nargs=1,
                    help='the dropout rate of the model.')

parser.add_argument('learning_rate', metavar='learning_rate', type=float, nargs=1,
                    help='the learning rate for training of the model.')

args = parser.parse_args()

#Setting up deterministic mode within tensorflow.


import random 

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

os.environ['PYTHONHASHSEED']=str(123)
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

#@title ##Reviewing the dataset.

import os

base_dir = '/data'

real_dir = os.path.join(base_dir, 'laion400m-laion4.75/laion400m-laion4.75+/laion400m-laion4.75+')
stable_dir = os.path.join(base_dir, 'StableDiff/StableDiff')

real_files = os.listdir(real_dir)

stable_files = os.listdir(stable_dir)

#Delete corrupt images from the dataset.

def is_image(filename, verbose=False):

    data = open(filename,'rb').read(10)

    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True

    # check if file is PNG
    if data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
        if verbose == True:
             print(filename+" is: PNG.")
        return True

    # check if file is GIF
    if data[:6] in [b'\x47\x49\x46\x38\x37\x61', b'\x47\x49\x46\x38\x39\x61']:
        if verbose == True:
             print(filename+" is: GIF.")
        return True

    return False

print("Looking for corrupt files...")

# go through all files in desired folder
for filename in tqdm(os.listdir(real_dir)):
     # check if file is actually an image file
     if is_image(os.path.join(real_dir, filename), verbose=False) == False:
          # if the file is not valid, remove it
          os.remove(os.path.join(real_dir, filename))
          print("Invalid file!")

# go through all files in desired folder
for filename in tqdm(os.listdir(stable_dir)):
     # check if file is actually an image file
     if is_image(os.path.join(stable_dir, filename), verbose=False) == False:
          # if the file is not valid, remove it
          os.remove(os.path.join(stable_dir, filename))
          print("Invalid file!")
          # go through all files in desired folder

#Set up testing and training data.

train_data = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    labels='inferred',
    color_mode='rgb',
    shuffle=True,
    seed=123,
    validation_split=0.15,
    subset="training",
    interpolation='bilinear',
    follow_links=False,
    batch_size=batch_size,
    image_size=(256, 256)
)

val_data = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    labels='inferred',
    batch_size=batch_size,
    color_mode='rgb',
    shuffle=True,
    seed=123,
    validation_split=0.15,
    subset="validation",
    follow_links=False,
    image_size=(256, 256)
)

#Setting up Data Augmentations.


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(256, 256, 1)),
    layers.RandomRotation(0.05),
    layers.RandomBrightness(factor=0.1),
    layers.RandomZoom(0.1),
  ]
)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=(256, 256, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

#Add on Dropout and a Dense Layer with 256 nodes onto the base model.

dropout_amount = float(args.dropout)

x = data_augmentation(inputs)

x = base_model(inputs, training=False)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(dropout_amount)(x)
x = layers.Dense(256)(x)        
outputs = layers.Dense(2)(x)     

model = tf.keras.Model(inputs, outputs)

# Print the model summary.
model.summary()

#Compile model with Sparse Categorical Crossentropy loss.

lr_rate = float(args.learning_rate)

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.learning_rate)), metrics=['accuracy',])

#And finally, train the model for 4 epochs.

history = model.fit(train_data, epochs=args.epochs, validation_data = val_data, callbacks=[tensorboard_callback])

print("")
print("Training done! Saving model with name SpottingDiffusion.....")
print("")

model.save("SpottingDiffusion")
