import os
from pathlib import Path
from imutils import paths
import pandas as pd
import numpy as np
import random
import argparse
import random
import pickle

import matplotlib.pyplot as plt

# openCV
import cv2

# Tensor Flow
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

from tensorflow.keras import callbacks
from tensorflow.keras.applications import xception
from tensorflow.keras.models import Model

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from keras.optimizers import Adam
from keras_preprocessing.image import img_to_array

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


df_train = pd.read_csv("df_train.csv")
df_test = pd.read_csv("df_test.csv")
df_val = pd.read_csv("df_val.csv")

from keras.preprocessing.image import ImageDataGenerator

# 데이터 종류에 맞게 ImageDataGenerator 객체 생성
tr_gen = ImageDataGenerator(
    horizontal_flip=True,
    rescale=1 / 255.0,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
)
val_gen = ImageDataGenerator(rescale=1 / 255.0)
test_gen = ImageDataGenerator(rescale=1 / 255.0)
# columns=["tables", "sofas", "lamps", "chairs", "dressers", "beds",
columns = [
    "Asian",
    "Beach",
    "Contemporary",
    "Craftsman",
    "Eclectic",
    "Farmhouse",
    "Industrial",
    "Mediterranean",
    "Midcentury",
    "Modern",
    "Rustic",
    "Scandinavian",
    "Southwestern",
    "Traditional",
    "Transitional",
    "Tropical",
    "Victorian",
]
# 데이터 종류에 맞는 Pandas.DataFrame으로부터 Numpy Array Iterator 생성
tr_flow_gen = tr_gen.flow_from_dataframe(
    dataframe=df_train,
    x_col="file_name",
    y_col=columns,
    target_size=(244, 244),
    class_mode="raw",
    batch_size=64,
    shuffle=True,
)
val_flow_gen = val_gen.flow_from_dataframe(
    dataframe=df_val,
    x_col="file_name",
    y_col=columns,
    target_size=(244, 244),
    class_mode="raw",
    batch_size=64,
    shuffle=True,
)
test_flow_gen = test_gen.flow_from_dataframe(
    dataframe=df_test,
    x_col="file_name",
    y_col=columns,
    target_size=(244, 244),
    class_mode="raw",
    batch_size=64,
    shuffle=True,
)

from tensorflow.python.keras.models import Model

base_model = vgg16.VGG16(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(17, activation="softmax")(x)

model0 = Model(inputs=base_model.input, outputs=predictions)

model0.summary()

epo = 100
init_lr = 1e-3
bs = 512
image_dims = (224, 224, 3)

from tensorflow.python.keras.optimizers import adamax_v2

opt = adamax_v2.Adamax(learning_rate=0.001)

# opt = Adam(lr=init_lr)
model0.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

cp_callback = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="./model/model0-{epoch:02d}.h5", monitor="accuracy", verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir="./logs"),
]

# Train the model with the new callback

with tf.device("/device:GPU:0"):
    hist0 = model0.fit(
        tr_flow_gen,
        steps_per_epoch=len(df_train) / bs,
        epochs=epo,
        validation_data=val_flow_gen,
        validation_steps=len(df_val) / bs,
        callbacks=cp_callback,
        verbose=1,
    )  # Pass callback to training

import pickle

model0.save("./model/multilabel0")

with open("trainhist0", "wb") as file_pi:
    pickle.dump(hist0.history, file_pi)
