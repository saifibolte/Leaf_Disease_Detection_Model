import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow
from tensorflow.keras.utils import img_to_array, load_img
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19, preprocess_input , decode_predictions

# Data Analysis
#Calculating the classes
len(os.listdir('/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'))

#Data generate
train_datagen = ImageDataGenerator(zoom_range= 0.5, shear_range= 0.3,
                                   horizontal_flip= True,
                                   preprocessing_function= preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function= preprocess_input)

train = train_datagen.flow_from_directory(directory='/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
                                          target_size=(299,299),
                                          batch_size=64)
val = val_datagen.flow_from_directory(directory='/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
                                          target_size=(299,299),
                                          batch_size=64)
t_img, label = train.next()
t_img.shape

def plotImage(img_arr,label):
  for im, l in zip(img_arr,label):
    plt.imshow(im)
    plt.figure(figsize= (5,5))
    plt.show()

plotImage(t_img[:3], label[:3])

#Building Model
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras

base_model = VGG19(input_shape=(299,299,3), include_top=False)

for layer in base_model.layers:
  layer.trainable = False

base_model.summary()
X = Flatten()(base_model.output)
X= Dense(units= 38, activation= 'softmax')(X)

#Creating Model
model = Model(base_model.input, X)
model.summary()
model.compile(optimizer= 'adam', loss= keras.losses.categorical_crossentropy, metrics= ['accuracy'])

#Early Stopping and Model Check Point
from keras.callbacks import ModelCheckpoint, EarlyStopping

#early stopping
es = EarlyStopping(monitor= 'val_accuracy',
                   min_delta= 0.01,
                   patience= 3,
                   verbose=1)

#model check point
mc = ModelCheckpoint(filepath="best_model.h5",
                     monitor= 'val_accuracy',
                     min_delta= 0.01, patience= 3,
                     verbose=1,
                     save_best_only= True)

cb = [es,mc]

his = model.fit(train,
                steps_per_epoch= 64,
                epochs=50,
                verbose=1,
                callbacks= cb,
                validation_data= val,
                validation_steps= 64)

h = his.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c='red')
plt.title('accuracy vs val_accuracy')
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c='red')
plt.title('loss vs val_loss')
plt.show()

#load best model
from keras.models import load_model

model= load_model('/content/best_model.h5')
acc = model.evaluate(val)[1]
print(f"The accuracy of the model = {acc*100} %")

#Creating dictionaries of Classes
ref= dict(zip(list(train.class_indices.values()),list(train.class_indices.keys())))
for i in ref:
  print(ref[i])

def prediction(path):
  img= load_img(path, target_size=(299,299))
  i = img_to_array(img)
  im=preprocess_input(i)
  img= np.expand_dims(im, axis= 0)
  pred = np.argmax(model.predict(img))
  print(f'the image belongs to {ref[pred]}')

path=('/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Grape___Black_rot/004175d8-dc74-4285-8401-3cc9565730bb___FAM_B.Rot 0626.JPG')
prediction(path)
