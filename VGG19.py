import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from PIL import Image
from numpy import array
from numpy import asarray
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.applications
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import plot_model
from keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_path = r'C:\DeepfakeDetection\Dataset\Train'
test_path = r'C:\DeepfakeDetection\Dataset\Test'


IMAGE_SIZE = [224,224,3]

vgg19 = VGG19(include_top = False,
            input_shape = IMAGE_SIZE,
            weights = 'imagenet')

vgg19.trainable = True
for layer in vgg19.layers:
    if layer.name in ['block4_conv1', 'block5_conv1']:
        layer.trainable = True
    else:
        layer.trainable = False

from keras.layers import Dropout

x = Flatten()(vgg19.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(2, activation='softmax')(x)

model_vgg = Model(inputs=vgg19.input, outputs=prediction)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
   patience=6,
   mode='min',
   min_delta=0.001,
   restore_best_weights=True
)

from tensorflow.keras.optimizers import Adam
adam=Adam(learning_rate=1e-4)

model_vgg.compile( loss = 'categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy',  'f1_score','precision','recall'] )

train_generate = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_generate = ImageDataGenerator(
    rescale=1./255
)


train_set = train_generate.flow_from_directory(train_path,
                                            target_size = ( 224 , 224 ),
                                            batch_size = 32,
                                            class_mode = 'categorical')

test_set = test_generate.flow_from_directory(test_path,
                                             target_size = ( 224 , 224 ),
                                            batch_size = 32,
                                            class_mode = 'categorical')

checkpoint = ModelCheckpoint(filepath='vgg19.h5', verbose=2, save_best_only=True)
callbacks = [checkpoint, early_stopping]

model_history = model_vgg.fit(train_set,
                          validation_data = test_set,
                          epochs = 50,
                          steps_per_epoch = 5,
                          callbacks = callbacks,
                          validation_steps = 32,
                          verbose = 1)


test_loss, test_accuracy, test_f1, test_precision, test_recall = model_vgg.evaluate(test_set)
print(f"Best Model - Test Loss: {test_loss}\t\nTest Accuracy: {test_accuracy}\t\nTest F1: {test_f1}\t\nTest precision:{test_precision}\t\nTest recall: {test_recall}")
accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(accuracy) + 1)



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'bo-', label='Точность на обучающем наборе')
plt.plot(epochs, val_accuracy, 'ro-', label='Точность на тестовом наборе')
plt.title('Точность от эпохи')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Потеря на обучающем наборе')
plt.plot(epochs, val_loss, 'ro-', label='Потеря на тестовом наборе')
plt.title('Потеря от эпохи')
plt.xlabel('Эпохи')
plt.ylabel('Потеря')
plt.legend()

plt.tight_layout()
plt.show()

