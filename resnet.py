import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.applications
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import plot_model
from keras.layers import Input,Dense,Flatten, Dropout
from tensorflow.keras.models import Model
from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import layers
from tensorflow.keras.applications import ResNet50V2
from sklearn.preprocessing import normalize
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


train_path = r'C:\DeepfakeDetection\Dataset\Train'
test_path = r'C:\DeepfakeDetection\Dataset\Test'


IMAGE_SIZE = [256,256,3]

early_stopping = EarlyStopping(
    monitor='val_loss',
   patience=6,
   mode='min',
   min_delta=0.001,
   restore_best_weights=True
)


ResNet = ResNet50V2(
    include_top=False,
    weights = 'imagenet',
    input_shape=IMAGE_SIZE,
)


for layer in ResNet.layers:
    layer.trainable = True

x = Flatten()(ResNet.output)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=ResNet.input, outputs=prediction)


model.compile( loss = 'categorical_crossentropy',
              optimizer = Adam(learning_rate = 1e-4),
              metrics = ['accuracy',  'f1_score','precision','recall'] )



train_generate = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 30,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = 'nearest')

test_generate = ImageDataGenerator(rescale=1./255)

EPOCHS=50
BATCH_SIZE=32

train_set = train_generate.flow_from_directory(train_path,
                                            target_size = (256, 256),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'categorical')


test_set = test_generate.flow_from_directory(test_path,
                                             target_size = (256, 256),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'categorical')


checkpoint = ModelCheckpoint(filepath='ResNet.h5', verbose=2, save_best_only=True)
callbacks = [checkpoint]


model_history = model.fit(train_set,
                          validation_data = test_set,
                          epochs = EPOCHS,
                          steps_per_epoch = 5,
                          callbacks = callbacks,
                          validation_steps = 32,
                          verbose = 1)


test_loss, test_accuracy, test_f1, test_precision, test_recall = model.evaluate(test_set)
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
plt.title('Точность')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Потеря  на обучающем наборе')
plt.plot(epochs, val_loss, 'ro-', label='Потеря на тестовом наборе')
plt.title('Потеря')
plt.xlabel('Эпохи')
plt.ylabel('Потеря')
plt.legend()

plt.tight_layout()
plt.show()
