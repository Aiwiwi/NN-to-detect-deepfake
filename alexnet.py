import  tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.regularizers import l2
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class AlexNet():
    def __init__(self, SHAPE, num_classes):
        super().__init__()
        self.model = tf.keras.Sequential()
        self.model.add(layers.Input(shape=SHAPE))
        self.model.add(layers.Conv2D(kernel_size=(11,11),strides=4, activation='relu',
        kernel_regularizer=l2(0.01),kernel_initializer='he_normal',filters=96, trainable=True))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.MaxPooling2D(pool_size=(3,3),strides=2, trainable=True))
        self.model.add(layers.Conv2D(kernel_size=(5,5),padding='same', activation='relu',
        kernel_regularizer=l2(0.01),kernel_initializer='he_normal',filters=256, trainable=True))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.MaxPooling2D(pool_size=(3,3),strides=2, trainable=True))
        self.model.add(layers.Conv2D(kernel_size=(3,3),padding='same', activation='relu',
        kernel_regularizer=l2(0.01),kernel_initializer='he_normal',filters=384, trainable=True))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Conv2D(kernel_size=(3,3),padding='same', activation='relu',
        kernel_regularizer=l2(0.01),kernel_initializer='he_normal',filters=384, trainable=True))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Conv2D(kernel_size=(3,3),padding='same', activation='relu',
        kernel_regularizer=l2(0.01),kernel_initializer='he_normal',filters=256, trainable=True))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.MaxPooling2D(pool_size=(3,3),strides=2, trainable=True))
        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.01), trainable=True))
        self.model.add(layers.BatchNormalization())        
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.01), trainable=True))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Dense(1000,activation='softmax', kernel_regularizer=l2(0.01), trainable=True))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256,activation='softmax',kernel_regularizer=l2(0.01), trainable=True))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5)) 

        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01)))

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                       loss='binary_crossentropy', metrics=['accuracy',  'f1_score','precision','recall'])
    def getmodel(self):
        return self.model
    

train_path = r'C:\DeepfakeDetection\Dataset\Train'
test_path = r'C:\DeepfakeDetection\Dataset\Test'

image_size=(227,227,3)
EPOCHS = 75
BATCH_SIZE = 32
n_classes = 2

a = AlexNet(image_size, n_classes)
model = a.getmodel()

train_generate = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')


test_generate = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_set = train_generate.flow_from_directory(train_path,
                                               target_size=(227,227),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')

test_set = test_generate.flow_from_directory(test_path,
                                             target_size=(227,227),
                                             batch_size=BATCH_SIZE,
                                             class_mode='categorical')

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
   patience=6,
   mode='min',
   min_delta=0.001,
   restore_best_weights=True
)


callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    early_stopping
]

model_history = model.fit(train_set,
                          validation_data = test_set,
                          epochs = EPOCHS,
                          steps_per_epoch=50,
                          validation_steps=32,
                          callbacks = callbacks,
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
plt.plot(epochs, loss, 'bo-', label='Потеря на обучающем наборе')
plt.plot(epochs, val_loss, 'ro-', label='Потеря на тестовом наборе')
plt.title('Потеря')
plt.xlabel('Эпохи')
plt.ylabel('Потеря')
plt.legend()
