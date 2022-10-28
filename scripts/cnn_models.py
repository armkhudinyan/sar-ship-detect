from typing import Tuple
#Import Keras.
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D, Dense, Dropout, Flatten, Activation
# from keras.optimizers import Adam, SGD, RMSprop 
from keras.callbacks import ModelCheckpoint, EarlyStopping


def CONV2D_PixelWise(input_shape: Tuple[int,int,int]):
    '''Input shape should have 3 dimensions'''
    model = Sequential()
    # conv block 1
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', 
                    input_shape=input_shape))
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2))) 
    model.add(Dropout(0.1))
    # conv block 2
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2))) # image size: 80x80 => 40x40
    model.add(Dropout(0.1))    
    # conv block 3
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    # conv block 4
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    # reshape the tensor vector indo NxN for passing into Conv2D
    #model.add(Reshape((20,20), input_shape=(12,)))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Dropout(0.1))    
    # conv block 5
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Dropout(0.1))
    # output Layer
    model.add(Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same'))  
    # compile model
    #mypotim = SGD(lr=0.01, momentum=0.9)
    #mypotim = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    mypotim = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', #'sparse_categorical_crossentropy',  
                  optimizer=mypotim,
                  metrics=['accuracy'])                  
    model.summary()
    return model


def CONV2D_3Classes(input_shape):
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=input_shape)) 
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())
    #Dense Layers
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))
    #Dense Layer 2
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))
    #Sigmoid Layer
    #gmodel.add(Dense(1))
    #gmodel.add(Activation('sigmoid'))
    
    #Output Layer
    gmodel.add(Dense(3))               # set 3 classes for the last output layer
    gmodel.add(Activation('softmax'))  # 'softmax' as activ. func. for multiclass classif
    
    # compile model
    #mypotim = SGD(lr=0.01, momentum=0.9)
    mypotim=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='sparse_categorical_crossentropy',  
                  optimizer=mypotim,
                  metrics=['accuracy'])  # f1_score to be implemented (also ROC-AUC)
    gmodel.summary()
    return gmodel


def CONV2D_Binary(input_shape: Tuple[int,int,int]):
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=input_shape)) 
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())
    #Dense Layer 1
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))
    #Dense Layer 2
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    # Sigmoid Layer
    # gmodel.add(Dense(1))
    # gmodel.add(Activation('sigmoid'))
    
    #Output Layer
    gmodel.add(Dense(2))               # set 2 classes for the last output layer
    gmodel.add(Activation('softmax'))  # 'softmax' as activ. func. for multiclass classif
    
    # compile model
    #mypotim = SGD(lr=0.01, momentum=0.9)
    # mypotim=tf.keras.optimizers.Adam(learning_rate=0.0006, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # gmodel.compile(loss= 'binary_crossentropy', #sparse_categorical_crossentropy', #'binary_crossentropy',  
    #               optimizer=mypotim,
    #               metrics=['accuracy'])  # f1_score to be implemented (also ROC-AUC)

    mypotim=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel


def CONV2D_Binary2(input_shape: Tuple[int,int,int]):
    gmodel=Sequential()
    #Conv Layer 0
    gmodel.add(Conv2D(64, kernel_size=(1, 1),activation='relu', input_shape=input_shape)) 
    gmodel.add(MaxPooling2D(pool_size=(1, 1)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 1   
    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu')) 
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(MaxPooling2D(pool_size=(1, 1)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())
    #Dense Layer 1
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))
    #Dense Layer 2
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    # Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))
    
    # #Output Layer
    # gmodel.add(Dense(2))               # set 3 classes for the last output layer
    # gmodel.add(Activation('softmax'))  # 'softmax' as activ. func. for multiclass classif
    
    # compile model
    #mypotim = SGD(lr=0.01, momentum=0.9)
    mypotim=tf.keras.optimizers.Adam(learning_rate=0.0006, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss= 'binary_crossentropy', #sparse_categorical_crossentropy', #'binary_crossentropy',  
                  optimizer=mypotim,
                  metrics=['accuracy'])  # f1_score to be implemented (also ROC-AUC)
    gmodel.summary()
    return gmodel

def get_callbacks(filepath, patience=10):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]