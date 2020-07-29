# -*- coding: utf-8 -*-

from tensorflow.keras.applications import MobileNetV2, VGG16
#from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, BatchNormalization, MaxPooling2D, ZeroPadding2D, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

train_directory = 'dataset1/train'
validation_directory = 'dataset1/val'

def mobilenet_model(img_shape = (224,224,3)):
    """
    Downloads mobilenet model without imagenet weights and adds the last layers 
    for classification.

    Parameters
    ----------
    img_shape : tuple, optional
        DESCRIPTION. The default is (224,224,3).
        Shape of the input image for the model. Default is 224,224,3 as it is the
        default for MobileNet.
    Returns
    -------
    model : keras model
        Returns MobileNet model with classification layers added.

    """
    base_model = MobileNetV2(input_shape = img_shape, include_top = False, weights = None)
    
    MN = base_model.output
    
    MN = GlobalAveragePooling2D()(MN)
    MN = Dense(512, activation = 'relu')(MN) 
    MN = Dropout(0.5)(MN)
    MN = Dense(256, activation = 'relu')(MN)
    MN = Dropout(0.5)(MN)
    MN = Dense(28, activation = 'softmax')(MN)
    
    model = Model(base_model.input, MN)
    model.compile(optimizer = RMSprop(lr = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy', Precision(), Recall()])
    model.summary()
    return model

def vgg_model(img_shape = (224,224,3)):
    
    """
    Downloads mobilenet model without imagenet weights and adds the last layers
    for classification.

    Parameters
    ----------
    img_shape : tuple, optional
        DESCRIPTION. The default is (224,224,3).
        Shape of the input image for the model. Default is 224,224,3 as it is the
        default for MobileNet.
    Returns
    -------
    model : keras model
        Returns MobileNet model with classification layers added.

    """
    base_model = VGG16(input_shape = img_shape, include_top = False, weights = None)
    
    MN = base_model.output
    
    MN = Flatten()(MN)
    MN = Dense(4096, activation = 'relu')(MN)
    MN = Dropout(0.5)(MN)
    MN = Dense(4096, activation = 'relu')(MN)
    MN = Dropout(0.5)(MN)
    MN = Dense(28, activation = 'softmax')(MN)
    
    model = Model(base_model.input, MN) 
    model.compile(optimizer = Adam(lr = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy', Precision(), Recall()])
    model.summary()
    return model

def alexnet():
    model = Sequential()
    
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11),\
     strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(227*227*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(28))
    model.add(Activation('softmax'))
    model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model

def custom_model(input_shape = (50,50,3)):
	model = Sequential()
	model.add(Conv2D(16, (2,2), input_shape = input_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
	model.add(Conv2D(64, (5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(28, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr = 1e-2), metrics=['accuracy'])
	return model

train_gen = ImageDataGenerator(samplewise_center = True,
                               samplewise_std_normalization = True,
                               preprocessing_function=preprocess_input
                               )

validation_gen = ImageDataGenerator(samplewise_center = True,
                               samplewise_std_normalization = True,
                               preprocessing_function=preprocess_input
                               )


train_generator = train_gen.flow_from_directory(train_directory, target_size = (227,224), class_mode='categorical', batch_size = 16, shuffle = True)
validation_generator = validation_gen.flow_from_directory(validation_directory, target_size = (227,224), class_mode='categorical', batch_size = 16, shuffle = True)
callbacks = [ReduceLROnPlateau(patience=2), ModelCheckpoint(filepath = 'final_models/mobilenet_adam_lr001.h5', save_best_only = True)]
model = mobilenet_model()
model.summary()
history = model.fit_generator(train_generator, validation_data = validation_generator, epochs = 10, verbose = 1, callbacks=callbacks)
model.save('models/mobilenet_adam_lr001.h5')

from matplotlib import pyplot
pyplot.plot(range(10), history.history["loss"], 'r', label = "Training loss")
pyplot.plot(range(10), history.history["val_loss"], 'g', label = "Validation loss")
pyplot.xlabel("Epochs")
pyplot.ylabel("Loss")
pyplot.title("Training and Validation Loss")
pyplot.legend(loc = "upper right")
pyplot.grid(b = True)
pyplot.savefig("figures/loss_lr001_adam_mobilenet.png")
pyplot.show()

pyplot.plot(range(10), history.history["accuracy"], 'r', label = "Training accuracy")
pyplot.plot(range(10), history.history["val_accuracy"], 'g', label ="Validation accuracy")
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracy")

pyplot.title("Training and Validation Accuracy")
pyplot.legend(loc = "lower right")
pyplot.grid(b = True)
pyplot.savefig("figures/acc_lr001_adam_mobilenet.png")

pyplot.show()


