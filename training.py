##importing necessary modules
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from keras.callbacks import ModelCheckpoint

## model architecture
model =Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(), ## Flattens the output from convolution layers into 1D vector
    Dropout(0.5), ## Dropout layer to reduce overfitting 0.5 is a dropout ratio

    ## two fully connected layers
    Dense(50, activation='relu'), ##helps to change the dimentionality of the output
    Dense(2, activation='softmax') ##softmax activation is used for class probabilities
])

## model compilation using Adam optimizer, binary entropy loss function and accuracy as evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

## Loading the training data
Training_Directory = "C:/Users/eTraders/Downloads/face-mask-dataset/Dataset/train/train"

## Augmenting the training data using ImageDataGenerator class from keras

## Creating instance of ImageDataGenerator and specifying the desired data augmentation options
Train_Datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

## flow_from_directory: creates a generator to load batches from dir on disk and applies 
#                       the specified data augemtation options to each image in the batch
train_generator = Train_Datagen.flow_from_directory(Training_Directory, 
                                                    batch_size=10, 
                                                    target_size=(150, 150))

## Loading the validation data and applying same steps as above
Validation_Directory = "C:/Users/eTraders/Downloads/face-mask-dataset/Dataset/test/test"
Validation_Datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = Validation_Datagen.flow_from_directory(Validation_Directory, 
                                                         batch_size=10, 
                                                         target_size=(150, 150))

## Creating model chechpoint for saving the model weights
checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

## fit called with train_generator object: generates batches of augmented image data on the fly and 
#                                          feeds them to the model for training
history = model.fit(train_generator,
                              epochs=10,
                              validation_data=validation_generator,
                              callbacks=[checkpoint])