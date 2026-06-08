# import libraries
import os

## matrix operation and mathematical function
import numpy as np

## TensorFlow is a deep learning library
## TensorFlow provides the tools for building and training neural networks
import tensorflow as tf 

## TensorFlow Keras is Keras which is included with TensorFlow simplifies the neural network creation. 
## We have layers which is used to add layers to our convolutional neural network (CNN)
## We have models which use to define and compile a neural network model
from tensorflow.keras import layers, models

## Image data generator is a class in Keras used for image data augmentation which improves the models ability to generalize by creating variations in the training data.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## We are going to use custom image
from tensorflow.keras.preprocessing import image

## Used for plotting the training and validation accuracy loss curves
import matplotlib.pyplot as plt

## Define paths to the data set and update them with the actual dataset location
## Dataset url - https://www.kaggle.com/datasets/tongpython/cat-and-dog
train_dir = 'data/training_set'
validation_dir = 'data/test_set'

## Define the iImageDataGenerators for data augmentation and rescaling
## Train data gen prepares the training data by applying data augmentation techniques such as rescaling rotation, zoom and horizontal flipping, so it helps the model generalize better by artificially creating new training examples.
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Rescale pixel values from (0-255) tp (0-1)
    rotation_range=40,              # Randomly rotate images for get better testing
    width_shift_range=0.2,          # Randomly shift images horizontally
    height_shift_range=0.2,         # Randomly shift image vertically
    shear_range=0.2,                # Randomly shear images
    zoom_range=0.2,                 # Randomly zoom in on images
    horizontal_flip=True,           # Randoly flip images horizontally
    fill_mode='nearest'             # Fill pixels that may have been lost after tranformation
)

## For the validation data we rescale. We don't have to do any data augmentation.
validation_datagen = ImageDataGenerator(rescale=1./255)

## Each sub directory within this train directory or validation directory should correspond to a particular class which is cats and dogs.
## Loading the training data and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),         # Resize all iages to 150x150
    batch_size=32,
    class_mode='binary',            # Binary classification (Dog or Cat)
)

## Create the validation generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),         # Resize all iages to 150x150   
    batch_size=32,
    class_mode='binary',            # Binary classification (Dog or Cat)
)

# Define a path to save the model
MODEL_PATH = 'cat_dog_model.keras'

# Check if a saved model exists
if os.path.exists(MODEL_PATH):
    # Load the existing model instead of training again
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    # Define the CNN model
    model = models.Sequential()
    
    # Create the first convolutional model/layer
    ## Conv2D creates convolutional layers to extract spatial features from the images
    ## First ayer has 32 filters and size 3 by 3, which is the kernel size for all the layers and the activation style
    ## We are using ReLU activation and input shape. Since we are converting that 150 by 150, we are specifying 150 by 150
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    ## Reduces the spatial dimensions of the feature maps by taking the maximum value in two by two blocks
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Create the second convolutional model/layer
    ## The second layer has 64 filters
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    ## Reduces the spatial dimensions of the feature maps by taking the maximum value in two by two blocks
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Create the third convolutional model/layer
    ## The third layer has 128 filters
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    ## Reduces the spatial dimensions of the feature maps by taking the maximum value in two by two blocks
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Create the third convolutional model/layer
    ## The third layer has 128 filters
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    ## Reduces the spatial dimensions of the feature maps by taking the maximum value in two by two blocks
    model.add(layers.MaxPooling2D((2, 2)))
    
    ## Flatten the output from the convolutional layer and add fully connected layers.
    model.add(layers.Flatten())
    ## The first dense layer has 512 units with ReLU activation.
    model.add(layers.Dense(512, activation='relu'))
    ## Final Dense layer which has only one unit with sigmoid activation to predict a binary outcome, which is cat or dog
    model.add(layers.Dense(1, activation='sigmoid')) # Output layer for binary classification

    # Compile the model
    ## Adam optimizer is used for efficient gradient based optimization
    ## loss='binary_crossentropy' function is suitable for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ## Print summary of the model
    model.summary()

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=50
    )
    
    ## Plot the training and validation accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    ## Training accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    ## Training loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

    # Save the model after training
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
## Test the model with the custom image
## This loads and pre-process a custom image and then makes a prediction using the trained model
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Load the image
    img_array = image.img_to_array(img)                 # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)       # Add batch dimension
    img_array /= 255.0                                  # Normalize the image (rescale pixel values to [0, 1])

    prediction = model.predict(img_array)               # Make the prediction

    if prediction[0] > 0.5:
        print(f"The image is predicted to be a Dog with a confidence of {prediction[0][0]:.2f}")
    else:
        print(f"The image is predicted to be a Cat with a confidence of {1 - prediction[0][0]:.2f}")

# Example: Test the classifier with a new image
predict_image(model, 'test-image-dog.jpg')
predict_image(model, 'test-image-cat.jpg')