import numpy as np

## TensorFlow is a deep learning library
## Here we use it to build and train the CNN
import tensorflow as tf 

## TensorFlow Keras is Keras which is included with TensorFlow simplifies the neural network creation. 
## We are importing data sets which are the module for loading popular data sets, including mNIST.
## We have layers which is used to add layers to our CNN
## We have models which use to define and compile a neural network model
from tensorflow.keras import datasets, layers, models

## two categorical converts labels to one hot encoded format, which is necessary for classification tasks.
from tensorflow.keras.utils import to_categorical

## To see this data in real time and see the image of which it predicted
import matplotlib.pyplot as plt

# 2: Load the MNIST Dataset
## Keras comes with built in datasets, including mNIST, making it very easy to load the data
## This function it will give us two different tuples which will have train images and train labels and test images and test labels
## train images are the training data and corresponding labels which are digits 0 to 9
## second tuple that we get is as I mentioned is test images and test labels. Now these are the testing data and labels for evaluating model accuracy.
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


# 3: Preprocess the Data
## Data needs to be normalized to improve the model's performance
## Preprocessing: normalize the pixel values to be between 0 and 1
## So what here normalization means we divide each pixel value by 255 to scale the values between 0 and 1, making the training faster and more stable
train_images = train_images / 255.0
test_images = test_images / 255.0

## Reshape images to (28, 28, 1) as they are grayscale
## CNN expects each image in 3D shape which is height, width and channels. Since mNIST images are grayscale, each has only one channel, and that's why 28 28 1 which represents height, width and one color channel
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

## Convert the labels to one hot encoded format. Now I we do this using the two categorical that we have imported, which converts labels Example 3 to 1 hot vectors. Example 000100000 for classifying 0 to 9. Now the purpose of this is this format is required for multi-class classification with categorical cross entropy loss.
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 4: Build the CNN model
## We'll use a simple feed forward neural network which is fully connected layers for classification
## Model dot sequential initializes a sequential model which allows layers to be stacked linearly
model = models.Sequential()

## Build the first convolution layer
## calling this conv 2D 30 233 activation which adds a 2D convolution convolutional layer where 32 is the number of filters to apply three by three is the size of each filters, which is three by three pixels, and activation equal to ReLU is the activation function applied to each pixel, setting negative values to zero, and then finally input shape 2828 one, which specifies the input dimension for the first layer.
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))

## Creates the first convolutional layer
## reduces the dimensions of the output by taking the maximum value in each two by two block, which decreases computational load and helps detect important features
model.add(layers.MaxPooling2D((2,2)))

## Creates the second convolutional layer
## The second convolutional layer first we call that conv 2d function which adds another convolutional layer with 64 filters
## The max pooling 2D reduces the output dimension by taking the maximum value in each two by two block
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

## Creates the third convolutional layer
## No pooling is applied after this layer, allowing it to like preserve the spatial dimensions for flattening, which means we are we have to flatten
model.add(layers.Conv2D(64, (3,3), activation='relu'))

## Flatten the 3D aoutput to 1D and add Dense layer
## flatten converts the 3D output of the previous layer to a 1D vector, enabling it to connect to fully connected dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

## Add a dense layer - output layer with 10 neurons (for 10 digit classes)
## This will create the final layer with ten units for each digit which is 0 to 9. That's why I've used the number ten. And then I'm using softmax which converts the output to probabilities, helping the model classify each digit.
model.add(layers.Dense(10, activation='softmax'))

## Compile the model
## specifying the Adam optimizer for adaptive learning. Then for loss I'm saying categorical and cross entropy which has loss function for multi-class classification with one hot encoded labels and metrics equal to accuracy which tracks the model accuracy during training and testing.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5: Train the model
## We'll train the model on the training set and validate it on the test set
## Trains the model on training data. We have passed the training images and training labels, which are the training inputs and outputs that we created epochs equal to five. Now this is the number of training epochs, meaning the model will see the entire training data set five times next batch size equal to 64 number of samples processed before updating any model parameters
## Validation data which it can validate against. We have passed the test images and test labels, which validate which is our validation data used to evaluate model performance during the training process
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# 6: Evaluate and Make Predictions
## Evaluate the model on test data
## This function evaluates model performance on test data returning loss and accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

## show test_loss and test_accuracy
print(f"Test accuracy: {test_acc * 100:.2f}%")

## Make Predictions on test images
## the predict test images function here uses the trained model to predict class probabilities for each test image, and the np.argmax finds the index with the highest probability in the first prediction, which corresponds to the predicted digit
predictions = model.predict(test_images)
print(f"repdiction for fourth test image: {np.argmax(predictions[4])}")

plt.imshow(test_images[4].reshape(28,28), cmap='gray')
plt.title(f"Predicted Label: {predictions[4].argmax()}")
plt.show()