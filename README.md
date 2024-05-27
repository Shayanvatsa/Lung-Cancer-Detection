# Lung Cancer Detection Using CNN

This project aims to develop a convolutional neural network (CNN) for classifying lung cancer cases using chest X-ray images. The model leverages deep learning techniques to automate diagnosis, facilitating early detection and precise characterization of lung abnormalities. This README provides an overview of the project, its features, and instructions for setting up and running the code.

## Features

- **Data Augmentation:** Enhances the dataset by generating variations of the training images to improve model generalization.
- **Regularization:** Includes dropout layers to prevent overfitting by randomly setting a fraction of input units to zero during training.
- **Early Stopping:** Stops training when the model's performance stops improving, preventing overfitting.
- **Confusion Matrix:** Evaluates the model's performance by displaying the true positives, true negatives, false positives, and false negatives.
- **Model Checkpoint:** Saves the best model during training based on validation accuracy.
- **Training and Validation Accuracy/Loss Visualization:** Plots the accuracy and loss for both training and validation sets over epochs.

## Model Architecture

The model consists of several convolutional, pooling, dropout, and dense layers:

1. **Conv2D (128 filters, 3x3, ReLU):** Extracts features from the input image.
2. **AvgPool2D (2x2):** Reduces the spatial dimensions of the feature maps.
3. **Conv2D (128 filters, 3x3, ReLU):** Further extracts features.
4. **Conv2D (128 filters, 3x3, ReLU):** Extracts higher-level features.
5. **MaxPooling2D (2x2):** Reduces the spatial dimensions of the feature maps.
6. **Conv2D (128 filters, 3x3, ReLU):** Further extracts features.
7. **Conv2D (128 filters, 3x3, ReLU):** Extracts higher-level features.
8. **MaxPooling2D (2x2):** Reduces the spatial dimensions of the feature maps.
9. **Conv2D (64 filters, 3x3, ReLU):** Extracts features.
10. **Conv2D (64 filters, 3x3, ReLU):** Extracts higher-level features.
11. **MaxPooling2D (2x2):** Reduces the spatial dimensions of the feature maps.
12. **Flatten:** Converts the 2D feature maps into a 1D feature vector.
13. **Dropout (0.2):** Regularizes the model to prevent overfitting.
14. **Dense (3000 units, ReLU):** Fully connected layer to learn non-linear combinations of the flattened features.
15. **Dense (1500 units, ReLU):** Fully connected layer to further learn combinations of features.
16. **Dense (3 units, Softmax):** Output layer for classification into three categories (Benign, Malignant, Normal).

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/lung-cancer-detection.git
   cd lung-cancer-detection
   ```
##Training the Model

Load and preprocess the dataset:

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

## Define the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

## Load the dataset
### Ensure you have your dataset in an appropriate format
x_train, y_train = load_data('data/train')
x_val, y_val = load_data('data/val')
x_test, y_test = load_data('data/test')

## Fit the ImageDataGenerator on your training data
datagen.fit(x_train)

## Define the number of augmented images to generate per original image
augmented_images_per_original = 6

## Generate augmented images and append them to the training data
augmented_x_train = []
augmented_y_train = []
for i in range(len(x_train)):
    for _ in range(augmented_images_per_original):
        augmented_image = datagen.flow(np.expand_dims(x_train[i], axis=0), batch_size=1)[0][0]
        augmented_x_train.append(augmented_image)
        augmented_y_train.append(y_train[i])

## Convert lists to numpy arrays
augmented_x_train = np.array(augmented_x_train)
augmented_y_train = np.array(augmented_y_train)

## Concatenate original and augmented training data
x_train_augmented = np.concatenate((x_train, augmented_x_train), axis=0)
y_train_augmented = np.concatenate((y_train, augmented_y_train), axis=0)


## Define and compile the model:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPooling2D, Flatten, Dropout, Dense

model = Sequential([
    Conv2D(128, (3, 3), padding='same', input_shape=x_train_augmented.shape[1:], activation='relu'),
    AvgPool2D(2, 2),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.2, seed=12),
    Dense(3000, activation='relu'),
    Dense(1500, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

## Train the model with early stopping and model checkpoint:

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

history = model.fit(x_train_augmented, y_train_augmented, validation_data=(x_val, y_val), epochs=50,
                    callbacks=[early_stopping, model_checkpoint])

## Evaluate the model on the test set:

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

## Generate confusion matrix:

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant', 'Normal'], yticklabels=['Benign', 'Malignant', 'Normal'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

## Plot training and validation accuracy:

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

## Plot training and validation loss:

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()

## Limitations

- **No Frontend Interface:**
  * Currently, the project lacks a frontend interface, making it less user-friendly for healthcare professionals.
- **Stage Detection Not Implemented:**
  * The model does not identify the stage of lung cancer, such as stages I, II, III, or IV.
- **Limited Dataset:**
  * The model was trained on a relatively small dataset, which may not fully represent the variability in lung cancer images.
- **Generalization:**
  * The model's performance may vary when applied to images from different sources or with different characteristics than the training set.
- **Computational Resources:**
  * Training deep learning models requires significant computational power and time, which might not be accessible to all users.

## Future Work

- **Incorporating Clinical Data:**
  * Integrate additional clinical metadata, such as patient’s medical history and laboratory results, to improve diagnostic accuracy.
- **Real-time Deployment:**
  * Optimize the model for deployment in real-time clinical settings, such as hospital radiology departments or mobile health applications.
- **Validation and Regulatory Approval:**
  * Conduct validation studies and seek regulatory approval from healthcare authorities.
- **Continued Monitoring and Improvement:**
  * Continuously monitor model performance in real-world settings and iteratively improve the model based on feedback and new insights.
- **Frontend Interface Development:**
  * Build a user-friendly frontend interface to enhance accessibility and usability for healthcare professionals, including features like image upload, real-time predictions, visualization of model outputs, and interactive tools for analyzing diagnostic results.
- **Stage Detection:**
- * Implement functionality to identify the stage of lung cancer (I, II, III, or IV) to provide more detailed diagnostic information.
    
## License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

