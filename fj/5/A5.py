'''
For the fifth question, especially due to the excessive number of images in the data set, 
the difficulty of manual annotation, and the long training time of the model, this paper establishes a new model to replace yolo for apple recognition. 
Two models, CNN and random forest, were used for training. After image pre-processing, the CNN model creates labels, 
divides the data set, and sets three convolution layers to increase the depth of the model and nonlinear feature extraction capability,
Dropout regularization reduces overfitting, and Adam optimizer is used as the optimization algorithm of the model. After the training is completed,
the model is evaluated on the test set, and the apple ID is screened and saved. The model has good generalization ability and high accuracy. 
In the random forest model, the color histogram of the three channels of the image is calculated as the fruit feature, and the number of trees is specified as 100,
and the performance index of the model is also output.
'''
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import glob
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define a function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(f'Attachment 2/{folder}/*.jpg'):
        img = cv2.imread(filename)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images

apple_images = load_images_from_folder('Apple')

len(apple_images), apple_images[0].shape

other_fruits = ['Carambola', 'Pear', 'Plum', 'Tomatoes']
other_fruits_images = []
for fruit in other_fruits:
    images = load_images_from_folder(fruit)
    other_fruits_images.extend(images)

# Create labels, where apple is 1 and others are 0
labels = np.concatenate([np.ones(len(apple_images)), np.zeros(len(other_fruits_images))])

# Combine images and labels
images = np.array(apple_images + other_fruits_images)

# One-hot encode the labels
labels = to_categorical(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

from tensorflow.keras.models import load_model

# Load the model
model = load_model('fruit_classification_model.h5')
# Save the model
# model.save('fruit_classification_model.h5')

# Evaluate the model on the testing set
_, accuracy = model.evaluate(X_test, y_test)
print("Model Accuracy:", accuracy)

# Calculate the confusion matrix on the testing set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix-CNN')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# Load and preprocess the test images
dataset3_path='Attachment 3'
test_images = []
test_image_ids = []
for filename in sorted(glob.glob(f'{dataset3_path}/*.jpg')):
    img = cv2.imread(filename)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_images.append(img)
        test_image_ids.append(os.path.basename(filename).split('.')[0])

test_images = np.array(test_images) / 255.0  # Normalize

# Use the CNN model to make predictions on the test images
test_predictions = model.predict(test_images)
test_predictions = np.argmax(test_predictions, axis=1)

# Save the IDs of the images recognized as apples to an Excel file
apple_ids = [id for id, pred in zip(test_image_ids, test_predictions) if pred == 1]

df = pd.DataFrame(apple_ids, columns=['Apple_ID'])
df.to_excel('apple_ids_cnn.xlsx', index=False)

from sklearn.metrics import classification_report

# Output precision, recall and F1 score of the confusion matrix
print(classification_report(y_true, y_pred))

from sklearn.metrics import precision_recall_curve

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_true, y_pred)
print(precision)
print( recall)

# Plot the PR curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve-CNN')
plt.show()

'''Random Forest Model'''
import zipfile
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import cv2
import glob
import  pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define a function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(f'Attachment 2/{folder}/*.jpg'):
        img = cv2.imread(filename)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images

apple_images = load_images_from_folder('Apple')
len(apple_images), apple_images[0].shape

# Define a function to compute the color histogram of an image
def compute_histogram(image, bins=8):
    hist = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(3)]
    hist = np.concatenate(hist)
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Compute features for apple images
apple_features = [compute_histogram(image) for image in apple_images]

# Show samples of apple images
sample_images = apple_images[:6]

plt.figure(figsize=(15, 3))
for i, img in enumerate(sample_images, 1):
    plt.subplot(1, 6, i)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle('Sample of Apple')
plt.show()

sample_image = sample_images[0]  
color = ('r', 'g', 'b')
plt.figure(figsize=(8, 6))
for i, col in enumerate(color):
    hist = cv2.calcHist([sample_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

other_fruits = ['Carambola', 'Pear', 'Plum', 'Tomatoes']
other_fruits_features = []
for fruit in other_fruits:
    images = load_images_from_folder(fruit)
    features = [compute_histogram(image) for image in images]
    other_fruits_features.extend(features)

for fruit in other_fruits:
    fruit_images = load_images_from_folder(fruit)
    sample_images = fruit_images[:6]
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(sample_images, 1):
        plt.subplot(1, 6, i)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f'Sample of {fruit}')
    plt.show()
    sample_image = sample_images[0]  
    color = ('r', 'g', 'b')
    plt.figure(figsize=(8, 6))
    for i, col in enumerate(color):
        hist = cv2.calcHist([sample_image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title(f'{fruit} Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Create labels, 1 for apple and 0 for others
labels = np.concatenate([np.ones(len(apple_features)), np.zeros(len(other_fruits_features))])

# Concatenate features and labels
features = np.array(apple_features + other_fruits_features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Create and train the random forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy, report)

# Confusion matrix on the testing set
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

from sklearn.metrics import roc_curve, auc

# Compute parameters for ROC curve
y_pred_proba = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.title('ROC')
plt.legend(loc='lower right')
plt.show()

dataset3_path = "Attachment 3"
test_images = []
test_image_ids = []
for filename in sorted(glob.glob(f'{dataset3_path}/*.jpg')):
    img = cv2.imread(filename)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_images.append(img)
        test_image_ids.append(os.path.basename(filename).split('.')[0])

# Compute features for test images
test_features = [compute_histogram(image) for image in test_images]

# Make predictions using the model
test_predictions = classifier.predict(test_features)

# Compute accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
apple_ids = [id for id, pred in zip(test_image_ids, test_predictions) if pred == 1]

# Create a DataFrame containing apple IDs
df = pd.DataFrame(apple_ids, columns=['Apple_ID'])

# Save the DataFrame to an Excel file
df.to_excel('apple_ids.xlsx', index=False)