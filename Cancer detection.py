#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageEnhance
#keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.utils import shuffle


# In[2]:


train_dir= "C:\\Users\\saumi\\Desktop\\Cancer detection project\\MRI images\\Training"
test_dir= "C:\\Users\\saumi\\Desktop\\Cancer detection project\\MRI images\\Testing"

#load and shuffle train data
train_paths = []
train_labels = []

#for training
for label in os.listdir(train_dir):
    for image in os.listdir(os.path.join(train_dir, label)):
        train_paths.append(os.path.join(train_dir,label, image))
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels) #prevent overfitting

#for testing
test_paths = []
test_labels = []

for label in os.listdir(test_dir):
    for image in os.listdir(os.path.join(test_dir,label)):
        test_paths.append(os.path.join(os.path.join(test_dir,label,image)))
        test_labels.append(label)
test_paths


# In[3]:


#displaying images
#selecting random 10 images to display
random_indices = random.sample(range(len(train_paths)),10)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i, idx in enumerate(random_indices):
    img_path = train_paths[idx]
    img = Image.open(img_path)
    img = img.resize((128,128))

    #displaying images 
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Label :{train_labels[idx]}")
plt.tight_layout()
plt.show()


# In[4]:


#Image augmentation
def augment_image(image):
    image = np.array(image)
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8,1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8,1.2))
    image = np.array(image)/255.0
    return image

#Load images and apply augmentation (0,255)
def open_images(paths):
    images = []
    for path in paths:
        img = load_img(path, target_size=(IMAGE_SIZE,IMAGE_SIZE))
        img = augment_image(img)
        images.append(img)
    return np.array(images)

def encoder(labels): #encoder labels converting integer names to integer
    unique_labels = os.listdir(train_dir)
    encoded = [unique_labels.index(label) for label in labels]
    return np.array(encoded)

#data generator for batching
def data_generator(paths, labels, batch_size=12):
    while True:
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                batch_images = open_images(batch_paths)         # shape: (batch, 128, 128, 3)
                batch_labels = encoder(batch_labels)            # shape: (batch,) ← rank 1

                yield batch_images, batch_labels




# In[ ]:


# Model Architecture (128,128,3)
import math
IMAGE_SIZE = 128
base_model = VGG16(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top= False, weights='imagenet')

#freeze all layers of the VGG16 base model
for layer in base_model.layers:
    layer.trainable = False

# Set only the last few layers 
base_model.layers[-2].trainable = True 
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

#Build Model
model = Sequential()
model.add(Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(base_model) # VGG16 model
model.add(Flatten()) # Flatten Layer
model.add(Dropout(0.3)) # To prevent overfitting drop 30% neurons

model.add(Dense(128, activation='relu')) # Dense layer
model.add(Dropout(0.2)) #Dropout layer

model.add(Dense(len(os.listdir(train_dir)),activation='softmax')) #multiclass classification

#compile model
model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

#Parameters
batch_size = 20
steps = math.ceil(len(train_paths) / batch_size)
epochs = 5

history = model.fit(
    data_generator(train_paths, train_labels, batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=steps
)


# In[ ]:


#plotting losses
plt.figure(figsize=(8,4))
plt.grid(True)
plt.plot(history.history['sparse_categorical_accuracy'],'.g-',linewidth=2)
plt.plot(history.history['loss'],'.r-', linewidth=2)
plt.title('Model training history')
plt.xlabel('epoch')
plt.xticks([x for x in range(epochs)])
plt.legend(['Accuracy','Loss'],loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
import numpy as np

# 1. Prediction on test data
test_images = open_images(test_paths)  # Load and augment test images
test_labels_encoded = encoder(test_labels)  # Encode the test labels

# Predict using the trained model
test_predictions = model.predict(test_images)

# 2. Classification Report
print("Classification Report:")
print(classification_report(test_labels_encoded, np.argmax(test_predictions, axis=1)))


# In[ ]:




