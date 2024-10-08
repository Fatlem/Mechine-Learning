#!/usr/bin/env python
# coding: utf-8

# ## Importing the Dependencies 📚
# 
# Let's start with importing the necessary dependencies for our analysis and model building process. 🚀

# In[1]:


import os

import time

import random

import itertools

from PIL import Image

#--------------------------------------------------------------------------------------#

import numpy as np

import pandas as pd

#--------------------------------------------------------------------------------------#

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

#--------------------------------------------------------------------------------------#

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

#--------------------------------------------------------------------------------------#

import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Input, Dense, Dropout

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#--------------------------------------------------------------------------------------#

print(f'Tensorflow Version: {tf.__version__}')

print("GPU Available:", tf.config.list_physical_devices('GPU')[0])

#--------------------------------------------------------------------------------------#

import warnings

warnings.filterwarnings("ignore")


# # Visualization and Preprocessing 📊

# In[2]:


# helper functions

#--------------------------------------------------------------------------------------#

def data_appender(data, path, folder):
    folderpath = os.path.join(path, folder)
    files = os.listdir(folderpath)
    
    for file in files:
        filepath = os.path.join(folderpath, file)
        data['imgpath'].append(filepath)
        data['labels'].append(folder)
    
    return data

#--------------------------------------------------------------------------------------#

def dataset_splitter(dataset, train_size = 0.9, test_size = 0.5, shuffle = True, random_state = 0):
    train_df, temp_df = train_test_split(dataset, train_size = train_size, 
                                         shuffle = shuffle, random_state = random_state)
    
    val_df, test_df = train_test_split(temp_df, test_size = test_size, 
                                       shuffle = shuffle, random_state = random_state)

    train_df = train_df.reset_index(drop = True)
    val_df = val_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)
    
    return train_df, val_df, test_df

#--------------------------------------------------------------------------------------#


# ## Organizing the Images 📂
# 
# Let's begin by organizing the images into a dataset that we'll be working with. 📊

# In[3]:


data = {'imgpath': [], 'labels': []}
path = '/kaggle/input/rice-image-dataset/Rice_Image_Dataset'
folders = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

for folder in folders:
    data = data_appender(data = data, path = path, folder = folder)

dataset = pd.DataFrame(data)
dataset.head()


# ## Exploratory Visualizations 📊
# 
# In this section, we'll explore various visualizations to gain insights into our dataset. 📈

# ### Visualizing Random Images 🔍

# In[4]:


rows = 5
cols = 5

selected_indices = random.sample(range(len(dataset)), rows * cols)

fig, axes = plt.subplots(rows, cols, figsize = (12, 12))
gs = gridspec.GridSpec(rows, cols, wspace = 0.0, hspace = 0.0)

for i, idx in enumerate(selected_indices):
    row = i // cols
    col = i % cols
    img_path = dataset['imgpath'].iloc[idx]
    label = dataset['labels'].iloc[idx]
    img = Image.open(img_path)
    axes[row, col].imshow(img)
    axes[row, col].axis('off')
    axes[row, col].set_title(label, fontsize = 10)

plt.show()


# ### Distribution of Labels 📊

# In[5]:


fig, ax = plt.subplots(figsize = (10, 5))

labels = dataset['labels'].value_counts().index
sizes = dataset['labels'].value_counts().values
colors = sns.color_palette('pastel')

wedges, texts, autotexts = ax.pie(sizes, colors = colors, 
                                  autopct = '%1.1f%%', startangle = 140, 
                                  explode = (0.1, 0, 0, 0, 0), wedgeprops = dict(edgecolor = 'black'))

ax.set_title('Distribution of Labels', fontsize = 16, fontweight = 'bold')
ax.axis('equal') 
ax.legend(wedges, labels, loc = "best", fontsize = 12)

for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')

for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()


# ## Data Pre-Processing 🧪
# 
# In this section, we'll preprocess our data to prepare it for model training. 📝
# 

# In[6]:


train_df, val_df, test_df = dataset_splitter(dataset)

print(f'\nTraining Dataframe of shape {train_df.shape}: \n{train_df.head()}')
print(f'\nValidation Dataframe of shape {val_df.shape}: \n{val_df.head()}')
print(f'\nTesting Dataframe of shape {test_df.shape}: \n{test_df.head()}')


# In[7]:


get_ipython().run_cell_magic('time', '', "\nseed = 0\nbatch_size = 128\nimg_size = (224, 224)\n\ngenerator = ImageDataGenerator(rescale = 1./255)\n\ntrain_data = generator.flow_from_dataframe(train_df, x_col = 'imgpath', y_col = 'labels', \n                                           color_mode = 'rgb', class_mode = 'categorical', \n                                           batch_size = batch_size, target_size = img_size, \n                                           shuffle = True, seed = seed)\n\nval_data = generator.flow_from_dataframe(val_df, x_col = 'imgpath', y_col = 'labels', \n                                         color_mode = 'rgb', class_mode = 'categorical', \n                                         batch_size = batch_size, target_size = img_size, \n                                         shuffle = False)\n\ntest_data = generator.flow_from_dataframe(test_df, x_col = 'imgpath', y_col = 'labels', \n                                          color_mode = 'rgb', class_mode = 'categorical', \n                                          batch_size = batch_size, target_size = img_size, \n                                          shuffle = False)\n")


# # Creating and Training the Classifier 🛠️
# 
#  this section, we'll develop and train our classification model for rice grain classification. 📈

# ## Creating the Classifier 🤖

# In[8]:


base_model = MobileNetV2(include_top = False, weights = 'imagenet', 
                       input_shape = img_size + (3,), pooling = 'max')

for layer in base_model.layers:
    layer.trainable = False


# In[9]:


model = Sequential([
    Input(shape = img_size + (3,), name = 'input_layer'),
    base_model,
    
    Dense(512, activation = 'relu'),
    Dropout(0.4, seed = seed),
    
    Dense(256, activation = 'relu'),
    Dropout(0.4, seed = seed),
    
    Dense(len(set(train_data.classes)), activation = 'softmax', name = 'output_layer')
])

model.compile(optimizer = Adam(learning_rate = 0.001), 
              loss = CategoricalCrossentropy(), metrics = ['accuracy'])

model.summary()


# ## Training the Classifier 🎓

# In[10]:


get_ipython().run_cell_magic('time', '', "\nmodel_es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 2, restore_best_weights = True)\nmodel_rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 1, mode = 'min')\n\nhistory = model.fit(train_data, validation_data = val_data, \n                    epochs = 10, callbacks = [model_es, model_rlr])\n")


# # Evaluating the Classifier 📏
# 
# In this section, we'll assess the performance of our email classification model. 📈

# In[11]:


test_results = model.evaluate(test_data, verbose = 0)

print(f'Test Loss: {test_results[0]:.5f}')
print(f'Test Accuracy: {(test_results[1] * 100):.2f}%')


# In[12]:


class_labels = list(train_data.class_indices.keys())

test_classes = test_data.classes
predicted_classes = np.argmax(model.predict(test_data, verbose = 0), axis = 1)

print(f'Classification Report (Test) --> \n\n' + \
f'{classification_report(test_classes, predicted_classes, target_names = class_labels)}')


# ## Evaluation Visualization 📊
# 
# Here, we'll visualize the evaluation metrics to gain insights into our classifier's performance. 📈

# ### Accuracy and Loss Curve 📈

# In[13]:


_, ax = plt.subplots(ncols = 2, figsize = (15, 6))

# accuracy

ax[0].plot(history.history['accuracy'], marker = 'o', color = 'blue', markersize = 7)
ax[0].plot(history.history['val_accuracy'], marker = 'x', color = 'red', markersize = 7)
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend(['Train', 'Validation'])
ax[0].grid(alpha = 0.2)

# loss

ax[1].plot(history.history['loss'], marker = 'o', color = 'blue', markersize = 7)
ax[1].plot(history.history['val_loss'], marker = 'x', color = 'red', markersize = 7)
ax[1].set_title('Model Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend(['Train', 'Validation'])
ax[1].grid(alpha = 0.2)

plt.show()


# ### Confusion Matrix 📊

# In[14]:


test_matrix = confusion_matrix(test_classes, predicted_classes)

class_labels = list(test_data.class_indices.keys())

disp_test = ConfusionMatrixDisplay(confusion_matrix = test_matrix, display_labels = class_labels)

plt.figure(figsize = (10, 10))

disp_test.plot(cmap = 'YlGnBu', colorbar = False)
plt.title('Confusion Matrix')

plt.show()


# **Thank you for reviewing my notebook! If you found it helpful, I'd appreciate an upvote.** 😄
