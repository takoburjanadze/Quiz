#!/usr/bin/env python
# coding: utf-8

# In[6]:


import ipaddress
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Read the dataset
df = pd.read_csv("C:/Users/Lenovo/Downloads/Darknet.csv")

# Drop unnecessary columns
df = df.drop(["Flow ID", "Timestamp", "Label2"], axis=1)

# Drop rows with missing values
df = df.dropna()

# Convert IP addresses to integers
df['Src IP'] = df['Src IP'].apply(lambda x: int(ipaddress.ip_address(x)))
df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ipaddress.ip_address(x)))

# Encode the label column
label_encoder = LabelEncoder()
df['Label1'] = label_encoder.fit_transform(df['Label1'])

# Save the preprocessed dataset
df.to_csv("processed.csv", index=False)

# Identify and handle any problematic values in the dataset
problematic_columns = []
for col in df.columns:
    if df[col].dtype == np.float64 or df[col].dtype == np.int64:
        max_value = df[col].max()
        min_value = df[col].min()
        if max_value == np.inf or min_value == -np.inf:
            problematic_columns.append(col)

# Remove rows with problematic values
if problematic_columns:
    print("Removing rows with problematic values...")
    df = df[~df[problematic_columns].isin([np.inf, -np.inf]).any(axis=1)]

# Split features and label
features = df.drop(['Label1'], axis=1)
label = df['Label1']

# Scale the features using Min-Max scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Save scaled dataset
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['Label1'] = label
scaled_df.to_csv("scaled3.csv", index=False)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, label, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(scaled_features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

# Printing the accuracy
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

with open('accuracy3.txt', 'w') as f:
    f.write(f'Test Loss: {loss:.4f}\n')
    f.write(f'Test Accuracy: {accuracy:.4f}\n')


# In[7]:


# Load the scaled dataset
scaled_df = pd.read_csv("scaled3.csv")

# Display the first few rows of the dataset
print(scaled_df.head())


# In[15]:


import os

# Get the current working directory
current_directory = os.getcwd()

# Check if the file exists in the current directory or its subdirectories
file_found = False
for root, dirs, files in os.walk(current_directory):
    if "scaled3.csv" in files:
        file_path = os.path.join(root, "scaled3.csv")
        print(f"The 'scaled3.csv' file is located at: {file_path}")
        file_found = True
        break

if not file_found:
    print("The 'scaled3.csv' file was not found in the current directory or its subdirectories.")


# In[20]:


import gzip
import shutil

# Compress the CSV file using gzip compression
with open('C:\\Users\\Lenovo\\OneDrive\\Desktop\\Scaled3\\scaled3.csv', 'rb') as f_in:
    with gzip.open('C:\\Users\\Lenovo\\OneDrive\\Desktop\\Scaled3\\scaled3.csv.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Compression completed: C:\\Users\\Lenovo\\OneDrive\\Desktop\\Scaled3\\scaled3.csv.gz")


# In[ ]:




