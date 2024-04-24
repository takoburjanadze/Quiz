#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[18]:


df = pd.read_csv("C:/Users/Lenovo/Downloads/processed.csv")


# In[19]:


features = df.drop(['Label1'], axis=1)
label = df['Label1']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)


# In[21]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test))


# In[22]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

with open('accuracy1.txt', 'w') as file:
    file.write(f'Test Loss: {loss:.4f}\n')
    file.write(f'Test Accuracy: {accuracy:.4f}')


# In[ ]:




