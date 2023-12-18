
import numpy as np
import keras
import tensorflow as tf

import random as r
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
#import tensorflow_decision_forests as tfdf



df = pd.read_csv('student-mat.csv', delimiter=";")

columns_to_keep = [
    'sex', #C 0
    'age', #N 1
    'Pstatus', #C 2
    'Medu', #N 3
    'Fedu', #N 4
    'guardian', #C 5
    'studytime', #N 6
    'failures', #N 7
    'paid', #C 8
    'nursery', #C 9
    'internet', #C 10
    'Dalc', #N 11
    'Walc', #N 12
    'health', #N 13
    'absences' #N 14
]
def num_to_letter(num):
    if num < 21:
        return 'C'
    if num < 41:
        return 'B'
    if num < 61:
        return 'A'

X = pd.DataFrame()

for i in range(len(columns_to_keep)):
    X.insert(loc=i,
                  column=columns_to_keep[i],
                  value=df[(columns_to_keep[i])])


y_int = df['G1'] + df['G2'] + df['G3']
Y = []
grade_list = []
for num in y_int:
    Y.append(num_to_letter(num))





print(Y)

print(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
#numeric_transformed = scaler.fit_transform(X_train[numeric_features])

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Combine transformers using ColumnTransformer
numeric_features = [1,3,4,6,7,11,12,13,14]  # List of numeric feature column indices
categorical_features = [0,2,5,8,9]  # List of categorical feature column indices
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)  # Assuming categorical feature is the last column
    ])

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(16, (4,4),))
model.add(keras.layers.ReLU())

model.add(keras.layers.Conv2D(32, (4, 4)))
model.add(keras.layers.ReLU())


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128))
model.add(keras.layers.ReLU())
model.add(keras.layers.Dense(10, activation="softmax"))
# Create transformers for numeric and categorical features

model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy, #
              metrics=["accuracy"])

# Create a column transformer

model.fit(X_train,Y_train, epochs=5)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Model Accuracy: {accuracy}')

feature_importances = model.feature_importances
print(feature_importances)

