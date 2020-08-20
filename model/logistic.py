# Importing the libraries
import numpy as np
import pandas as pd
import pickle


# Importing the dataset
dataset = pd.read_csv('model/maize.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# encoding dependant variable
from sklearn.preprocessing import LabelEncoder
labelencode_y = LabelEncoder()
y = labelencode_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


pickle.dump(classifier, open('new_maize.pkl', 'wb'))
model = pickle.load(open('new_maize.pkl', 'rb'))