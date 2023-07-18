import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#Read the data
df = pd.read_csv('D:/Skillsupdate/python-project-color-detection/news.csv')

#Get the labels
#labels = df.label: Extracts the 'label' column from the DataFrame and assigns it to the variable 'labels'.
labels=df.label

#Split the dataset,train_test_split: Splits the dataset into training and testing sets.
#train_test_split: Splits the dataset into training and testing sets.
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
#TfidfVectorizer: Converts a collection of raw documents into a matrix of TF-IDF features.
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set, transform test set
#tfidf_vectorizer.fit_transform: Learns the vocabulary and transforms the training set text into TF-IDF features.
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 

#tfidf_vectorizer.transform: Transforms the test set text into TF-IDF features using the learned vocabulary.
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier
#PassiveAggressiveClassifier: Initializes the Passive Aggressive Classifier for classification.
pac=PassiveAggressiveClassifier(max_iter=50)

#pac.fit: Trains the classifier on the TF-IDF features and the corresponding labels.
pac.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
#pac.predict: Predicts the labels of the test set using the trained model.
y_pred=pac.predict(tfidf_test)

#accuracy_score: Calculates the accuracy of the model by comparing predicted labels with true labels.
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#Build confusion matrix
#confusion_matrix: Builds a confusion matrix to evaluate the model's performance.
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])