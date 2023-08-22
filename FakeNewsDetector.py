import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
data = pd.read_csv('dataset.csv')  # Replace with the path to your dataset

# Drop rows with missing values
data.dropna(subset=['title'], inplace=True)

# Split dataset into features and labels
X = data['title']
y = data['label']

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the Random Forest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Streamlit app
st.title("Real vs. Fake News Classifier")

# Dropdown for selecting a news title
selected_title = st.selectbox("Select a news title", X)

# Predict the label
if st.button("Predict"):
    selected_title_vectorized = vectorizer.transform([selected_title])
    prediction = clf.predict(selected_title_vectorized)
    if prediction[0] == 1:
        result = "Real"
    else:
        result = "Fake"
    st.write(f"The title is predicted to be: {result}")

