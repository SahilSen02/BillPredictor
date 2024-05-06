import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
from GovText import GovLoader


def clean_data(data):
    data["Full Text"] = data["Full Text"].str.replace('<.*?>', ' ')  # Remove HTML tags
    data["Full Text"] = data["Full Text"].str.replace('[^a-zA-Z]', ' ')  # Keep only alphabetic characters
    data["Full Text"] = data["Full Text"].str.lower()  # Convert to lowercase

    train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

    X_train = train_data["Full Text"]
    y_train = train_data['Partisan Lean']
    X_test = test_data["Full Text"]
    y_test = test_data['Partisan Lean']

    return X_train, y_train, X_test, y_test

def train_model(X_train, X_test, y_train, y_test):

    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    train_preds = model_pipeline.predict(X_train)

    return predictions, train_predictions

if __name__ == '__main__':

    if len(sys.argv) > 1:
        num_bills = sys.argv[1]
    else:
        num_bills = 500

    govloader = GovLoader(num_bills=num_bills)

    data_df = govloader.data_df

    X_train, y_train, X_test, y_test = clean_data(data_df)


    predictions, train_predictions = train_model(X_train, X_test, y_train, y_test)

    X_train['Partisan Lean'] = y_train
    X_train['Predictions'] = train_predictions

    X_test['Partisan Lean'] = y_test
    X_test['Predictions'] = predictions

    X_train.to_csv('Training Data.csv')

    X_test.to_csv('Testing Data.csv')