import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class TweetDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def load_data(self):
        """
        Load and preprocess the tweet dataset
        """
        df = pd.read_csv(self.data_path)
        return df

    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for training by vectorizing text and splitting into train/test sets
        """
        df = self.load_data()
        
        # Convert sentiment labels to numeric values
        label_map = {'Negative': 0, 'Positive': 1}
        df['sentiment'] = df['sentiment'].map(label_map)
        
        # Vectorize the text data
        X = self.vectorizer.fit_transform(df['review'])
        y = df['sentiment']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def transform_text(self, texts):
        """
        Transform new text data using the fitted vectorizer
        """
        return self.vectorizer.transform(texts)