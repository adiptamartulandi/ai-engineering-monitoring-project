import numpy as np
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class SentimentModel:
    def __init__(self, params=None, save_dir='models'):
        self.params = params or {
            'penalty': 'l1',
            'solver': 'saga'
        }
        self.model = None
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Machine Learning model and evaluate if validation data is provided
        """
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)

        # Evaluate the model on training dataset
        train_predictions = self.predict(X_train)
        train_predictions = (train_predictions > 0.5).astype(int)
        train_accuracy = accuracy_score(y_train, train_predictions)
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # Evaluate the model on validation dataset
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_predictions = (val_predictions > 0.5).astype(int)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

    def predict(self, X):
        """
        Make predictions using the trained model and return probabilities
        """
        if self.model is None:
            raise ValueError("Model needs to be trained before making predictions")
        # Get probability predictions for positive class (class 1)
        probabilities = self.model.predict_proba(X)
        return probabilities[:, 1]

    def save_model(self, model_path):
        """
        Save the trained model to disk
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, model_path):
        """
        Load a trained model from disk
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)