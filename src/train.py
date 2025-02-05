import pickle
import os

from data.dataset import TweetDataset
from models.logistic_regression import SentimentModel


def main():
    # Initialize dataset
    print("Initializing dataset...")
    dataset = TweetDataset('data/data_sentiment_analysis_twitter.csv')
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = dataset.prepare_data()
    
    # Initialize and train model
    print("Initializing and training model...")
    model = SentimentModel()
    model.train(X_train, y_train, X_test, y_test)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model's state instead of the entire model object
    with open('models/sentiment_model.pkl', 'wb') as f:
        pickle.dump(model.model, f)
    
    # Save the vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(dataset.vectorizer, f)
    
    # Save the label mapping
    label_map = {'negative': 0, 'positive': 1}
    with open('models/label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)

    print("Model trained and saved successfully!")

if __name__ == '__main__':
    main()