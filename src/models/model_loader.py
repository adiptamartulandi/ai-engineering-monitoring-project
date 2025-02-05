import pickle

from typing import Tuple, Dict
from src.models.logistic_regression import SentimentModel
from src.config import MODEL_PATH, VECTORIZER_PATH, LABEL_MAP_PATH


def load_model() -> Tuple[SentimentModel, object, Dict[str, int]]:
    """
    Load the trained model, vectorizer, and label mapping
    """
    try:
        # Initialize the model
        model = SentimentModel()
        
        # Load model state
        with open(MODEL_PATH, 'rb') as f:
            model.model = pickle.load(f)
            
        # Load vectorizer
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
            
        # Load label mapping
        with open(LABEL_MAP_PATH, 'rb') as f:
            label_map = pickle.load(f)
            
        # Create reverse mapping for predictions
        id2label = {v: k for k, v in label_map.items()}
        
        return model, vectorizer, id2label
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")