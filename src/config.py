from prometheus_client import Counter, Histogram

# API Settings
API_TITLE = "Tweet Sentiment Analysis API"
HOST = "0.0.0.0"
PORT = 8000

# Model Paths
MODEL_PATH = 'models/sentiment_model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'
LABEL_MAP_PATH = 'models/label_map.pkl'

# Prometheus Metrics
PREDICTION_TIME = Histogram('prediction_time_seconds', 'Time spent processing prediction')
PREDICTION_COUNTER = Counter('prediction_requests_total', 'Total number of prediction requests', ['sentiment'])