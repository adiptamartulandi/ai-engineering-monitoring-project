# Tweet Sentiment Analysis API

This project implements a sentiment analysis model for tweets using Logistic Regression, wrapped in a FastAPI application with monitoring capabilities using Prometheus and Grafana.

## Project Structure

```
.
├── src/
│   ├── app.py          # FastAPI application
│   ├── train.py        # Model training script
│   ├── inference.py    # Inference logic
│   ├── config.py       # Configuration settings
│   ├── dto.py          # Data transfer objects
│   ├── models/         # Model implementations
│   │   ├── logistic_regression.py
│   │   ├── model_loader.py
│   │   └── __init__.py
│   ├── data/          # Data processing
│   │   ├── dataset.py
│   │   └── __init__.py          
│   └── test/          # Testing
│       └── locustfile.py     
├── data/              # Data directory
├── models/            # Saved model artifacts
├── requirements.txt   # Project dependencies
├── Dockerfile        # Container definition
├── prometheus.yml    # Prometheus configuration
└── docker-compose.yml # Container orchestration
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python src/train.py
```

4. Start the application:

   a. For local development:
   ```bash
   uvicorn src.app:app --reload
   ```

   b. With monitoring (Prometheus + Grafana):
   ```bash
   docker-compose up --build
   ```

## API Usage

The API exposes a single endpoint for sentiment prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this new feature!"}'
```

Response format:
```json
{
    "sentiment": "positive",
    "confidence": 0.95
}
```

## Monitoring

- FastAPI Application: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default credentials: admin/admin)

### Available Metrics

- `prediction_time_seconds`: Histogram of prediction processing time
- `prediction_requests_total`: Counter of total predictions by sentiment
- `http_requests_total`: Counter of HTTP requests by endpoint and status

## Query in Prometheus

1. Access the Prometheus UI at http://localhost:9090
2. Go to the 'Graph' tab
3. Try these example queries:
   - Rate of predictions: `rate(prediction_requests_total[5m])`
   - Average response time: `rate(prediction_time_seconds_sum[5m]) / rate(prediction_time_seconds_count[5m])`
   - HTTP success rate: `sum(rate(http_requests_total{status="200"}[5m]))`

## Visualization in Grafana

### Connection to Prometheus

1. Access Grafana at http://localhost:3000
2. Go to Configuration > Data Sources
3. Click "Add data source" and select "Prometheus"
4. In the HTTP URL field, enter `http://prometheus:9090` (internal Docker network URL)
5. Click "Save & Test" to verify the connection

### Creating Dashboard

1. Click '+ Create' > 'Dashboard'
2. Add a new panel and select Prometheus as the data source
3. Use these example queries for basic metrics:
   - `prediction_requests_total` - Shows total predictions by sentiment
   - `prediction_time_seconds_bucket` - Shows prediction latency distribution
   - `http_requests_total` - Shows total HTTP requests by endpoint

## Performance Testing

To run performance tests:

```bash
locust -f src/test/locustfile.py
```

Access the Locust web interface at http://localhost:8089 to configure and start the test.

## Model Details

The sentiment analysis model uses Logistic Regression with the following features:
- Binary classification (positive/negative sentiment)
- Scikit-learn implementation
- Accuracy metrics for both training and validation sets

The model is automatically saved after training and loaded during API initialization.