import time

from fastapi import FastAPI, HTTPException
from src.dto import TweetRequest, TweetResponse
from src.models.model_loader import load_model
from prometheus_fastapi_instrumentator import Instrumentator
from src.config import API_TITLE, HOST, PORT, PREDICTION_TIME, PREDICTION_COUNTER


app = FastAPI(title=API_TITLE)

# Load model, vectorizer and label mapping
model, vectorizer, id2label = load_model()


@app.post("/predict", response_model=TweetResponse)
async def predict_sentiment(request: TweetRequest):
    start_time = time.time()
    try:
        # Transform input text using TF-IDF
        X = vectorizer.transform([request.text])

        # Make prediction
        probabilities = model.predict(X)
        prediction = int(probabilities[0] > 0.5)  # Convert probability to class
        confidence = probabilities[0] if prediction == 1 else 1 - probabilities[0]

        # Get predicted sentiment
        sentiment = id2label[prediction]

        # Update Prometheus metrics
        PREDICTION_TIME.observe(time.time() - start_time)
        PREDICTION_COUNTER.labels(sentiment=sentiment).inc()

        return TweetResponse(sentiment=sentiment, confidence=float(confidence))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)