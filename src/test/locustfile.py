import json

from locust import HttpUser, task, between


class TweetSentimentUser(HttpUser):
    wait_time = between(1, 3)  # Random wait time between requests

    def on_start(self):
        """
        Initialize test data
        """
        self.test_tweets = [
            "I absolutely love this product! It's amazing!",
            "This is the worst experience ever. Very disappointed.",
            "The weather is nice today, going for a walk.",
            "Can't believe how terrible the service was.",
            "Just got promoted! Best day ever!"
        ]

    @task
    def predict_sentiment(self):
        """
        Test sentiment prediction endpoint
        """
        # Randomly select a tweet from test data
        for tweet in self.test_tweets:
            headers = {"Content-Type": "application/json"}
            payload = {"text": tweet}
            
            # Send POST request to prediction endpoint
            with self.client.post(
                "/predict",
                json=payload,
                headers=headers,
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if "sentiment" in result and "confidence" in result:
                            response.success()
                        else:
                            response.failure("Invalid response format")
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON response")
                else:
                    response.failure(f"Request failed with status code: {response.status_code}")