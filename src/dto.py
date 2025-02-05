from pydantic import BaseModel


class TweetRequest(BaseModel):
    text: str

class TweetResponse(BaseModel):
    sentiment: str
    confidence: float