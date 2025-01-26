import re
from typing import Dict, List
from prediction_service import PredictionService
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prediction_service = PredictionService()


class Request(BaseModel):
    text: List = []


class SinglePrediction(BaseModel):
    post: str
    hate_content_probability_percentage: str
    label: str


class Response(BaseModel):
    predictions: List[SinglePrediction] = []


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict(request: Request) -> Response:
    # Get posts from the request JSON
    posts = request.text

    if not posts:
        raise HTTPException(status_code=400, detail="No posts provided")

    # vectorize and tokenize posts
    post_vectors = []
    for post in posts:
        tokens = re.findall(r"\w+|[^\w\s]", post)
        vector = prediction_service.vectorize_post(tokens, prediction_service.embedding_matrix, max_len=50)
        post_vectors.append(vector)

    # Convert vectors to tensor
    post_vectors = torch.tensor(post_vectors, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = prediction_service.best_model(post_vectors).squeeze()
        test_preds = outputs.cpu().numpy()
    
    result = Response(
        predictions=[
            SinglePrediction(
                post=post,
                hate_content_probability_percentage=f"{prob * 100:.2f}%",
                label="Hate" if prob >= 0.5 else "Free"
            ) for post, prob in zip(posts, test_preds)
        ]
    )     

    return result
