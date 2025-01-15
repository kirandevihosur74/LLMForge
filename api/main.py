from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import logging
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize FastAPI app
app = FastAPI()

# Load the fine-tuned model pipeline
model_pipeline = pipeline("text-classification", model="../scripts/models/fine_tuned_model")

# Add Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)


# Input model for the API
class InputText(BaseModel):
    input_text: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/predict/")
async def predict(input_data: InputText):
    logging.info(f"Received input: {input_data.input_text}")
    prediction = model_pipeline(input_data.input_text)
    logging.info(f"Prediction: {prediction}")
    return {"prediction": prediction}