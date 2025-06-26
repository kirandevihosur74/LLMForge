# LLMForge: Sentiment Analysis with BERT

LLMForge is a comprehensive machine learning project that implements sentiment analysis using fine-tuned BERT models. The project includes model training, conversion between frameworks, and a production-ready REST API with monitoring capabilities.

## 🚀 Features

- **BERT-based Sentiment Analysis**: Fine-tuned BERT model for binary sentiment classification
- **Multi-Framework Support**: PyTorch training with TensorFlow conversion capabilities
- **Production API**: FastAPI-based REST API with Prometheus monitoring
- **Docker Support**: Containerized deployment ready
- **Comprehensive Testing**: Model validation and testing scripts

## 📁 Project Structure

```
LLMForge/
├── api/
│   └── main.py              # FastAPI application with sentiment analysis endpoints
├── scripts/
│   ├── train.py             # BERT model training on IMDb dataset
│   ├── test_model.py        # PyTorch model testing
│   ├── convert.py           # PyTorch to TensorFlow model conversion
│   └── test_tf_model.py     # TensorFlow model testing
├── Dockerfile               # Docker configuration for API deployment
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLMForge
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### 1. Model Training

Train a BERT model on the IMDb dataset for sentiment analysis:

```bash
cd scripts
python train.py
```

This will:
- Load and preprocess the IMDb dataset
- Fine-tune a BERT-base-uncased model
- Save the trained model to `./models/fine_tuned_model/`
- Generate training logs in `./logs/`

### 2. Model Testing

Test the trained PyTorch model:

```bash
python test_model.py
```

### 3. Model Conversion

Convert the PyTorch model to TensorFlow format:

```bash
python convert.py
```

Test the converted TensorFlow model:

```bash
python test_tf_model.py
```

### 4. API Deployment

#### Local Development

Start the FastAPI server locally:

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Docker Deployment

Build and run the Docker container:

```bash
docker build -t llmforge-api .
docker run -p 8000:8000 llmforge-api
```

## 🔌 API Endpoints

### Base URL
- **Development**: `http://localhost:8000`
- **Docker**: `http://localhost:8000`

### Available Endpoints

#### GET `/`
- **Description**: Welcome message
- **Response**: `{"message": "Welcome to the Sentiment Analysis API!"}`

#### POST `/predict/`
- **Description**: Predict sentiment for input text
- **Request Body**:
  ```json
  {
    "input_text": "The movie was absolutely fantastic!"
  }
  ```
- **Response**:
  ```json
  {
    "prediction": [
      {
        "label": "LABEL_1",
        "score": 0.9876
      }
    ]
  }
  ```

#### GET `/metrics`
- **Description**: Prometheus metrics endpoint for monitoring
- **Response**: Prometheus-formatted metrics

## 📊 Model Details

- **Base Model**: `bert-base-uncased`
- **Task**: Binary sentiment classification (positive/negative)
- **Dataset**: IMDb movie reviews
- **Training**: 1 epoch with learning rate 2e-5
- **Batch Size**: 8 (with gradient accumulation steps of 2)

## 🔧 Configuration

### Training Parameters
- **Learning Rate**: 2e-5
- **Batch Size**: 8
- **Epochs**: 1
- **Evaluation**: After each epoch
- **Checkpointing**: Every 10,000 steps

### Model Output
- **PyTorch Model**: `./models/fine_tuned_model/`
- **TensorFlow Model**: `./models/tf_model/`
- **Logs**: `./logs/`

## 📈 Monitoring

The API includes Prometheus instrumentation for monitoring:
- Request metrics
- Response times
- Error rates
- Custom business metrics

Access metrics at: `http://localhost:8000/metrics`

## 🐳 Docker

The project includes a Dockerfile for easy deployment:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY ./api /app
RUN pip install --no-cache-dir fastapi uvicorn transformers tensorflow
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing

The project includes comprehensive testing scripts:

1. **Model Testing**: Validate model predictions with sample inputs
2. **Framework Conversion**: Test PyTorch to TensorFlow conversion
3. **API Testing**: Test the REST API endpoints

## 📋 Requirements

### Core Dependencies
- **PyTorch**: 2.5.1
- **Transformers**: 4.48.0
- **TensorFlow**: 2.13.0
- **FastAPI**: 0.95.1
- **Datasets**: 3.2.0

### Additional Dependencies
- **Prometheus FastAPI Instrumentator**: For monitoring
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **Scikit-learn**: Machine learning utilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔮 Future Enhancements

- [ ] Multi-class sentiment analysis
- [ ] Real-time streaming API
- [ ] Model versioning and A/B testing
- [ ] Kubernetes deployment manifests
- [ ] Model performance benchmarking
- [ ] Additional language support
