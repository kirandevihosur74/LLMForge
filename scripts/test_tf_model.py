import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

def test_tensorflow_model():
    # Load the TensorFlow model
    model_path = "./models/tf_model"
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test the model with sample text
    test_text = "The movie was absolutely fantastic! I loved every moment of it."
    inputs = tokenizer(test_text, return_tensors="tf")
    outputs = model(**inputs)

    # Get predictions
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    print(f"Input: {test_text}")
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    test_tensorflow_model()