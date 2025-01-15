from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

def convert_to_tensorflow_savedmodel():
    # Load the fine-tuned PyTorch model
    print("Loading PyTorch model...")
    tf_model = TFAutoModelForSequenceClassification.from_pretrained("./models/fine_tuned_model", from_pt=True)

    # Save the TensorFlow model
    print("Saving TensorFlow model...")
    tf_model.save_pretrained("./models/tf_model")

    # Save the tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./models/fine_tuned_model")
    tokenizer.save_pretrained("./models/tf_model")

    print("Model and tokenizer saved in TensorFlow SavedModel format!")

if __name__ == "__main__":
    convert_to_tensorflow_savedmodel()