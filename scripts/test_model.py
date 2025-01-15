from transformers import pipeline

def test_fine_tuned_model():
    # Load the fine-tuned model
    model_path = "./models/fine_tuned_model"
    classifier = pipeline("text-classification", model=model_path)

    # Test the model with some sample inputs
    test_texts = [
        "The movie was absolutely fantastic! I loved every moment of it.",
        "This was the worst movie I have ever seen. Total waste of time.",
        "It was okay, not the best but definitely not the worst.",
    ]

    for text in test_texts:
        result = classifier(text)
        print(f"Input: {text}")
        print(f"Prediction: {result}\n")

if __name__ == "__main__":
    test_fine_tuned_model()