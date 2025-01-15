from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
import multiprocessing

# Set paths for saving models and logs
FINE_TUNED_MODEL_DIR = "./models/fine_tuned_model"
LOG_DIR = "./logs"

# Function to load and preprocess dataset
def load_and_preprocess_data():
    """
    Load and preprocess the IMDb dataset.
    Returns tokenized training and testing datasets along with the tokenizer.
    """
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")

    # Shuffle and split datasets
    train_data = dataset["train"].shuffle(seed=42)
    test_data = dataset["test"].shuffle(seed=42)

    # Load tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenization function
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_data = train_data.map(tokenize, batched=True, num_proc=1)
    test_data = test_data.map(tokenize, batched=True, num_proc=1)

    return train_data, test_data, tokenizer


# Function to train the model
def train_model(train_data, test_data, tokenizer):
    """
    Train a pre-trained model on the tokenized IMDb dataset.
    Saves the fine-tuned model to the specified directory.
    """
    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Set training arguments
    print("Setting training arguments...")
    training_args = TrainingArguments(
        output_dir=FINE_TUNED_MODEL_DIR,    # Directory to save the model
        evaluation_strategy="epoch",       # Evaluate after every epoch
        learning_rate=2e-5,                # Learning rate
        per_device_train_batch_size=8,    # Batch size for training
        num_train_epochs=1,                # Number of training epochs
        save_steps=10_000,                 # Save checkpoint every 10,000 steps
        save_total_limit=2,                # Keep only 2 checkpoints
        logging_dir=LOG_DIR,               # Directory to save logs
        logging_steps=500,               # Log every 500 steps
        gradient_accumulation_steps=2
    )

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    print(f"Saving fine-tuned model to {FINE_TUNED_MODEL_DIR}...")
    model.save_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
    print("Model training and saving complete!")


# Main function
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    # Ensure output directories exist
    os.makedirs(FINE_TUNED_MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Step 1: Load and preprocess data
    train_data, test_data, tokenizer = load_and_preprocess_data()

    # Step 2: Train the model
    train_model(train_data, test_data, tokenizer)

    print("Pipeline execution complete!")