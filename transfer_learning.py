import importlib
import subprocess
import sys

# Function to check and install packages
def install_if_missing(package, import_name=None):
    try:
        if import_name is None:
            import_name = package
        importlib.import_module(import_name)
    except ImportError:
        print(f"⚠️  {package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ------------------------------------------------------------
# Step 0: Ensure required packages are installed
# ------------------------------------------------------------
install_if_missing("transformers")
install_if_missing("datasets")
install_if_missing("evaluate")
install_if_missing("accelerate")
install_if_missing("scikit-learn", "sklearn")   # scikit-learn is installed as sklearn

# ------------------------------------------------------------
# Step 1: Now import everything safely
# ------------------------------------------------------------
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
# ------------------------------------------------------------
# Step 3: Load dataset (IMDB reviews)
# ------------------------------------------------------------
dataset = load_dataset("imdb")

# Inspect
print(dataset)
print(dataset["train"][0])

# ------------------------------------------------------------
# Step 4: Tokenization
# ------------------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ------------------------------------------------------------
# Step 5: Load model (pretrained DistilBERT)
# ------------------------------------------------------------
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# ------------------------------------------------------------
# Step 6: Prepare training/validation/test sets
# ------------------------------------------------------------
# Use smaller subsets for quick training (optional)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))

# ------------------------------------------------------------
# Step 7: Metrics
# ------------------------------------------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

# ------------------------------------------------------------
# Step 8: Training arguments
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,   # can increase to 3-5 for better accuracy
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True
)

# ------------------------------------------------------------
# Step 9: Trainer API
# ------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ------------------------------------------------------------
# Step 10: Train
# ------------------------------------------------------------
trainer.train()

# ------------------------------------------------------------
# Step 11: Evaluate
# ------------------------------------------------------------
results = trainer.evaluate()
print("Final Evaluation:", results)

# ------------------------------------------------------------
# Step 12: Inference
# ------------------------------------------------------------
test_text = "This movie was absolutely fantastic! Great acting and story."
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=-1).item()
print("Prediction:", "Positive" if pred == 1 else "Negative")


# Save model and tokenizer
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("Model and tokenizer saved!")

