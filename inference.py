# inference.py
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd
import os

# -----------------------------
# Step 1: Load trained model and tokenizer from checkpoint
# -----------------------------
CHECKPOINT_DIR = "./results/checkpoint-626"  # Use the latest checkpoint

if not os.path.exists(CHECKPOINT_DIR):
    raise FileNotFoundError(f"Checkpoint folder not found at {CHECKPOINT_DIR}. Make sure training has been done!")

model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
tokenizer = DistilBertTokenizerFast.from_pretrained(CHECKPOINT_DIR)

# -----------------------------
# Step 2: Function to predict sentiment
# -----------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    return "Positive" if pred == 1 else "Negative"

# -----------------------------
# Step 3: Read reviews (choose one)
# -----------------------------

# Option 1: Single manual review
review_text = "This movie was a complete disaster, really disappointing!"
print(f"Review: {review_text}")
print(f"Sentiment: {predict_sentiment(review_text)}\n")

# Option 2: Read reviews from a CSV file (column must be named 'review')
csv_file = "reviews.csv"  # Change to your CSV path
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    if "review" not in df.columns:
        raise ValueError("CSV file must have a column named 'review'")
    
    print("Batch predictions from CSV:")
    df["sentiment"] = df["review"].apply(predict_sentiment)
    print(df)
    # Optional: Save results to CSV
    df.to_csv("reviews_with_sentiment.csv", index=False)
