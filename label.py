# Install required libraries if not already installed
# pip install transformers torch

from transformers import pipeline

# Load a binary sentiment classifier (POSITIVE/NEGATIVE)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Input sentence
sentence = input("Enter a sentence: ")

# Predict sentiment
result = classifier(sentence)[0]

# Output
print(f"Label: {result['label']}, Confidence: {round(result['score'], 4)}")
