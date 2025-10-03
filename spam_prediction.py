import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# ------------------ CONFIG ------------------
CHECKPOINT_PATH = "./results_spam/checkpoint-558"  # <-- change to your checkpoint folder
TEXT_FILE = "text_messages.txt"                   # file with one message per line
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ LOAD MODEL ------------------
print(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}...")
model = BertForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT_PATH)
model.to(DEVICE)
model.eval()

# ------------------ READ TEXT FILE ------------------
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    messages = [line.strip() for line in f if line.strip()]  # remove empty lines

print(f"Loaded {len(messages)} messages from {TEXT_FILE}")

# ------------------ TOKENIZE ------------------
inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

# ------------------ PREDICT ------------------
with torch.no_grad():
    outputs = model(**inputs)
    preds = outputs.logits.argmax(dim=1)

# ------------------ PRINT RESULTS ------------------
for msg, label in zip(messages, preds):
    print(f"{msg} → {'Spam' if label.item() == 1 else 'Ham'}")
