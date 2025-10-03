from datasets import load_dataset

# Load dataset
dataset = load_dataset("sms_spam")

# Split into train/test (e.g., 80/20 split)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

print(dataset)  # now it will show "train" and "test"
print(dataset['train'][0])

# ---------------- Tokenizer ----------------
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sms"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)

# ---------------- Model ----------------
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# ---------------- Training ----------------
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

training_args = TrainingArguments(
    output_dir="./results_spam",
    eval_strategy="epoch",   # fixed from eval_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ---------------- Inference ----------------
test_sms = ["Congratulations! You won ₹5000 cash!", "Are we still meeting tomorrow?"]

inputs = tokenizer(test_sms, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
preds = outputs.logits.argmax(dim=1)

for msg, label in zip(test_sms, preds):
    print(f"{msg} → {'Spam' if label.item() == 1 else 'Ham'}")

