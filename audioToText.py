
from transformers import pipeline

# Load pretrained ASR model
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Run on audio file
result = asr("D:/Altos/training/ml/assignments/tranfer learning/output.mp3")
print(result["text"])
