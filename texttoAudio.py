from gtts import gTTS 
# Input text
text = "WARNING: amjed Please provide a package name or names!"

# Convert to speech
tts = gTTS(text=text, lang='en')
tts.save("output12.mp3")