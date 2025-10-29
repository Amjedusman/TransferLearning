import cv2
import numpy as np
import os

# --- Load the Haar Cascade file ---
cascade_path = r'D:\Altos\training\ml\assignments\tranfer learning\haarcascade_frontalface_default.xml'
print("Cascade path exists:", os.path.exists(cascade_path))

facedetect = cv2.CascadeClassifier(cascade_path)

# --- Safety check: exit if not loaded ---
if facedetect.empty():
    print("⚠️ Error loading cascade file! Check the path again.")
    exit()

# --- Initialize video capture (0 = default webcam) ---
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("⚠️ Could not access webcam.")
    exit()

faces_data = []
i = 0

print("✅ Press 'q' to quit and save captured faces.")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))

        # Collect 100 face samples (1 every 10 frames)
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)

        i += 1

        # Display number of collected samples on screen
        cv2.putText(frame, f"Samples: {len(faces_data)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Face Capture", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# --- Save the collected face data ---
faces_data = np.asarray(faces_data)
print(f"\n✅ Collected {len(faces_data)} face samples.")

# Create a folder for saving data if not exists
save_path = r'D:\Altos\training\ml\assignments\tranfer learning\faces_data.npy'
np.save(save_path, faces_data)
print(f"💾 Face data saved at: {save_path}")
