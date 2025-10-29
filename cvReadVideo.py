import cv2

cap = cv2.VideoCapture("sample-video.mp4")   # 0 = webcam, or filename

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()