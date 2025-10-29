import cv2

img = cv2.imread("cat.jpg")   # read image

# cv2.line(img, (0,0), (200,200), (255,250,0), 3)   # Blue line

# cv2.rectangle(img, (50,50), (200,200), (0,255,0), -1) #rectangle

# cv2.circle(img, (150,150), 50, (0,0,255), -1)  # -1 = filled circle

# text in cv
cv2.putText(img, "Hello OpenCV", (10,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


cv2.imshow("Cat", img)        # display image in a window
cv2.waitKey(0)                # waits for key press
cv2.destroyAllWindows()
# cv2.imwrite("saved_cat.png", img)

