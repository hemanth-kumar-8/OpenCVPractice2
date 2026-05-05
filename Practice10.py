import cv2 as cv

# 1. Load the classifier ONCE at the start
# Use cv.data.haarcascades to ensure the path is correct
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv.VideoCapture(0)

while True:
    ret, frame = img.read()
    if not ret:
        break

    # 2. Keep the processing INSIDE the loop
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # 3. Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # 4. Draw rectangles on the 'frame'
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xFF == 27: # Press 'Esc' to quit
        break

img.release()
cv.destroyAllWindows()