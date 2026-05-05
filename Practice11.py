import cv2 as cv

cap = cv.VideoCapture(0)

ret, frame1 = cap.read()
prev = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(
        prev, next, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    cv.imshow("Frame", frame2)

    prev = next

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()