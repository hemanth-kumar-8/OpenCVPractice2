import cv2 as cv

cap = cv.VideoCapture("14346674-uhd_3840_2160_30fps.mp4")

ret, frame = cap.read()
if not ret:
    print("Error reading video")
    exit()

# Resize instead of crop
frame = cv.resize(frame, (800, 600))

bbox = cv.selectROI("Select Object", frame, False)

if bbox[2] == 0 or bbox[3] == 0:
    print("Selection cancelled")
    exit()

tracker = cv.TrackerKCF_create()
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (800, 600))

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv.putText(frame, "Lost", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv.imshow("Tracker", frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()