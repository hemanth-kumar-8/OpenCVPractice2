import cv2 as cv
import numpy as np

# Load video
cap = cv.VideoCapture("14346674-uhd_3840_2160_30fps.mp4")

# Create background subtractor
fgbg = cv.createBackgroundSubtractorMOG2()

# Create resizable window
cv.namedWindow("Output", cv.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    
    # Break if video ends
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Resize output (change size if needed)
    fgmask_resized = cv.resize(fgmask, (800, 600))

    # Show output
    cv.imshow("Output", fgmask_resized)

    # Press ESC to exit
    if cv.waitKey(30) & 0xFF == 27:
        break

# Release resources
cap.release()
cv.destroyAllWindows()