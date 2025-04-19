import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("data/people.mp4")

# Get the frame rate of the video (fps)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Delay between frames in milliseconds

# Read the first frame and convert to grayscale
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    cap.release()
    exit()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Detect good features to track (corners, for example)
prev_points = cv2.goodFeaturesToTrack(
    prev_gray,
    mask=None,
    **{
        "maxCorners": 100,  # Maximum number of corners
        "qualityLevel": 0.3,  # Minimum quality level for corners
        "minDistance": 7,  # Minimum distance between corners
        "blockSize": 7,  # Size of block to compute corner
    },
)

# Ensure that points are found
if prev_points is None:
    print("Error: No features found in the first frame.")
    cap.release()
    exit()

# Create a mask image for drawing
mask = np.zeros_like(first_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (sparse) using the Lucas-Kanade method
    next_points, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Check if the flow calculation was successful
    if next_points is None or status is None:
        print("Error: Optical flow calculation failed.")
        break

    # Select good points (where tracking is successful)
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # If there are no good points, skip this frame
    if len(good_new) == 0:
        print("Warning: No good points to track.")
        prev_gray = gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)
        continue

    # Draw the tracking points
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    # Overlay the optical flow vectors on the original frame
    output_frame = cv2.add(frame, mask)

    cv2.imshow("Sparse Optical Flow", output_frame)

    # If the user presses ESC, exit the loop
    if cv2.waitKey(frame_delay) & 0xFF == 27:
        break

    # Update the previous frame and points for the next iteration
    prev_gray = gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
