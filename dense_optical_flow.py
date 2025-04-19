import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("data/people.mp4")

# Read the first frame and convert to grayscale
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    cap.release()
    exit()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Initialize HSV image: same size as frame, 3 channels
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255  # Full saturation for visualization

# Just for intuition
print(f"Original frame shape: {first_frame.shape}")  # (H, W, 3)
print(f"Grayscale frame shape: {prev_gray.shape}")  # (H, W)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow between previous and current grayscale frame
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # flow is shape (H, W, 2) → one 2D vector per pixel
    print(f"Flow shape: {flow.shape}")  # Should be (H, W, 2)

    # Split flow into magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    print(f"mag shape: {mag.shape}, ang shape: {ang.shape}")  # Both (H, W)
    print(
        f"Sample vector at (100,100): dx={flow[100, 100, 0]:.2f}, dy={flow[100, 100, 1]:.2f}"
    )
    print(
        f" → mag={mag[100, 100]:.2f}, angle={ang[100, 100] * 180 / np.pi:.2f} degrees"
    )

    # Hue = angle of motion (direction)
    hsv[..., 0] = ang * 180 / np.pi / 2  # OpenCV hue range is [0,180]

    # Value = magnitude of motion (speed)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for display
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Dense Optical Flow", bgr_flow)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
