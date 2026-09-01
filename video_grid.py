import cv2
import numpy as np
from picamera2 import Picamera2

# Input camera stream resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Output flattened board size (square)
WARP_SIZE = 1080

points = []
matrix = None

def click_event(event, x, y, flags, params):
    global matrix
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"Corner {len(points)} added: ({x}, {y})")
        
        # Calculate matrix once 4 points are selected
        if len(points) == 4:
            pts1 = np.float32(points)
            pts2 = np.float32([[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            print("Perspective matrix calculated. Streaming live warped video...")

# Initialize Pi Camera at 1280x1080
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(config)

try:
    picam2.start()
    
    cv2.namedWindow("Live Feed - Click 4 Corners")
    cv2.setMouseCallback("Live Feed - Click 4 Corners", click_event)

    print("Instructions:")
    print("1. Click 4 corners clockwise starting from Top-Left on the video feed.")
    print("2. Press 'c' to clear points and re-calibrate.")
    print("3. Press 'q' to exit.")

    while True:
        frame = picam2.capture_array()
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw selected corner markers
        for pt in points:
            cv2.circle(img, (pt[0], pt[1]), 8, (0, 0, 255), -1)

        cv2.imshow("Live Feed - Click 4 Corners", img)

        # Render flattened video stream when matrix is set
        if matrix is not None:
            flattened_board = cv2.warpPerspective(img, matrix, (WARP_SIZE, WARP_SIZE))
            
            # Draw 8x8 grid lines on the warped board
            for i in range(1, 8):
                spacing = int(WARP_SIZE / 8) * i
                cv2.line(flattened_board, (spacing, 0), (spacing, WARP_SIZE), (0, 255, 0), 2)
                cv2.line(flattened_board, (0, spacing), (WARP_SIZE, spacing), (0, 255, 0), 2)

            cv2.imshow("Live Flattened 8x8 Grid", flattened_board)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            points = []
            matrix = None
            cv2.destroyWindow("Live Flattened 8x8 Grid")
            print("Calibration reset. Click 4 corners again.")

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    picam2.close()