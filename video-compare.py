import cv2
import time

# Load the video file
cap1 = cv2.VideoCapture("output.mp4")
cap2 = cv2.VideoCapture("output1.mp4")
# Check if the video file is opened successfully
while cap1.isOpened() and cap2.isOpened():
    # Read a frame from the video
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    if success1 and success2:
        # concatenate the two frames
        frame1 = cv2.resize(frame1, (640, 480))
        frame1 = cv2.putText(frame1, "Normal", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (83, 59, 107), 2, cv2.LINE_AA)
        frame2 = cv2.resize(frame2, (640, 480))
        frame2 = cv2.putText(frame2, "SAHI", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (83, 59, 107), 2, cv2.LINE_AA)
        frame = cv2.hconcat([frame1, frame2])
        # Display the frame
        cv2.imshow("Video", frame)
        time.sleep(0.05)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
