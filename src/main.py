import cv2
import yaml
import sys
import time
from modules.detection import DetectorYolov8s
from modules.visualization.visualizer import Visualizer
from modules.detection.sahi.detector import DetectorSAHI

sys.path.insert(0, '../')

video_path = 'data/pothole.mp4'
cap = cv2.VideoCapture(video_path)

with open("configs\datasets\data.yaml") as f:
    data_cfg = yaml.load(f, Loader=yaml.FullLoader)
with open("configs\default.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

video_result = cv2.VideoWriter('pothole.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               20, (640, 640))
detector = DetectorSAHI("yolov8s.yaml")
visualizer = Visualizer(cfg)

# Loop through the video frames
start_time = time.time()
num_frames = 0
print("cap.isOpened():", cap.isOpened())
while cap.isOpened():
    # Read a frame from the video

    success, frame = cap.read()

    if success:
        num_frames += 1
        # Run YOLOv8 inference on the frame
        frame, boxes, scores = detector.get_boxes(frame)

        # Visualize the results on the frame
        annotated_frame = visualizer.plot(
            frame, boxes, scores, names=data_cfg["names"])
        video_result.write(annotated_frame)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
total_time = time.time() - start_time
print("FPS:", num_frames / total_time)
# Release the video capture object and close the display window
cap.release()
video_result.release()
cv2.destroyAllWindows()
