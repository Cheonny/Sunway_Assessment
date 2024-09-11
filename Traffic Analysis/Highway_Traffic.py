'''
Vehicle Speed Estimation
Test case: High way
Model: YOLOv8n (Better inference speed which > YOLOv5n > YOLOv5s)
FPS: around 20 - 30 (near real time performance)
GPU: Yes (Intel UHD Graphics)
Based on the results, can conduct
      Vehicle classification, congestion analysis, traffic flow rate
'''
from collections import defaultdict
import cv2
import numpy as np
import cv2
import cvzone
from ultralytics import YOLO,solutions

# Load YOLO model
model = YOLO("yolov5n.pt")  # yolov8n/yolov5n both perform well

# Open the video file
video_path = "C:\\Users\\USER\\Desktop\\sunway_test\\Traffic Monitoring\\Highway_Traffic.mp4"
output_path = 'C:\\Users\\USER\\Desktop\\sunway_test\\Traffic Monitoring\\Processed_Highway_Traffic_Video.mp4'

cap = cv2.VideoCapture(video_path)

## Initialize VideoWriter - failed to write
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # or use 'XVID' for .avi
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize line/region for speed estimator
line_pts = [(0, 220), (720, 220)]
names = model.model.names

# Initialize speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=True,
    # spdl_dist_thresh=50
)

# Store the track history
track_history = defaultdict(lambda: [])
center_points = {}
count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = cv2.resize(frame, (720,420)) # Resize input frames 
    if success:

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            cx, cy = int(x + w / 2), int(y + h / 2)
            cv2.circle(frame, (cx,cy), 5, (0,0,255),-1)
            # Find out if that object was detected already

            if track_id not in track_history:
                count += 1

            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            inference = results[0].speed['inference']
            fps = np.ceil(1000/int(inference))
            cvzone.putTextRect(annotated_frame, f'Frame Per Seconds = {fps}', [20,50], thickness=2, scale=1, border=1)
            cvzone.putTextRect(annotated_frame, f'Vehicles Count = {count}', [20,20], thickness=2, scale=1.1, border=1)
        im0 = speed_obj.estimate_speed(annotated_frame, results, (255, 0, 0))
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
cv2.destroyAllWindows()