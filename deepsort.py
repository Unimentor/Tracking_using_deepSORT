import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize the YOLOv8 model.
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Initialize DeepSORT tracker.
tracker = DeepSort(max_age=10)

# Open the default camera (device index 0).
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open video camera.")
    exit()

# ...existing code...

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Run YOLOv8 detection on the frame.
    results = model(frame)[0]
    detections = []
    
    # Loop through each detection.
    for box in results.boxes:
        # Retrieve bounding box coordinates, confidence, and class.
        x1, y1, x2, y2 = box.xyxy[0]
        conf = 0.9
        cls = int(box.cls[0])
        
        # Convert coordinates to integers.
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w = x2 - x1
        h = y2 - y1
        
        # Prepare detection for DeepSORT:
        # Each detection is a tuple: ([x1, y1, w, h], confidence, class_name)
        detections.append(([x1, y1, w, h], float(conf), model.names[cls]))
    
    # Update the DeepSORT tracker with current detections.
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Draw the tracking bounding boxes and IDs on the frame.
    for track in tracks:
        # Only consider confirmed tracks.
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()  # Get left, top, right, bottom coordinates.
        
        # Draw the bounding box.
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        
        # Display the class name above the bounding box.
        class_name = track.get_det_class()  # Assuming track has a method to get the class name
        cv2.putText(frame, f"Class: {class_name}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the processed frame.
    cv2.imshow("DeepSORT Tracking", frame)
    
    # Exit if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Optional: slow down the loop if needed.
    time.sleep(0.02)

# Release the video capture and close windows.
cap.release()
cv2.destroyAllWindows()