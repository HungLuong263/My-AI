from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort

model = YOLO("yolov8n.pt")  # đã đúng
tracker = Sort()
cap = cv2.VideoCapture("clipcarr.mp4")

count_line_y = 300
vehicle_count = 0
counted_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # YOLOv8 inference
    detections = []

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            # Box data
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Lọc các phương tiện giao thông: car(2), motorcycle(3), bus(5), truck(7)
            if cls in [2, 3, 5, 7] and conf > 0.3:
                detections.append([x1, y1, x2, y2, conf])

    # Chuyển sang numpy và update tracker
    tracks = tracker.update(np.array(detections))

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Đếm nếu vượt qua đường line
        if count_line_y - 5 < cy < count_line_y + 5 and track_id not in counted_ids:
            vehicle_count += 1
            counted_ids.add(track_id)

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Vẽ đường đếm và số lượng xe
    cv2.line(frame, (0, count_line_y), (frame.shape[1], count_line_y), (0, 0, 255), 2)
    cv2.putText(frame, f'Vehicles: {vehicle_count}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
