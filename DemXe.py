import torch
import cv2

# Đường dẫn đến mô hình huấn luyện
weights_path = '/Users/hung/yolov5/runs/train/my_model46/weights/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True).to(device)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    exit()

# Open video file
cap = cv2.VideoCapture('/Users/hung/clipcarr.mp4')

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
out = cv2.VideoWriter('clipcarr_out_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Định nghĩa các thông số đếm xe
line_position = 400  # Vị trí y của đường kẻ để đếm xe
counted_cars = 0  # Tổng số xe đã đếm
car_ids = set()  # Tập hợp lưu các ID của xe đã đi qua

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Suy luận trên khung hình
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Chuyển kết quả phát hiện thành mảng numpy

    # Vẽ đường kẻ trên khung hình
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 255, 255), 1)

    # Duyệt qua các đối tượng phát hiện
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection[:6]
        
        # Kiểm tra nếu lớp là xe hơi (giả sử lớp 0 là xe hơi)
        if int(cls) == 0:
            mid_y = (y1 + y2) / 2  # Trung điểm của box theo trục y
            #car_id = f"{int(x1)}_{int(y1)}"  # ID giả lập dựa trên vị trí để phân biệt từng xe
            car_id = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"

            # Kiểm tra nếu box đã đi qua vạch đếm và chưa được đếm
            
            if mid_y > line_position and car_id not in car_ids:
                counted_cars += 1
                car_ids.add(car_id)
                print(f"Xe {car_id} đã đi qua vạch. Tổng số xe: {counted_cars}")
            
            # Vẽ box quanh xe và hiển thị số lượng đếm
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    # Hiển thị số lượng xe đã đếm trong khung hình hiện tại
    cv2.putText(frame, f"Counted Cars: {counted_cars}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Inference on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results.render()[0]  # YOLOv5 render method returns list of frames

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Optionally display the frame
    cv2.imshow('Video', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
