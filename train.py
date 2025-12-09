import json
from pathlib import Path
from ultralytics import YOLO

# Định nghĩa đường dẫn
PROJECT_DIR = "/content/drive/MyDrive/Chinsu_Project"
DATASET_DIR = f"{PROJECT_DIR}/dataset"

# Xử lý đường dẫn root (giữ nguyên logic của bạn)
try:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
except NameError:
    project_root = Path(PROJECT_DIR)

config_file = project_root / "config.json"

# Đọc cấu hình từ file config.json
with open(config_file, 'r') as f:
    config = json.load(f)

image_size = config["image_size"]
batch_size = config["batch_size"]
epochs = config["epochs"]

# Khởi tạo mô hình YOLO
model = YOLO("yolov8n.pt")

# Bắt đầu quá trình huấn luyện (Training)
model.train(
    data = f"{DATASET_DIR}/data.yaml",
    epochs = epochs,
    imgsz = image_size,
    batch = batch_size,
    device = 0,
    project=f"{PROJECT_DIR}/runs",
    name="model_detect",
)

print(f"Training xong! Kết quả lưu tại: {PROJECT_DIR}/runs/model_detect")
