import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import random
import torch
import tempfile

# For MiDaS depth estimation
class MiDaSDepthEstimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "DPT_Small"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def estimate_depth(self, image):
        input_batch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        return depth_map

# Safety analyzer for tailgating and pedestrian distance
class SafetyAnalyzer:
    def __init__(self, tailgating_time=2, pedestrian_distance=1.0, vehicle_speed=13.89):
        self.TAILGATING_TIME = tailgating_time  # seconds
        self.PEDESTRIAN_DISTANCE = pedestrian_distance  # meters
        self.vehicle_speed = vehicle_speed  # m/s (default 50 km/h)

    def calculate_distance(self, bbox1, bbox2, depth_map):
        x1, y1, x2, y2 = bbox1
        x1c, y1c = int((x1 + x2) / 2), int((y1 + y2) / 2)
        x2c, y2c = int((bbox2[0] + bbox2[2]) / 2), int((bbox2[1] + bbox2[3]) / 2)

        # Clamp coords to image size
        h, w = depth_map.shape
        x1c, y1c = np.clip([x1c, y1c], [0,0], [w-1,h-1])
        x2c, y2c = np.clip([x2c, y2c], [0,0], [w-1,h-1])

        d1 = depth_map[y1c, x1c]
        d2 = depth_map[y2c, x2c]

        # Euclidean distance approximation in meters (depth difference)
        return abs(d1 - d2)

    def calculate_time_gap(self, distance):
        if self.vehicle_speed <= 0:
            return float('inf')
        return distance / self.vehicle_speed

    def analyze_safety(self, vehicles, pedestrians, depth_map):
        # vehicles and pedestrians are list of (id, bbox)
        # bbox = (x1, y1, x2, y2)
        for v_id, v_bbox in vehicles:
            for p_id, p_bbox in pedestrians:
                dist = self.calculate_distance(v_bbox, p_bbox, depth_map)
                if dist < self.PEDESTRIAN_DISTANCE:
                    return "Unsafe: Too close to pedestrian"

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                dist = self.calculate_distance(vehicles[i][1], vehicles[j][1], depth_map)
                time_gap = self.calculate_time_gap(dist)
                if time_gap < self.TAILGATING_TIME:
                    return "Unsafe: Tailgating detected"

        return "Safe"

# Load YOLO model
model = YOLO("FinalModel_yolov8.pt")  # Your trained model with pedestrian=0, vehicle=1

st.set_page_config(page_title="🚗 Reckless Driving Detector", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #d62728;'>🚦Reckless Driving Behavior Recognition For Road Safety Monitoring🚦</h1>
    <h4 style='text-align: center;'>⚠️ Road Safety Monitoring System ⚠️</h4><hr> 
    """,
    unsafe_allow_html=True
)

st.sidebar.title("📂 Choose Input Type")
media_type = st.sidebar.radio("Select input 🎯:", ("🖼️ Image", "🎥 Video"))

def get_class_colors(class_names):
    random.seed(42)
    return {name: [random.randint(0, 255) for _ in range(3)] for name in class_names}

class_colors = get_class_colors(model.names.values())

custom_colors = {
    "Vehicle": (137, 207, 240),     # Light Blue
    "Pedestrian": (255, 179, 71),   # Light Orange
}

def draw_boxes(image_np, results, safety_status=None):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[class_id] if hasattr(results, 'names') else model.names[class_id]
        label_text = f"{label} {conf:.2f}"
        color = custom_colors.get(label, (255, 255, 255))
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_np, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image_np, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if safety_status:
        cv2.putText(image_np, f"Safety Status: {safety_status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if "Unsafe" in safety_status else (0, 255, 0), 3)
    return image_np

depth_estimator = MiDaSDepthEstimator()
safety_analyzer = SafetyAnalyzer()

if media_type == "🖼️ Image":
    uploaded_image = st.sidebar.file_uploader("Upload your image", type=["jpg", "jpeg", "png"]) 
    if uploaded_image:
        from PIL import Image
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        results = model.track(image_np, tracker="bytetrack.yaml")[0]

        depth_map = depth_estimator.estimate_depth(image_np)

        boxes = results.boxes.xyxy.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)
        ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else np.arange(len(boxes))

        vehicles = []
        pedestrians = []

        for box, cls_id, track_id in zip(boxes, clss, ids):
            x1, y1, x2, y2 = map(int, box)
            if cls_id == 1:
                vehicles.append((track_id, (x1, y1, x2, y2)))
            elif cls_id == 0:
                pedestrians.append((track_id, (x1, y1, x2, y2)))

        safety_status = safety_analyzer.analyze_safety(vehicles, pedestrians, depth_map)

        processed = draw_boxes(image_np.copy(), results, safety_status)
        st.image(processed, caption="🖼️ Detection Result — Stay safe out there!", use_container_width=True)

elif media_type == "🎥 Video":
    uploaded_video = st.sidebar.file_uploader("Upload your video", type=["mp4", "avi", "mov"]) 
    if uploaded_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()
        st.info("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, tracker="bytetrack.yaml")[0]
            depth_map = depth_estimator.estimate_depth(frame)

            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else np.arange(len(boxes))

            vehicles = []
            pedestrians = []

            for box, cls_id, track_id in zip(boxes, clss, ids):
                x1, y1, x2, y2 = map(int, box)
                if cls_id == 1:
                    vehicles.append((track_id, (x1, y1, x2, y2)))
                elif cls_id == 0:
                    pedestrians.append((track_id, (x1, y1, x2, y2)))

            safety_status = safety_analyzer.analyze_safety(vehicles, pedestrians, depth_map)

            annotated = draw_boxes(frame.copy(), results, safety_status)
            stframe.image(annotated, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Video processing complete! Drive safe! 🚗💨")
