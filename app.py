import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

# --- Your original UI Design & Layout ---
st.set_page_config(page_title="🚦 Road Safety Monitoring 🚦", layout="wide")
st.title("🚦 Road Safety Monitoring System 🛣️")
st.sidebar.header("⚙️ Settings Panel")

# Sidebar controls
show_tracking = st.sidebar.checkbox("Enable Tracking 🕵️‍♂️", True)
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold 🔍", 0.1, 1.0, 0.3)
show_depth = st.sidebar.checkbox("Enable Depth Estimation 🌊", True)

# Video upload or webcam
video_file = st.file_uploader("Upload a Video 🎥 (mp4, avi)", type=["mp4", "avi"])
if video_file:
    cap = cv2.VideoCapture(video_file)
else:
    st.info("Using webcam by default 📷")
    cap = cv2.VideoCapture(0)

# --- Load Models and Tracker (cached) ---
@st.cache_resource(show_spinner=True)
def load_models():
    # Load YOLO model
    model = YOLO("FinalModel_yolov8.pt")  # Your trained model file
    
    # Load BYTETracker
    tracker = BYTETracker(track_thresh=confidence_threshold, match_thresh=0.8, track_buffer=30, frame_rate=30)
    
    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # or "MiDaS_small"
    midas.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    
    # MiDaS transforms
    transform = Compose([
        Resize(384, 384),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    return model, tracker, midas, transform, device

model, tracker, midas, midas_transform, device = load_models()

# Placeholder for video frames display
frame_window = st.image([])
depth_window = st.image([]) if show_depth else None

st.write("### ▶️ Video Output with Tracking and Depth:")

def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    input_tensor = midas_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    # Normalize for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
    depth_colormap = (normalized_depth * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_MAGMA)
    return depth_colormap

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ No more frames or video ended.")
        break

    # Run detection with the current confidence threshold
    results = model(frame)[0]
    
    # Filter detections by confidence threshold
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    filtered_indices = scores >= confidence_threshold
    boxes = boxes[filtered_indices]
    scores = scores[filtered_indices]

    # Prepare detections for BYTETracker
    if len(boxes) > 0 and show_tracking:
        dets = np.concatenate([boxes, scores[:, None]], axis=1)
        online_targets = tracker.update(dets)
    else:
        online_targets = tracker.update(np.empty((0, 5)))

    # Draw boxes and IDs on the frame
    for track in online_targets:
        bbox = track.tlbr.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track.track_id} 🆔", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show depth map if enabled
    if show_depth:
        depth_map = estimate_depth(frame)
        depth_window.image(depth_map, channels="BGR")

    # Show the frame in the app
    frame_window.image(frame, channels="BGR")

cap.release()
st.success("🎉 Video processing completed!")
