import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from PIL import Image
from collections import defaultdict, deque
import time
import math
from datetime import datetime, timedelta

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

st.set_page_config(
    page_title="Road Safety Monitoring",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SafetyAnalyzer:
    def __init__(self, fps=30):
        self.prev_positions = {}
        self.fps = fps
        self.TAILGATING_TIME = 2
        self.PEDESTRIAN_DISTANCE = 1.0
        self.COLLISION_WARNING_TIME = 3
        self.SPEED_ESTIMATION_FRAMES = 5
        self.vehicle_history = defaultdict(lambda: deque(maxlen=self.SPEED_ESTIMATION_FRAMES))
        self.pedestrian_history = defaultdict(lambda: deque(maxlen=self.SPEED_ESTIMATION_FRAMES))
        
    def analyze_frame(self, frame, detections):
        safety_status = "Safe"
        violations = []
        collision_warning = False
        collision_detected = False
        
        vehicles = []
        pedestrians = []
        
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']
            track_id = detection.get('track_id', 0)
            
            if class_id == 0:
                vehicles.append({
                    'id': track_id,
                    'bbox': bbox,
                    'confidence': confidence
                })
            elif class_id == 1:
                pedestrians.append({
                    'id': track_id,
                    'bbox': bbox,
                    'confidence': confidence
                })
        
        for ped in pedestrians:
            for veh in vehicles:
                distance = self.calculate_pixel_distance(ped['bbox'], veh['bbox'])
                real_distance = distance * 0.01  
                
                if real_distance <= 0.2:
                    safety_status = "COLLISION DETECTED"
                    violations.append("COLLISION DETECTED - Vehicle and pedestrian collision!")
                    collision_detected = True
                elif real_distance < self.PEDESTRIAN_DISTANCE:
                    ped_speed = 2.0
                    veh_speed = 13.89
                    relative_speed = abs(veh_speed - ped_speed)
                    
                    if relative_speed > 0:
                        time_to_collision = real_distance / relative_speed
                        
                        if time_to_collision <= self.COLLISION_WARNING_TIME:
                            safety_status = "COLLISION WARNING"
                            violations.append(f"⚠COLLISION WARNING - {time_to_collision:.1f}s until potential impact!")
                            collision_warning = True
                        else:
                            safety_status = "Unsafe"
                            violations.append(f"Vehicle too close to pedestrian ({real_distance:.2f}m)")
                    else:
                        safety_status = "Unsafe"
                        violations.append(f"Vehicle too close to pedestrian ({real_distance:.2f}m)")
        
        for i, veh1 in enumerate(vehicles):
            for j, veh2 in enumerate(vehicles[i+1:], i+1):
                distance = self.calculate_pixel_distance(veh1['bbox'], veh2['bbox'])
                real_distance = distance * 0.01
                
                if real_distance <= 0.3:
                    if not collision_detected:  
                        safety_status = "COLLISION DETECTED"
                    violations.append("COLLISION DETECTED - Vehicle-to-vehicle collision!")
                    collision_detected = True
                else:
                    assumed_speed = 13.89
                    time_gap = real_distance / assumed_speed if assumed_speed > 0 else 0
                    
                    if time_gap <= self.COLLISION_WARNING_TIME and real_distance < 5:
                        if not collision_detected and not collision_warning:
                            safety_status = "COLLISION WARNING"
                        violations.append(f"⚠COLLISION WARNING - {time_gap:.1f}s gap between vehicles!")
                        collision_warning = True
                    elif time_gap < self.TAILGATING_TIME and real_distance < 15: 
                        if safety_status == "Safe":
                            safety_status = "Unsafe"
                        violations.append(f"Tailgating detected ({time_gap:.2f}s gap, {real_distance:.2f}m)")
        
        return safety_status, violations
    
    def calculate_pixel_distance(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        x1, y1, x2, y2 = bbox2
        center2 = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

@st.cache_resource
def load_yolo_model():
    if not YOLO_AVAILABLE:
        st.info("🔄 YOLO not available - Running in demo mode")
        return "demo_mode"
        
    try:
        model = YOLO('FinalModel_yolov8.pt') 
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def run_detection(model, image, conf_threshold=0.5, iou_threshold=0.45):
    if not YOLO_AVAILABLE or model == "demo_mode" or model is None:
        height, width = image.shape[:2] if len(image.shape) > 2 else (400, 600)
        return [
            {'bbox': [width*0.1, height*0.3, width*0.4, height*0.7], 'confidence': 0.85, 'class_id': 0, 'track_id': 1},
            {'bbox': [width*0.6, height*0.4, width*0.75, height*0.8], 'confidence': 0.75, 'class_id': 1, 'track_id': 2}
        ]
    
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    if cls == 0:  
                        class_id = 1 
                    elif cls in [2, 5, 7]: 
                        class_id = 0  
                    else:
                        continue  
                    
                    detections.append({
                        'bbox': box,
                        'confidence': float(conf),
                        'class_id': class_id,
                        'track_id': i
                    })
        
        return detections
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return []

def draw_safety_annotations(image, detections, safety_status, violations):
    if not CV2_AVAILABLE:
        return image
        
    import cv2
    
    if isinstance(image, Image.Image):
        img_copy = np.array(image)
    else:
        img_copy = image.copy()
    
    colors = {
        'Vehicle': (0, 255, 0),
        'Pedestrian': (255, 0, 0),
        'Unsafe': (0, 165, 255),
        'COLLISION WARNING': (0, 255, 255),
        'COLLISION DETECTED': (0, 0, 255)
    }
    
    if safety_status == "COLLISION DETECTED":
        status_color = colors['COLLISION DETECTED']
        bbox_color = colors['COLLISION DETECTED']
    elif safety_status == "COLLISION WARNING":
        status_color = colors['COLLISION WARNING'] 
        bbox_color = colors['COLLISION WARNING']
    elif safety_status == "Unsafe":
        status_color = colors['Unsafe']
        bbox_color = colors['Unsafe']
    else:
        status_color = (0, 255, 0)  
        bbox_color = None
    
    for detection in detections:
        bbox = detection['bbox']
        class_name = 'Vehicle' if detection['class_id'] == 0 else 'Pedestrian'
        confidence = detection['confidence']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        color = bbox_color if bbox_color else colors[class_name]
        
        thickness = 4 if safety_status in ["COLLISION DETECTED", "COLLISION WARNING"] else 2
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    font_scale = 1.2 if safety_status in ["COLLISION DETECTED", "COLLISION WARNING"] else 1
    thickness = 3 if safety_status in ["COLLISION DETECTED", "COLLISION WARNING"] else 2
    
    cv2.putText(img_copy, f"Status: {safety_status}", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)
    
    for i, violation in enumerate(violations):
        y_pos = 80 + i * 35
        text_size = cv2.getTextSize(violation[:60], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_copy, (8, y_pos - 25), (text_size[0] + 15, y_pos + 5), (0, 0, 0), -1)
        
        violation_color = (0, 0, 255) if "COLLISION" in violation else (0, 165, 255)
        cv2.putText(img_copy, violation[:60], (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, violation_color, 2)
    
    return img_copy

def main():
    st.title("🚗 Road Safety Monitoring")
    st.markdown("Reckless Driving Behavior Recognition For Road Safety Monitoring")
    
    st.session_state.model = load_yolo_model()
    
    real_time_detection_page()

def real_time_detection_page():
    st.header("Safety Detection")
    
    st.subheader("Safety Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**2-Second Rule**: Maintain 2-3 second following distance")
    with col2:
        st.warning("**⚠️ Collision Warning**: 3-second early warning system")
    with col3:  
        st.error("**🚨 Collision Alert**: Immediate collision detection")
    
    st.subheader("Detection Parameters")
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    with col2:
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45)
    
    st.subheader("Input Source")
    input_type = st.radio("Choose input type:", ["Upload Image", "Upload Video"])
    
    safety_analyzer = SafetyAnalyzer()
    
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                with st.spinner("Running detection..."):
                    detections = run_detection(st.session_state.model, img_array, 
                                             confidence_threshold, iou_threshold)
                    
                    safety_status, violations = safety_analyzer.analyze_frame(img_array, detections)
                    
                    if CV2_AVAILABLE:
                        annotated_img = draw_safety_annotations(img_array, detections, safety_status, violations)
                        st.image(annotated_img, use_column_width=True)
                    else:
                        st.image(img_array, use_column_width=True)
                        st.warning("OpenCV not available - showing original image")
                
                st.subheader("Safety Analysis")
                if safety_status == "COLLISION DETECTED":
                    st.error("🚨 COLLISION DETECTED - Immediate danger!")
                elif safety_status == "COLLISION WARNING":
                    st.warning("⚠️ COLLISION WARNING - Take immediate action!")
                elif safety_status == "Safe":
                    st.success("✅ Safe - No violations detected")
                else:
                    st.warning("⚠️ Unsafe - Violations detected")
                
                if violations:
                    st.subheader("Violations Detected")
                    for violation in violations:
                        if "COLLISION DETECTED" in violation:
                            st.error(f"{violation}")
                        elif "COLLISION WARNING" in violation:
                            st.warning(f"{violation}")
                        else:
                            st.write(f"• {violation}")
                
                if detections:
                    st.subheader("Detection Details")
                    detection_data = []
                    for i, det in enumerate(detections):
                        class_name = 'Vehicle' if det['class_id'] == 0 else 'Pedestrian'
                        detection_data.append({
                            'Object': class_name,
                            'Confidence': f"{det['confidence']:.2f}",
                            'Position': f"({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f})"
                        })
                    
                    df = pd.DataFrame(detection_data)
                    st.dataframe(df, use_container_width=True)
    
    elif input_type == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            col1, col2 = st.columns(2)
            with col1:
                frames_to_process = st.slider("Number of frames to process", 1, 20, 5)
            with col2:
                frame_interval = st.slider("Frame interval (process every nth frame)", 1, 30, 5)
            
            if st.button("🎬 Process Video Frames"):
                if not CV2_AVAILABLE:
                    st.error("OpenCV is required for video processing.")
                    return
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name
                
                import cv2
                with st.spinner("Processing video frames..."):
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    st.info(f"📹 Video: {frame_count} frames, {fps:.1f} FPS")
                    
                    progress_bar = st.progress(0)
                    safety_results = []
                    processed_frames = []
                    
                    frame_idx = 0
                    processed_count = 0
                    
                    while cap.isOpened() and processed_count < frames_to_process:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_idx % frame_interval == 0:
                            detections = run_detection(st.session_state.model, frame, 
                                                     confidence_threshold, iou_threshold)
                            safety_status, violations = safety_analyzer.analyze_frame(frame, detections)
                            
                            annotated_frame = draw_safety_annotations(frame, detections, safety_status, violations)
                            
                            safety_results.append({
                                'Frame': frame_idx,
                                'Status': safety_status,
                                'Violations': len(violations),
                                'Objects': len(detections),
                                'Vehicles': len([d for d in detections if d['class_id'] == 0]),
                                'Pedestrians': len([d for d in detections if d['class_id'] == 1]),
                                'Violation_Details': violations
                            })
                            
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            processed_frames.append((frame_idx, annotated_frame_rgb, safety_status, violations))
                            
                            processed_count += 1
                            progress_bar.progress(processed_count / frames_to_process)
                        
                        frame_idx += 1
                    
                    cap.release()
                    
                    if safety_results:
                        st.success(f"✅ Processed {len(safety_results)} frames!")
                        
                        collision_frames = len([r for r in safety_results if r['Status'] == 'COLLISION DETECTED'])
                        warning_frames = len([r for r in safety_results if r['Status'] == 'COLLISION WARNING'])
                        unsafe_frames = len([r for r in safety_results if r['Status'] == 'Unsafe'])
                        safe_frames = len([r for r in safety_results if r['Status'] == 'Safe'])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Collisions", collision_frames, delta="Critical" if collision_frames > 0 else None)
                        with col2:
                            st.metric("Warnings", warning_frames, delta="High Risk" if warning_frames > 0 else None)
                        with col3:
                            st.metric("Unsafe", unsafe_frames)
                        with col4:
                            st.metric("Safe", safe_frames)
                        
                        st.subheader("🎞️ Processed Frames with Advanced Safety Analysis")
                        
                        for i, (frame_num, frame_img, status, violations) in enumerate(processed_frames):
                            st.markdown(f"### Frame {frame_num}")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.image(frame_img, use_column_width=True)
                            
                            with col2:
                                if status == "COLLISION DETECTED":
                                    st.error(f"{status}")
                                elif status == "COLLISION WARNING":
                                    st.warning(f"{status}")
                                elif status == "Safe":
                                    st.success(f"{status}")
                                else:
                                    st.warning(f"{status}")
                                
                                frame_result = safety_results[i]
                                st.write(f"**Objects Detected:** {frame_result['Objects']}")
                                st.write(f"**Vehicles:** {frame_result['Vehicles']}")
                                st.write(f"**Pedestrians:** {frame_result['Pedestrians']}")
                                
                                if violations:
                                    st.write("**Safety Alerts:**")
                                    for violation in violations:
                                        if "COLLISION DETECTED" in violation:
                                            st.error(f"{violation}")
                                        elif "COLLISION WARNING" in violation:
                                            st.warning(f"{violation}")
                                        else:
                                            st.write(f"• {violation}")
                                else:
                                    st.write("**✅ No safety violations detected**")
                            
                            st.divider()
                        
                        st.subheader("📊 Frame Analysis Summary")
                        df_results = pd.DataFrame(safety_results)
                        df_display = df_results.drop('Violation_Details', axis=1)
                        st.dataframe(df_display, use_container_width=True)
                        
                        csv = df_display.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis Results",
                            data=csv,
                            file_name="video_safety_analysis_results.csv",
                            mime="text/csv"
                        )



if __name__ == "__main__":
    main()
