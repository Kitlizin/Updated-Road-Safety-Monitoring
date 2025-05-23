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
        self.PEDESTRIAN_DISTANCE = 2.0  # Updated to 2 meters for safety
        self.PEDESTRIAN_WARNING_DISTANCE = 3.0  # Additional early warning at 3 meters
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
        
        # Separate vehicles and pedestrians from detections
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
        
        # Debug output
        if len(vehicles) > 0 or len(pedestrians) > 0:
            print(f"Frame Analysis: {len(vehicles)} vehicles, {len(pedestrians)} pedestrians detected")
        
        # Enhanced Pedestrian Safety Analysis
        for ped in pedestrians:
            for veh in vehicles:
                distance = self.calculate_pixel_distance(ped['bbox'], veh['bbox'])
                # Improved distance conversion - calibrated for typical road scenarios
                real_distance = distance * 0.05  # Adjusted conversion factor
                
                print(f"Vehicle-Pedestrian distance: {real_distance:.2f}m (pixel distance: {distance:.1f})")
                
                # Critical collision zone (0.5m or less)
                if real_distance <= 0.5:
                    safety_status = "COLLISION DETECTED"
                    violations.append("🚨 CRITICAL: Vehicle-pedestrian collision imminent!")
                    collision_detected = True
                
                # Immediate danger zone (0.5m - 1m)
                elif real_distance <= 1.0:
                    if not collision_detected:
                        safety_status = "COLLISION WARNING"
                    violations.append(f"🚨 DANGER: Vehicle extremely close to pedestrian ({real_distance:.1f}m)!")
                    collision_warning = True
                
                # Unsafe zone (1m - 2m) - Your specified 2-meter safety distance
                elif real_distance <= self.PEDESTRIAN_DISTANCE:
                    if not collision_detected and not collision_warning:
                        safety_status = "Unsafe"
                    violations.append(f"⚠️ UNSAFE: Vehicle too close to pedestrian ({real_distance:.1f}m) - Maintain 2m minimum!")
                
                # Warning zone (2m - 3m) - Early warning
                elif real_distance <= self.PEDESTRIAN_WARNING_DISTANCE:
                    if safety_status == "Safe":
                        safety_status = "Caution"
                    violations.append(f"⚠️ CAUTION: Vehicle approaching pedestrian ({real_distance:.1f}m) - Exercise caution!")
                
                # Time-to-collision calculation for moving objects
                if real_distance <= self.PEDESTRIAN_WARNING_DISTANCE:
                    # Estimated speeds (could be improved with actual speed detection)
                    ped_speed = 1.5  # Average walking speed m/s
                    veh_speed = 10.0  # Conservative vehicle speed m/s
                    relative_speed = abs(veh_speed - ped_speed)
                    
                    if relative_speed > 0:
                        time_to_collision = real_distance / relative_speed
                        
                        if time_to_collision <= 1.0:  # Less than 1 second
                            if not collision_detected:
                                safety_status = "COLLISION WARNING"
                            violations.append(f"🚨 IMMEDIATE DANGER: {time_to_collision:.1f}s to potential collision!")
                            collision_warning = True
                        elif time_to_collision <= 2.0:  # Less than 2 seconds
                            if not collision_detected and not collision_warning:
                                safety_status = "Unsafe"
                            violations.append(f"⚠️ WARNING: {time_to_collision:.1f}s to potential collision!")
        
        # Vehicle-to-Vehicle Analysis (existing logic with improvements)
        for i, veh1 in enumerate(vehicles):
            for j, veh2 in enumerate(vehicles[i+1:], i+1):
                distance = self.calculate_pixel_distance(veh1['bbox'], veh2['bbox'])
                real_distance = distance * 0.05  # Consistent conversion factor
                
                # Vehicle collision detection
                if real_distance <= 0.8:  # Very close vehicles
                    if not collision_detected:
                        safety_status = "COLLISION DETECTED"
                    violations.append("🚨 COLLISION DETECTED: Vehicle-to-vehicle collision!")
                    collision_detected = True
                else:
                    # Estimate time gap based on assumed speed
                    assumed_speed = 15.0  # m/s (about 55 km/h)
                    time_gap = real_distance / assumed_speed if assumed_speed > 0 else 0
                    
                    # Collision warning for vehicles
                    if time_gap <= 1.0 and real_distance < 8:
                        if not collision_detected and not collision_warning:
                            safety_status = "COLLISION WARNING"
                        violations.append(f"🚨 VEHICLE WARNING: {time_gap:.1f}s gap between vehicles!")
                        collision_warning = True
                    # Tailgating detection
                    elif time_gap < self.TAILGATING_TIME and real_distance < 20:
                        if safety_status == "Safe":
                            safety_status = "Unsafe"
                        violations.append(f"⚠️ TAILGATING: {time_gap:.1f}s gap ({real_distance:.1f}m) - Maintain 2-second rule!")
        
        return safety_status, violations
    
    def calculate_pixel_distance(self, bbox1, bbox2):
        """Calculate Euclidean distance between centers of two bounding boxes"""
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
        model = YOLO('yolov8n.pt') 
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def run_detection(model, image, conf_threshold=0.5, iou_threshold=0.45):
    if not YOLO_AVAILABLE or model == "demo_mode" or model is None:
        # Enhanced demo mode with more realistic scenarios
        height, width = image.shape[:2] if len(image.shape) > 2 else (400, 600)
        return [
            # Vehicle close to pedestrian scenario
            {'bbox': [width*0.1, height*0.3, width*0.4, height*0.7], 'confidence': 0.85, 'class_id': 0, 'track_id': 1},
            {'bbox': [width*0.35, height*0.2, width*0.5, height*0.6], 'confidence': 0.75, 'class_id': 1, 'track_id': 2},
            # Additional vehicle
            {'bbox': [width*0.6, height*0.4, width*0.85, height*0.8], 'confidence': 0.80, 'class_id': 0, 'track_id': 3}
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
                    
                    # Map YOLO classes to our classes
                    if cls == 0:  # person in YOLO
                        class_id = 1  # pedestrian in our system
                    elif cls in [2, 5, 7]:  # car, bus, truck in YOLO
                        class_id = 0  # vehicle in our system
                    else:
                        continue  # skip other classes
                    
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
    
    # Enhanced color scheme
    colors = {
        'Vehicle': (0, 255, 0),      # Green
        'Pedestrian': (255, 0, 0),   # Blue
        'Safe': (0, 255, 0),         # Green
        'Caution': (0, 255, 255),    # Yellow
        'Unsafe': (0, 165, 255),     # Orange
        'COLLISION WARNING': (0, 255, 255),  # Yellow
        'COLLISION DETECTED': (0, 0, 255)    # Red
    }
    
    # Determine status color and bbox highlighting
    if safety_status == "COLLISION DETECTED":
        status_color = colors['COLLISION DETECTED']
        bbox_color = colors['COLLISION DETECTED']
    elif safety_status == "COLLISION WARNING":
        status_color = colors['COLLISION WARNING'] 
        bbox_color = colors['COLLISION WARNING']
    elif safety_status == "Unsafe":
        status_color = colors['Unsafe']
        bbox_color = colors['Unsafe']
    elif safety_status == "Caution":
        status_color = colors['Caution']
        bbox_color = colors['Caution']
    else:
        status_color = colors['Safe']
        bbox_color = None
    
    # Draw bounding boxes with enhanced styling
    for detection in detections:
        bbox = detection['bbox']
        class_name = 'Vehicle' if detection['class_id'] == 0 else 'Pedestrian'
        confidence = detection['confidence']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Use status color for critical situations, otherwise use class color
        color = bbox_color if bbox_color else colors[class_name]
        
        # Thicker lines for critical situations
        thickness = 4 if safety_status in ["COLLISION DETECTED", "COLLISION WARNING"] else 2
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Enhanced label with better visibility
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(img_copy, label, (x1 + 5, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Enhanced status display
    font_scale = 1.5 if safety_status in ["COLLISION DETECTED", "COLLISION WARNING"] else 1.2
    thickness = 4 if safety_status in ["COLLISION DETECTED", "COLLISION WARNING"] else 3
    
    # Add background for better visibility
    status_text = f"Status: {safety_status}"
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    cv2.rectangle(img_copy, (5, 5), (text_size[0] + 20, text_size[1] + 25), (0, 0, 0), -1)
    cv2.putText(img_copy, status_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)
    
    # Enhanced violation display
    for i, violation in enumerate(violations):
        y_pos = 80 + i * 40
        # Clean violation text (remove emojis for OpenCV)
        clean_violation = violation.replace("⚠️", "").replace("🚨", "").replace("⚠", "").strip()
        
        # Truncate long messages
        display_text = clean_violation[:65] + "..." if len(clean_violation) > 65 else clean_violation
        
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img_copy, (5, y_pos - 30), (text_size[0] + 20, y_pos + 10), (0, 0, 0), -1)
        
        # Color code violations
        if "COLLISION" in violation or "CRITICAL" in violation or "DANGER" in violation:
            violation_color = (0, 0, 255)  # Red
        elif "WARNING" in violation or "UNSAFE" in violation:
            violation_color = (0, 165, 255)  # Orange
        elif "CAUTION" in violation:
            violation_color = (0, 255, 255)  # Yellow
        else:
            violation_color = (255, 255, 255)  # White
            
        cv2.putText(img_copy, display_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, violation_color, 2)
    
    return img_copy

def main():
    st.title("🚗 Road Safety Monitoring System")
    st.markdown("**Advanced Reckless Driving Behavior Recognition For Road Safety Monitoring**")
    
    # Load model
    st.session_state.model = load_yolo_model()
    
    # Main detection interface
    real_time_detection_page()

def real_time_detection_page():
    st.header("🎯 Advanced Safety Detection")
    
    # Enhanced safety information
    st.subheader("📋 Safety Guidelines & Detection Zones")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("**✅ Safe Zone**: 3m+ from pedestrians")
    with col2:
        st.warning("**⚠️ Caution Zone**: 2-3m from pedestrians")
    with col3:
        st.error("**🚨 Danger Zone**: 1-2m from pedestrians")
    with col4:
        st.error("**💀 Critical Zone**: <1m from pedestrians")
    
    # Additional safety info
    st.info("**🚶‍♂️ Pedestrian Safety**: Maintain minimum 2-meter distance from pedestrians at all times")
    st.info("**🚗 Vehicle Safety**: Follow 2-second rule for vehicle spacing")
    
    # Detection parameters
    st.subheader("⚙️ Detection Parameters")
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, help="Higher values = more selective detection")
    with col2:
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, help="Higher values = less overlapping detections")
    
    # Input source selection
    st.subheader("📤 Input Source")
    input_type = st.radio("Choose input type:", ["Upload Image", "Upload Video"])
    
    safety_analyzer = SafetyAnalyzer()
    
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🔍 Detection Results")
                
                with st.spinner("Running advanced safety analysis..."):
                    detections = run_detection(st.session_state.model, img_array, 
                                             confidence_threshold, iou_threshold)
                    
                    safety_status, violations = safety_analyzer.analyze_frame(img_array, detections)
                    
                    if CV2_AVAILABLE:
                        annotated_img = draw_safety_annotations(img_array, detections, safety_status, violations)
                        st.image(annotated_img, use_column_width=True)
                    else:
                        st.image(img_array, use_column_width=True)
                        st.warning("OpenCV not available - showing original image")
                
                # Enhanced safety analysis display
                st.subheader("🛡️ Safety Analysis")
                if safety_status == "COLLISION DETECTED":
                    st.error("🚨 **COLLISION DETECTED** - Immediate danger!")
                elif safety_status == "COLLISION WARNING":
                    st.error("⚠️ **COLLISION WARNING** - Take immediate action!")
                elif safety_status == "Unsafe":
                    st.warning("⚠️ **UNSAFE CONDITIONS** - Violations detected")
                elif safety_status == "Caution":
                    st.warning("⚠️ **EXERCISE CAUTION** - Monitor situation closely")
                elif safety_status == "Safe":
                    st.success("✅ **SAFE** - No violations detected")
                else:
                    st.info(f"**Status**: {safety_status}")
                
                # Enhanced violations display
                if violations:
                    st.subheader("🚨 Safety Alerts")
                    for i, violation in enumerate(violations, 1):
                        if "COLLISION DETECTED" in violation or "CRITICAL" in violation:
                            st.error(f"**{i}.** {violation}")
                        elif "COLLISION WARNING" in violation or "DANGER" in violation:
                            st.error(f"**{i}.** {violation}")
                        elif "WARNING" in violation or "UNSAFE" in violation:
                            st.warning(f"**{i}.** {violation}")
                        elif "CAUTION" in violation:
                            st.warning(f"**{i}.** {violation}")
                        else:
                            st.write(f"**{i}.** {violation}")
                
                # Detection details
                if detections:
                    st.subheader("📊 Detection Details")
                    detection_data = []
                    for i, det in enumerate(detections, 1):
                        class_name = 'Vehicle' if det['class_id'] == 0 else 'Pedestrian'
                        detection_data.append({
                            'ID': i,
                            'Object Type': class_name,
                            'Confidence': f"{det['confidence']:.3f}",
                            'Center Position': f"({det['bbox'][0] + det['bbox'][2]}/2:.0f, {det['bbox'][1] + det['bbox'][3]}/2:.0f})",
                            'Size': f"{det['bbox'][2] - det['bbox'][0]:.0f} × {det['bbox'][3] - det['bbox'][1]:.0f}"
                        })
                    
                    df = pd.DataFrame(detection_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    vehicle_count = len([d for d in detections if d['class_id'] == 0])
                    pedestrian_count = len([d for d in detections if d['class_id'] == 1])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Objects", len(detections))
                    with col2:
                        st.metric("Vehicles", vehicle_count)
                    with col3:
                        st.metric("Pedestrians", pedestrian_count)
    
    elif input_type == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            col1, col2 = st.columns(2)
            with col1:
                frames_to_process = st.slider("Number of frames to process", 1, 30, 10)
            with col2:
                frame_interval = st.slider("Frame interval (process every nth frame)", 1, 30, 5)
            
            if st.button("🎬 Process Video with Advanced Analysis"):
                if not CV2_AVAILABLE:
                    st.error("OpenCV is required for video processing.")
                    return
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name
                
                import cv2
                with st.spinner("Processing video frames with advanced safety analysis..."):
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    st.info(f"📹 Video Info: {frame_count} frames, {fps:.1f} FPS")
                    
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
                            
                            # Enhanced result tracking
                            safety_results.append({
                                'Frame': frame_idx,
                                'Time (s)': round(frame_idx / fps, 2),
                                'Status': safety_status,
                                'Total Violations': len(violations),
                                'Objects': len(detections),
                                'Vehicles': len([d for d in detections if d['class_id'] == 0]),
                                'Pedestrians': len([d for d in detections if d['class_id'] == 1]),
                                'Critical_Alerts': len([v for v in violations if 'COLLISION' in v or 'CRITICAL' in v]),
                                'Violation_Details': violations
                            })
                            
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            processed_frames.append((frame_idx, annotated_frame_rgb, safety_status, violations))
                            
                            processed_count += 1
                            progress_bar.progress(processed_count / frames_to_process)
                        
                        frame_idx += 1
                    
                    cap.release()
                    
                    if safety_results:
                        st.success(f"✅ Successfully processed {len(safety_results)} frames!")
                        
                        # Enhanced summary statistics
                        collision_frames = len([r for r in safety_results if r['Status'] == 'COLLISION DETECTED'])
                        warning_frames = len([r for r in safety_results if r['Status'] == 'COLLISION WARNING'])
                        unsafe_frames = len([r for r in safety_results if r['Status'] == 'Unsafe'])
                        caution_frames = len([r for r in safety_results if r['Status'] == 'Caution'])
                        safe_frames = len([r for r in safety_results if r['Status'] == 'Safe'])
                        total_violations = sum([r['Total Violations'] for r in safety_results])
                        critical_alerts = sum([r['Critical_Alerts'] for r in safety_results])
                        
                        st.subheader("📊 Video Analysis Summary")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("🚨 Collisions", collision_frames, 
                                     delta="CRITICAL" if collision_frames > 0 else None,
                                     delta_color="inverse")
                        with col2:
                            st.metric("⚠️ Warnings", warning_frames,
                                     delta="HIGH RISK" if warning_frames > 0 else None,
                                     delta_color="inverse")
                        with col3:
                            st.metric("😟 Unsafe", unsafe_frames)
                        with col4:
                            st.metric("😐 Caution", caution_frames)
                        with col5:
                            st.metric("😊 Safe", safe_frames)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Violations", total_violations)
                        with col2:
                            st.metric("Critical Alerts", critical_alerts)
                        
                        # Risk assessment
                        risk_score = (collision_frames * 10 + warning_frames * 5 + unsafe_frames * 2 + caution_frames * 1) / len(safety_results)
                        
                        if risk_score >= 8:
                            st.error(f"🚨 **EXTREME RISK** - Risk Score: {risk_score:.1f}/10")
                        elif risk_score >= 5:
                            st.error(f"⚠️ **HIGH RISK** - Risk Score: {risk_score:.1f}/10")
                        elif risk_score >= 2:
                            st.warning(f"⚠️ **MODERATE RISK** - Risk Score: {risk_score:.1f}/10")
                        else:
                            st.success(f"✅ **LOW RISK** - Risk Score: {risk_score:.1f}/10")
                        
                        # Detailed frame analysis
                        st.subheader("🎞️ Detailed Frame Analysis")
                        
                        for i, (frame_num, frame_img, status, violations) in enumerate(processed_frames):
                            with st.expander(f"Frame {frame_num} - {status} ({len(violations)} alerts)", 
                                           expanded=(status in ["COLLISION DETECTED", "COLLISION WARNING"])):
                                
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.image(frame_img, use_column_width=True)
                                
                                with col2:
                                    # Status indicator
                                    if status == "COLLISION DETECTED":
                                        st.error(f"🚨 **{status}**")
                                    elif status == "COLLISION WARNING":
                                        st.error(f"⚠️ **{status}**")
                                    elif status == "Unsafe":
                                        st.warning(f"⚠️ **{status}**")
                                    elif status == "Caution":
                                        st.warning(f"⚠️ **{status}**")
                                    else:
                                        st.success(f"✅ **{status}**")
                                    
                                    # Frame statistics
                                    frame_result = safety_results[i]
                                    st.write(f"**Time:** {frame_result['Time (s)']}s")
                                    st.write(f"**Objects:** {frame_result['Objects']}")
                                    st.write(f"**Vehicles:** {frame_result['Vehicles']}")
                                    st.write(f"**Pedestrians:** {frame_result['Pedestrians']}")
                                    
                                    # Violation details
                                    if violations:
                                        st.write("**🚨 Safety Alerts:**")
                                        for j, violation in enumerate(violations, 1):
                                            clean_violation = violation.replace("🚨", "").replace("⚠️", "").replace("⚠", "").strip()
                                            if "COLLISION DETECTED" in violation or "CRITICAL" in violation:
                                                st.error(f"{j}. {clean_violation}")
                                            elif "COLLISION WARNING" in violation or "DANGER" in violation:
                                                st.error(f"{j}. {clean_violation}")
                                            elif "WARNING" in violation or "UNSAFE" in violation:
                                                st.warning(f"{j}. {clean_violation}")
                                            elif "CAUTION" in violation:
                                                st.warning(f"{j}. {clean_violation}")
                                            else:
                                                st.write(f"{j}. {clean_violation}")
                                    else:
                                        st.success("**✅ No safety violations detected**")
                        
                        # Data export and analysis
                        st.subheader("📈 Analysis Data & Export")
                        
                        # Create detailed DataFrame
                        df_results = pd.DataFrame(safety_results)
                        df_display = df_results.drop(['Violation_Details'], axis=1)
                        
                        # Add risk categorization
                        def categorize_risk(status):
                            if status == "COLLISION DETECTED":
                                return "CRITICAL"
                            elif status == "COLLISION WARNING":
                                return "HIGH"
                            elif status == "Unsafe":
                                return "MEDIUM"
                            elif status == "Caution":
                                return "LOW"
                            else:
                                return "MINIMAL"
                        
                        df_display['Risk_Level'] = df_display['Status'].apply(categorize_risk)
                        
                        st.dataframe(df_display, use_container_width=True)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv = df_display.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Analysis Results (CSV)",
                                data=csv,
                                file_name=f"video_safety_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Create detailed report
                            detailed_report = []
                            for result in safety_results:
                                detailed_report.append({
                                    'Frame': result['Frame'],
                                    'Time_s': result['Time (s)'],
                                    'Status': result['Status'],
                                    'Risk_Level': categorize_risk(result['Status']),
                                    'Total_Violations': result['Total Violations'],
                                    'Critical_Alerts': result['Critical_Alerts'],
                                    'Objects': result['Objects'],
                                    'Vehicles': result['Vehicles'],
                                    'Pedestrians': result['Pedestrians'],
                                    'Violations': '; '.join(result['Violation_Details'])
                                })
                            
                            detailed_df = pd.DataFrame(detailed_report)
                            detailed_csv = detailed_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download Detailed Report (CSV)",
                                data=detailed_csv,
                                file_name=f"detailed_safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        # Analytics insights
                        if PLOTLY_AVAILABLE:
                            st.subheader("📊 Safety Analytics")
                            
                            # Risk distribution chart
                            risk_counts = df_display['Risk_Level'].value_counts()
                            fig_risk = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Risk Level Distribution",
                                color_discrete_map={
                                    'CRITICAL': '#FF0000',
                                    'HIGH': '#FF8C00',
                                    'MEDIUM': '#FFD700',
                                    'LOW': '#90EE90',
                                    'MINIMAL': '#00FF00'
                                }
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)
                            
                            # Timeline analysis
                            fig_timeline = px.line(
                                df_display, 
                                x='Time (s)', 
                                y='Total Violations',
                                title="Safety Violations Over Time",
                                labels={'Total Violations': 'Number of Violations', 'Time (s)': 'Time (seconds)'}
                            )
                            fig_timeline.update_traces(line_color='red')
                            st.plotly_chart(fig_timeline, use_container_width=True)
                            
                            # Object detection timeline
                            fig_objects = go.Figure()
                            fig_objects.add_trace(go.Scatter(
                                x=df_display['Time (s)'],
                                y=df_display['Vehicles'],
                                mode='lines+markers',
                                name='Vehicles',
                                line=dict(color='blue')
                            ))
                            fig_objects.add_trace(go.Scatter(
                                x=df_display['Time (s)'],
                                y=df_display['Pedestrians'],
                                mode='lines+markers',
                                name='Pedestrians',
                                line=dict(color='red')
                            ))
                            fig_objects.update_layout(
                                title="Object Detection Over Time",
                                xaxis_title="Time (seconds)",
                                yaxis_title="Number of Objects"
                            )
                            st.plotly_chart(fig_objects, use_container_width=True)
                        
                        # Safety recommendations
                        st.subheader("💡 Safety Recommendations")
                        
                        recommendations = []
                        
                        if collision_frames > 0:
                            recommendations.append("🚨 **CRITICAL**: Immediate collision scenarios detected. Review traffic management and implement emergency protocols.")
                        
                        if warning_frames > 0:
                            recommendations.append("⚠️ **HIGH PRIORITY**: Multiple collision warnings detected. Consider speed reduction measures and enhanced signage.")
                        
                        if unsafe_frames > len(safety_results) * 0.3:
                            recommendations.append("⚠️ **MEDIUM PRIORITY**: High percentage of unsafe conditions. Review pedestrian crossing areas and vehicle spacing.")
                        
                        if sum([r['Pedestrians'] for r in safety_results]) > 0:
                            recommendations.append("🚶‍♂️ **PEDESTRIAN SAFETY**: Ensure 2-meter minimum distance from pedestrians. Consider dedicated pedestrian zones.")
                        
                        if sum([r['Vehicles'] for r in safety_results]) > len(safety_results) * 2:
                            recommendations.append("🚗 **TRAFFIC MANAGEMENT**: High vehicle density detected. Consider traffic flow optimization.")
                        
                        recommendations.append("📏 **2-SECOND RULE**: Maintain 2-second following distance between vehicles.")
                        recommendations.append("🎯 **CONTINUOUS MONITORING**: Regular safety monitoring recommended for optimal road safety.")
                        
                        for rec in recommendations:
                            st.write(rec)
                    
                    else:
                        st.error("No frames were processed. Please check your video file.")

# Additional utility functions
def get_safety_metrics(safety_results):
    """Calculate comprehensive safety metrics from results"""
    if not safety_results:
        return {}
    
    total_frames = len(safety_results)
    collision_frames = len([r for r in safety_results if r['Status'] == 'COLLISION DETECTED'])
    warning_frames = len([r for r in safety_results if r['Status'] == 'COLLISION WARNING'])
    unsafe_frames = len([r for r in safety_results if r['Status'] == 'Unsafe'])
    
    return {
        'safety_score': (total_frames - collision_frames * 3 - warning_frames * 2 - unsafe_frames) / total_frames * 100,
        'collision_rate': collision_frames / total_frames * 100,
        'warning_rate': warning_frames / total_frames * 100,
        'unsafe_rate': unsafe_frames / total_frames * 100
    }

if __name__ == "__main__":
    main()
