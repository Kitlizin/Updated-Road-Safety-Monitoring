import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from PIL import Image
from collections import defaultdict, deque
import time
import math

# Try to import optional dependencies
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

# Page configuration
st.set_page_config(
    page_title="Reckless Driving Safety Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SafetyAnalyzer:
    def __init__(self, fps=30):
        self.prev_positions = {}
        self.fps = fps
        self.TAILGATING_TIME = 2  # 2-second rule
        self.PEDESTRIAN_DISTANCE = 1.0  # 1 meter
        self.SPEED_ESTIMATION_FRAMES = 5
        self.vehicle_history = defaultdict(lambda: deque(maxlen=self.SPEED_ESTIMATION_FRAMES))
        self.pedestrian_history = defaultdict(lambda: deque(maxlen=self.SPEED_ESTIMATION_FRAMES))
        
    def analyze_frame(self, frame, detections):
        """Analyze frame for safety violations"""
        safety_status = "Safe"
        violations = []
        
        vehicles = []
        pedestrians = []
        
        # Process detections
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']
            track_id = detection.get('track_id', 0)
            
            if class_id == 0:  # Vehicle
                vehicles.append({
                    'id': track_id,
                    'bbox': bbox,
                    'confidence': confidence
                })
            elif class_id == 1:  # Pedestrian
                pedestrians.append({
                    'id': track_id,
                    'bbox': bbox,
                    'confidence': confidence
                })
        
        # Check pedestrian-vehicle distances
        for ped in pedestrians:
            for veh in vehicles:
                distance = self.calculate_pixel_distance(ped['bbox'], veh['bbox'])
                # Convert pixel distance to approximate real distance (simplified)
                real_distance = distance * 0.01  # Rough conversion factor
                
                if real_distance < self.PEDESTRIAN_DISTANCE:
                    safety_status = "Unsafe"
                    violations.append(f"Vehicle too close to pedestrian ({real_distance:.2f}m)")
        
        # Check tailgating between vehicles
        for i, veh1 in enumerate(vehicles):
            for j, veh2 in enumerate(vehicles[i+1:], i+1):
                distance = self.calculate_pixel_distance(veh1['bbox'], veh2['bbox'])
                real_distance = distance * 0.01  # Rough conversion factor
                
                # Approximate time gap based on distance and assumed speed
                assumed_speed = 13.89  # 50 km/h in m/s
                time_gap = real_distance / assumed_speed if assumed_speed > 0 else 0
                
                if time_gap < self.TAILGATING_TIME and real_distance < 15:  # 15m threshold
                    safety_status = "Unsafe"
                    violations.append(f"Tailgating detected ({time_gap:.2f}s gap, {real_distance:.2f}m)")
        
        return safety_status, violations
    
    def calculate_pixel_distance(self, bbox1, bbox2):
        """Calculate pixel distance between two bounding boxes"""
        x1, y1, x2, y2 = bbox1
        center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        x1, y1, x2, y2 = bbox2
        center2 = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def load_yolo_model(model_path="FinalModel_yolov8.pt"):
    """Load the trained YOLOv8 model"""
    if not YOLO_AVAILABLE:
        st.info("🔄 YOLO not available - Running in demo mode")
        return "demo_mode"
        
    try:
        # For demo purposes, use a pretrained model
        model = YOLO('yolov8n.pt')  # This will download automatically
        st.info("✅ Using YOLOv8n pretrained model")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def run_detection(model, image, conf_threshold=0.5, iou_threshold=0.45):
    """Run YOLOv8 detection on image"""
    if not YOLO_AVAILABLE or model == "demo_mode" or model is None:
        # Return dummy detections for demo
        height, width = image.shape[:2] if len(image.shape) > 2 else (400, 600)
        return [
            {'bbox': [width*0.1, height*0.3, width*0.4, height*0.7], 'confidence': 0.85, 'class_id': 0, 'track_id': 1},  # vehicle
            {'bbox': [width*0.6, height*0.4, width*0.75, height*0.8], 'confidence': 0.75, 'class_id': 1, 'track_id': 2}   # pedestrian
        ]
    
    try:
        # Convert PIL image to numpy array if needed
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
                    
                    # Map COCO classes to our classes (person=0->1, car/truck/bus=2,5,7->0)
                    if cls == 0:  # person
                        class_id = 1  # pedestrian
                    elif cls in [2, 5, 7]:  # car, bus, truck
                        class_id = 0  # vehicle
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
    """Draw bounding boxes and safety annotations on image"""
    if not CV2_AVAILABLE:
        return image
        
    import cv2
    
    if isinstance(image, Image.Image):
        img_copy = np.array(image)
    else:
        img_copy = image.copy()
    
    # Color coding (BGR format for OpenCV)
    colors = {
        'Vehicle': (0, 255, 0),  # Green
        'Pedestrian': (255, 0, 0),  # Blue
        'Unsafe': (0, 0, 255)  # Red
    }
    
    for detection in detections:
        bbox = detection['bbox']
        class_name = 'Vehicle' if detection['class_id'] == 0 else 'Pedestrian'
        confidence = detection['confidence']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color based on safety status
        color = colors['Unsafe'] if safety_status == "Unsafe" else colors[class_name]
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw safety status
    status_color = (0, 0, 255) if safety_status == "Unsafe" else (0, 255, 0)
    cv2.putText(img_copy, f"Status: {safety_status}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Draw violations
    for i, violation in enumerate(violations):
        cv2.putText(img_copy, violation[:50], (10, 70 + i * 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return img_copy

def main():
    st.title("🚗 Reckless Driving Safety Monitor")
    st.markdown("**Research Title:** Reckless Driving Behavior Recognition For Road Safety Monitoring")
    
    # Show system status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("OpenCV", "✅ Available" if CV2_AVAILABLE else "❌ Not Available")
    with col2:
        st.metric("YOLO", "✅ Available" if YOLO_AVAILABLE else "⚠️ Demo Mode")
    with col3:
        st.metric("Plotly", "✅ Available" if PLOTLY_AVAILABLE else "❌ Basic Charts")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Real-time Detection",
        "Batch Analysis", 
        "Safety Analytics"
    ])
    
    if page == "Real-time Detection":
        real_time_detection_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Safety Analytics":
        safety_analytics_page()

def real_time_detection_page():
    st.header("🎥 Real-time Detection")
    
    # Model loading section
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        model_path = st.text_input("Model Path", "FinalModel_yolov8.pt")
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                st.session_state.model = load_yolo_model(model_path)
                if st.session_state.model is not None:
                    st.success("✅ Model loaded successfully!")
    
    with col2:
        st.subheader("Detection Parameters")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45)
    
    # Initialize model if not exists
    if 'model' not in st.session_state:
        st.session_state.model = load_yolo_model(model_path)
    
    # Safety information
    st.subheader("🛡️ Safety Rules")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**2-Second Rule**: Maintain 2-3 second following distance (10-15m at 60-80 km/h)")
    with col2:
        st.info("**Pedestrian Safety**: Maintain minimum 1 meter distance when passing pedestrians")
    
    # Upload options
    st.subheader("Input Source")
    input_type = st.radio("Choose input type:", ["Upload Image", "Upload Video"])
    
    safety_analyzer = SafetyAnalyzer()
    
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                with st.spinner("Running detection..."):
                    # Run detection
                    detections = run_detection(st.session_state.model, img_array, 
                                             confidence_threshold, iou_threshold)
                    
                    safety_status, violations = safety_analyzer.analyze_frame(img_array, detections)
                    
                    # Draw annotations
                    if CV2_AVAILABLE:
                        annotated_img = draw_safety_annotations(img_array, detections, safety_status, violations)
                        st.image(annotated_img, use_column_width=True)
                    else:
                        st.image(img_array, use_column_width=True)
                        st.warning("OpenCV not available - showing original image")
                
                # Safety report
                st.subheader("Safety Analysis")
                if safety_status == "Safe":
                    st.success("✅ Safe - No violations detected")
                else:
                    st.error("⚠️ Unsafe - Violations detected:")
                    for violation in violations:
                        st.write(f"• {violation}")
                
                # Detection details
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
            
            if st.button("Process Video (Sample Frames)"):
                if not CV2_AVAILABLE:
                    st.error("OpenCV is required for video processing.")
                    return
                    
                # Save video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name
                
                import cv2
                with st.spinner("Processing video frames..."):
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    st.info(f"📹 Video: {frame_count} frames, {fps:.1f} FPS")
                    
                    # Process sample frames
                    progress_bar = st.progress(0)
                    sample_frames = min(5, frame_count)
                    safety_results = []
                    
                    for i in range(sample_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        detections = run_detection(st.session_state.model, frame, 
                                                 confidence_threshold, iou_threshold)
                        safety_status, violations = safety_analyzer.analyze_frame(frame, detections)
                        
                        safety_results.append({
                            'Frame': i+1,
                            'Status': safety_status,
                            'Violations': len(violations),
                            'Objects': len(detections)
                        })
                        
                        progress_bar.progress((i+1)/sample_frames)
                    
                    cap.release()
                    
                    # Display results
                    if safety_results:
                        st.subheader("📊 Sample Analysis Results")
                        df_results = pd.DataFrame(safety_results)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Summary
                        safe_frames = len([r for r in safety_results if r['Status'] == 'Safe'])
                        unsafe_frames = len(safety_results) - safe_frames
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Safe Frames", safe_frames)
                        with col2:
                            st.metric("Unsafe Frames", unsafe_frames)
                        with col3:
                            safety_pct = (safe_frames / len(safety_results)) * 100
                            st.metric("Safety %", f"{safety_pct:.1f}%")

def batch_analysis_page():
    st.header("📊 Batch Analysis")
    
    # Check model
    if 'model' not in st.session_state:
        st.session_state.model = load_yolo_model("FinalModel_yolov8.pt")
    
    st.subheader("Upload Multiple Files")
    uploaded_files = st.file_uploader("Choose multiple images", type=['jpg', 'jpeg', 'png'], 
                                     accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.5, key="batch_conf")
    with col2:
        max_files = st.number_input("Max files to process", 1, 50, 10)
    
    if uploaded_files and st.button("🚀 Analyze Batch"):
        safety_analyzer = SafetyAnalyzer()
        results = []
        
        files_to_process = uploaded_files[:max_files]
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(files_to_process):
            try:
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                detections = run_detection(st.session_state.model, img_array, confidence_threshold)
                safety_status, violations = safety_analyzer.analyze_frame(img_array, detections)
                
                results.append({
                    'Filename': uploaded_file.name,
                    'Status': safety_status,
                    'Objects': len(detections),
                    'Vehicles': len([d for d in detections if d['class_id'] == 0]),
                    'Pedestrians': len([d for d in detections if d['class_id'] == 1]),
                    'Violations': len(violations)
                })
                
                progress_bar.progress((i + 1) / len(files_to_process))
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if results:
            st.success(f"✅ Processed {len(results)} files!")
            
            # Summary metrics
            total_files = len(results)
            safe_files = len([r for r in results if r['Status'] == 'Safe'])
            unsafe_files = total_files - safe_files
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", total_files)
            with col2:
                st.metric("Safe", safe_files)
            with col3:
                st.metric("Unsafe", unsafe_files)
            with col4:
                safety_pct = (safe_files / total_files) * 100
                st.metric("Safety %", f"{safety_pct:.1f}%")
            
            # Results table
            st.subheader("📋 Detailed Results")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            
            # Simple charts without Plotly
            if PLOTLY_AVAILABLE:
                import plotly.express as px
                col1, col2 = st.columns(2)
                
                with col1:
                    status_counts = df_results['Status'].value_counts()
                    fig = px.pie(values=status_counts.values, names=status_counts.index,
                               title="Safety Status Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(df_results, x='Filename', y=['Vehicles', 'Pedestrians'],
                               title="Objects per File")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Basic charts using Streamlit
                st.subheader("📈 Status Distribution")
                status_counts = df_results['Status'].value_counts()
                st.bar_chart(status_counts)
            
            # Download option
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="batch_analysis_results.csv",
                mime="text/csv"
            )

def safety_analytics_page():
    st.header("📈 Safety Analytics Dashboard")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    safety_data = pd.DataFrame({
        'Date': dates,
        'Total_Detections': np.random.randint(50, 200, 30),
        'Safe_Incidents': np.random.randint(40, 180, 30),
        'Unsafe_Incidents': np.random.randint(5, 30, 30),
        'Tailgating': np.random.randint(0, 15, 30),
        'Pedestrian_Violations': np.random.randint(0, 10, 30)
    })
    
    safety_data['Safety_Percentage'] = (safety_data['Safe_Incidents'] / safety_data['Total_Detections']) * 100
    
    # Key metrics
    st.subheader("🎯 Key Safety Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_safety = safety_data['Safety_Percentage'].mean()
        st.metric("Average Safety %", f"{avg_safety:.1f}%")
    
    with col2:
        total_violations = safety_data['Unsafe_Incidents'].sum()
        st.metric("Total Violations", total_violations)
    
    with col3:
        best_day = safety_data['Safety_Percentage'].max()
        st.metric("Best Day %", f"{best_day:.1f}%")
    
    with col4:
        worst_day = safety_data['Safety_Percentage'].min()
        st.metric("Worst Day %", f"{worst_day:.1f}%")
    
    # Charts
    if PLOTLY_AVAILABLE:
        import plotly.graph_objects as go
        import plotly.express as px
        
        st.subheader("📊 Safety Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=safety_data['Date'], y=safety_data['Safety_Percentage'],
                                mode='lines+markers', name='Safety %'))
        fig.update_layout(title='Safety Percentage Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            total_tailgating = safety_data['Tailgating'].sum()
            total_pedestrian = safety_data['Pedestrian_Violations'].sum()
            fig = px.pie(values=[total_tailgating, total_pedestrian], 
                        names=['Tailgating', 'Pedestrian Violations'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            recent_data = safety_data.tail(7)
            fig = px.bar(recent_data, x='Date', y=['Safe_Incidents', 'Unsafe_Incidents'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Basic Streamlit charts
        st.subheader("📊 Safety Trends (Basic)")
        st.line_chart(safety_data.set_index('Date')['Safety_Percentage'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Violation Types")
            violation_data = {
                'Tailgating': safety_data['Tailgating'].sum(),
                'Pedestrian': safety_data['Pedestrian_Violations'].sum()
            }
            st.bar_chart(violation_data)
        
        with col2:
            st.subheader("Recent Activity")
            recent = safety_data.tail(7).set_index('Date')[['Safe_Incidents', 'Unsafe_Incidents']]
            st.bar_chart(recent)
    
    # Data table
    st.subheader("📋 Raw Data")
    st.dataframe(safety_data.sort_values('Date', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
