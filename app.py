import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import logging
import math
import json
import re
import zipfile
import requests
import tempfile
from datetime import datetime
import easyocr
from ultralytics import YOLO
import supervision as sv
from PIL import Image
import time
import random
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

from pathlib import Path

class ParkingSlotManager:
    def __init__(self, total_slots=100):
        self.total_slots = total_slots
        self.slots = {}
        self.slot_images = {}
        self.load_slot_data()
        
    def load_slot_data(self):
        """Load existing slot data or initialize new slots"""
        try:
            if os.path.exists('slot_data.json'):
                with open('slot_data.json', 'r') as f:
                    self.slots = json.load(f)
            else:
                self.initialize_slots()
        except Exception as e:
            logging.error(f"Error loading slot data: {str(e)}")
            self.initialize_slots()
            
    def save_slot_data(self):
        """Save slot data to file"""
        try:
            with open('slot_data.json', 'w') as f:
                json.dump(self.slots, f)
        except Exception as e:
            logging.error(f"Error saving slot data: {str(e)}")
            
    def initialize_slots(self):
        """Initialize parking slots with random occupancy"""
        for i in range(1, self.total_slots + 1):
            self.slots[str(i)] = {
                'occupied': random.random() < 0.2,
                'vehicle_number': None,
                'entry_time': None,
                'section': chr(65 + ((i-1) // 20)),  # Divide into sections A, B, C, D, E
                'floor': ((i-1) // 40) + 1  # Divide into floors
            }
        self.save_slot_data()
        
    def get_available_slot(self):
        """Get first available parking slot"""
        for slot_num, info in self.slots.items():
            if not info['occupied']:
                return slot_num
        return None
        
    def get_section_availability(self):
        """Get availability by section"""
        sections = {}
        for slot_num, info in self.slots.items():
            section = info['section']
            if section not in sections:
                sections[section] = {'total': 0, 'occupied': 0}
            sections[section]['total'] += 1
            if info['occupied']:
                sections[section]['occupied'] += 1
        return sections
        
    def get_floor_availability(self):
        """Get availability by floor"""
        floors = {}
        for slot_num, info in self.slots.items():
            floor = info['floor']
            if floor not in floors:
                floors[floor] = {'total': 0, 'occupied': 0}
            floors[floor]['total'] += 1
            if info['occupied']:
                floors[floor]['occupied'] += 1
        return floors
        
    def occupy_slot(self, slot_num, vehicle_number):
        """Occupy a specific slot"""
        if str(slot_num) in self.slots and not self.slots[str(slot_num)]['occupied']:
            self.slots[str(slot_num)].update({
                'occupied': True,
                'vehicle_number': vehicle_number,
                'entry_time': datetime.now().isoformat()
            })
            self.save_slot_data()
            return True
        return False
        
    def release_slot(self, slot_num):
        """Release an occupied slot"""
        if str(slot_num) in self.slots and self.slots[str(slot_num)]['occupied']:
            self.slots[str(slot_num)].update({
                'occupied': False,
                'vehicle_number': None,
                'entry_time': None
            })
            self.save_slot_data()
            return True
        return False
        
    def get_slot_by_vehicle(self, vehicle_number):
        """Get slot number for a vehicle"""
        for slot_num, info in self.slots.items():
            if info['vehicle_number'] == vehicle_number:
                return slot_num
        return None
        
    def get_parking_status(self):
        """Get current parking status"""
        total_occupied = sum(1 for info in self.slots.values() if info['occupied'])
        return {
            'total_slots': self.total_slots,
            'occupied_slots': total_occupied,
            'available_slots': self.total_slots - total_occupied,
            'occupancy_rate': (total_occupied / self.total_slots) * 100
        }
        
    def get_empty_slots(self):
        """Get list of empty slot numbers"""
        return [slot_num for slot_num, info in self.slots.items() if not info['occupied']]
        
    def get_slot_info(self, slot_num):
        """Get detailed information about a specific slot"""
        return self.slots.get(str(slot_num))

class AutomatedParkingDashboard:
    def __init__(self):
        """Initialize the automated parking system with core components"""
        # Core configuration
        self.config = {
            'VEHICLE_MODEL_PATH': 'models/yolov8n.pt',
            'DATA_DIR': 'data/PKLot',
            'LOG_DIR': 'logs',
            'RECEIPT_DIR': 'receipts',
            'TEMP_DIR': 'temp',
            'IMAGES_DIR': 'data/'
        }
        
        # Initialize system components
        self.setup_directories()
        self.setup_logging()
        
        # Initialize slot manager
        self.slot_manager = ParkingSlotManager(100)
        
        # Business logic initialization
        self.RATE_PER_HOUR = 50
        self.GST_RATE = 0.18
        
        # Additional rate configurations
        self.RATES = {
            'hourly': 50,
            'daily': 500,
            'weekly': 2500,
            'monthly': 8000
        }
        
        # Load core components
        self.initialize_models()
        self.initialize_database()
        self.initialize_ocr()
        self.initialize_dataset()
        
        # Initialize session state for real-time updates
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()

    def setup_directories(self):
        """Create necessary system directories"""
        for directory in self.config.values():
            if isinstance(directory, str) and not directory.endswith('.pt'):
                os.makedirs(directory, exist_ok=True)

    def setup_logging(self):
        """Configure system logging"""
        logging.basicConfig(
            filename=os.path.join(self.config['LOG_DIR'], 'parking_system.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def initialize_models(self):
        """Initialize vehicle detection model"""
        try:
            self.vehicle_model = YOLO('yolov8n.pt')
            logging.info("Vehicle detection model initialized successfully")
        except Exception as e:
            logging.error(f"Model initialization error: {str(e)}")
            st.error("Error initializing detection model")

    def initialize_database(self):
        """Initialize parking records database"""
        self.load_parking_records()

    def initialize_ocr(self):
        """Initialize OCR for license plate recognition"""
        try:
            self.reader = easyocr.Reader(['en'])
            logging.info("OCR initialized successfully")
        except Exception as e:
            logging.error(f"OCR initialization error: {str(e)}")
            st.error(e)

    def initialize_dataset(self):
        """Initialize and manage dataset"""
        try:
            dataset_path = self.config['DATA_DIR']
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            self.dataset_info = self.analyze_dataset(dataset_path)
        except Exception as e:
            logging.error(f"Dataset initialization error: {str(e)}")

    def analyze_dataset(self, dataset_path):
        """Analyze dataset structure and content"""
        info = {
            'path': dataset_path,
            'total_images': 0,
            'occupied_spots': 0,
            'empty_spots': 0,
            'weather_conditions': set(),
            'cameras': set()
        }
        
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    info['total_images'] += 1
                    if 'occupied' in file.lower():
                        info['occupied_spots'] += 1
                    else:
                        info['empty_spots'] += 1
                        
                    for weather in ['sunny', 'rainy', 'cloudy']:
                        if weather in root.lower():
                            info['weather_conditions'].add(weather)
                            
                    camera_match = re.search(r'camera\d+', root.lower())
                    if camera_match:
                        info['cameras'].add(camera_match.group())
        
        return info

    def load_parking_records(self):
        """Load or create parking records from CSV file"""
        try:
            if os.path.exists('parking_records.csv'):
                self.parking_records = pd.read_csv('parking_records.csv')
                self.parking_records['entry_time'] = pd.to_datetime(
                    self.parking_records['entry_time']
                )
                self.parking_records['exit_time'] = pd.to_datetime(
                    self.parking_records['exit_time']
                )
            else:
                self.parking_records = pd.DataFrame(columns=[
                    'vehicle_number', 'entry_time', 'exit_time', 
                    'duration', 'charges', 'status', 'slot_number',
                    'section', 'floor'
                ])
        except Exception as e:
            logging.error(f"Error loading parking records: {str(e)}")
            st.error("Error loading parking records")
            self.parking_records = pd.DataFrame(columns=[
                'vehicle_number', 'entry_time', 'exit_time', 
                'duration', 'charges', 'status', 'slot_number',
                'section', 'floor'
            ])

    def save_records(self):
        """Save parking records to CSV file"""
        try:
            self.parking_records.to_csv('parking_records.csv', index=False)
            logging.info("Parking records saved successfully")
        except Exception as e:
            logging.error(f"Error saving parking records: {str(e)}")

    def process_video_feed(self, video_source):
        """
        Process video feed with improved visualization and detection logic
        """
        try:
            # Initialize video capture
            if isinstance(video_source, str) and video_source.isdigit():
                cap = cv2.VideoCapture(int(video_source))
            elif hasattr(video_source, 'read'):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_source.read())
                tfile.flush()
                cap = cv2.VideoCapture(tfile.name)
            else:
                cap = cv2.VideoCapture(video_source)

            if not cap.isOpened():
                st.error("Failed to open video source")
                return

            # Create display elements
            frame_placeholder = st.empty()
            info_placeholder = st.empty()
            stop_button = st.button("Stop Processing")

            # Define visualization parameters
            BLUE = (255, 0, 0)  # BGR format
            GREEN = (0, 255, 0)
            RED = (0, 0, 255)
            FONT = cv2.FONT_HERSHEY_SIMPLEX

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Resize frame
                    height, width = frame.shape[:2]
                    target_width = 640
                    aspect_ratio = width / height
                    target_height = int(target_width / aspect_ratio)
                    frame = cv2.resize(frame, (target_width, target_height))
                    
                    # Create display frame
                    display_frame = frame.copy()
                    
                    # Detect vehicles using YOLO
                    results = self.vehicle_model(frame)[0]
                    
                    # Process detections
                    for result in results.boxes.data:
                        x1, y1, x2, y2, conf, class_id = result
                        
                        # Check if detection is a vehicle (class_id 2)
                        if int(class_id) == 2 and conf > 0.5:
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            
                            # Extract vehicle ROI
                            if y2 > y1 and x2 > x1:
                                vehicle_roi = frame[y1:y2, x1:x2]
                                
                                if vehicle_roi.size > 0:
                                    # Detect license plate
                                    plate_text = self.recognize_license_plate(vehicle_roi)
                                    
                                    if plate_text:
                                        # Draw detection visualization
                                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), GREEN, 2)
                                        
                                        # Add text background
                                        text = f"Plate: {plate_text}"
                                        (text_width, text_height), _ = cv2.getTextSize(
                                            text, FONT, 0.5, 1
                                        )
                                        cv2.rectangle(
                                            display_frame,
                                            (x1, y1 - text_height - 10),
                                            (x1 + text_width, y1),
                                            GREEN,
                                            -1
                                        )
                                        
                                        # Add text
                                        cv2.putText(
                                            display_frame,
                                            text,
                                            (x1, y1 - 5),
                                            FONT,
                                            0.5,
                                            BLUE,
                                            1,
                                            cv2.LINE_AA
                                        )
                                        
                                        # Process entry/exit
                                        self.handle_vehicle_entry_exit(plate_text)
                    
                    # Display frame
                    frame_placeholder.image(display_frame, channels="BGR")
                    
                    # Update status
                    status = self.get_current_status()
                    info_placeholder.markdown(f"""
                    ### Real-Time Parking Status
                    🚗 **Vehicles Currently Parked:** {status['vehicles_parked']}  
                    📊 **Total Vehicles Today:** {status['total_today']}  
                    💰 **Total Revenue:** ₹{status['revenue_today']:.2f}
                    """)
                    
                except Exception as e:
                    logging.error(f"Frame processing error: {str(e)}")
                    continue
                
                time.sleep(0.1)  # Control frame rate
            
            # Cleanup
            cap.release()
            if 'tfile' in locals():
                try:
                    os.unlink(tfile.name)
                except Exception as e:
                    logging.error(f"Error removing temporary file: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Video processing error: {str(e)}")
            st.error("Error in video processing")

    def annotate_frame(self, frame, results):
        """Annotate frame with detection results"""
        annotated_frame = frame.copy()
        
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, class_id = result
            
            if int(class_id) == 2 and conf > 0.5:  # Vehicle detection
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract vehicle ROI and process license plate
                vehicle_roi = frame[y1:y2, x1:x2]
                if vehicle_roi.size > 0:
                    plate_text = self.recognize_license_plate(vehicle_roi)
                    if plate_text:
                        cv2.putText(
                            annotated_frame,
                            f"Plate: {plate_text}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
        
        return annotated_frame

    def recognize_license_plate(self, image):
        """Recognize license plate from image"""
        try:
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image_gray = clahe.apply(image_gray)
            
            # OCR detection
            results = self.reader.readtext(image_gray)
            
            for _, text, conf in results:
                cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                if self.is_valid_license_format(cleaned_text) and conf > 0.4:
                    return cleaned_text
            
            return None
            
        except Exception as e:
            logging.error(f"License plate recognition error: {str(e)}")
            return None

    def is_valid_license_format(self, text):
        """Validate license plate format"""
        if not text or len(text) < 4:
            return False
            
        if not text[:2].isalpha():
            return False
            
        if not any(c.isdigit() for c in text[2:]):
            return False
            
        return True

    def handle_vehicle_entry_exit(self, plate_number):
        """Handle vehicle entry/exit"""
        try:
            current_time = datetime.now()
            
            # Initialize detection tracking
            if not hasattr(self, 'recent_detections'):
                self.recent_detections = {}
            
            # Check for recent detection
            if plate_number in self.recent_detections:
                last_detection = self.recent_detections[plate_number]
                time_diff = (current_time - last_detection['time']).total_seconds()
                
                if time_diff < 30:
                    self.recent_detections[plate_number]['time'] = current_time
                    return
            
            # Update detection tracking
            self.recent_detections[plate_number] = {
                'time': current_time,
                'count': self.recent_detections.get(plate_number, {}).get('count', 0) + 1
            }
            
            # Check vehicle status
            vehicle_record = self.parking_records[
                (self.parking_records['vehicle_number'] == plate_number) &
                (self.parking_records['status'] == 'parked')
            ]
            
            if vehicle_record.empty:
                if not self._check_recent_entry(plate_number):
                    self.record_entry(plate_number)
                    logging.info(f"New vehicle entry recorded: {plate_number}")
            else:
                entry_time = pd.to_datetime(vehicle_record.iloc[-1]['entry_time'])
                parking_duration = (current_time - entry_time).total_seconds()
                
                if parking_duration > 180:  # Minimum 3 minutes parking time
                    slot_number = vehicle_record.iloc[-1]['slot_number']
                    self.record_exit(plate_number)
                    self.generate_receipt(
                        plate_number,
                        entry_time,
                        current_time,
                        parking_duration / 3600,
                        math.ceil(parking_duration / 3600 * self.RATE_PER_HOUR),
                        slot_number
                    )
                    logging.info(f"Vehicle exit processed: {plate_number}")
                    
        except Exception as e:
            logging.error(f"Entry/exit handling error: {str(e)}")

    def _check_recent_entry(self, plate_number, threshold_minutes=3):
        """Check for recent entry to prevent duplicates"""
        try:
            current_time = datetime.now()
            recent_entries = self.parking_records[
                (self.parking_records['vehicle_number'] == plate_number) &
                (self.parking_records['status'] == 'parked')
            ]
            
            if not recent_entries.empty:
                last_entry = pd.to_datetime(recent_entries.iloc[-1]['entry_time'])
                time_diff = (current_time - last_entry).total_seconds() / 60
                return time_diff < threshold_minutes
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking recent entry: {str(e)}")
            return True

    def record_entry(self, vehicle_number):
        """Record vehicle entry with slot allocation"""

        if(vehicle_number.length<8):
            return
        try:
            entry_time = datetime.now()
            
            # Get available parking slot
            slot_number = self.slot_manager.get_available_slot()
            if slot_number is None:
                st.error("No parking slots available!")
                return
            
            # Get slot information
            slot_info = self.slot_manager.get_slot_info(slot_number)
            
            # Occupy the slot
            self.slot_manager.occupy_slot(slot_number, vehicle_number)
            
            new_record = pd.DataFrame([{
                'vehicle_number': vehicle_number,
                'entry_time': entry_time,
                'exit_time': None,
                'duration': None,
                'charges': None,
                'status': 'parked',
                'slot_number': slot_number,
                'section': slot_info['section'],
                'floor': slot_info['floor']
            }])
            
            self.parking_records = pd.concat([self.parking_records, new_record], 
                                           ignore_index=True)
            self.save_records()
            
            # Add entry notification
            st.success(
                f"Entry recorded: {vehicle_number}\n"
                f"Allocated Slot: {slot_number} (Section {slot_info['section']}, Floor {slot_info['floor']})"
            )
            
        except Exception as e:
            logging.error(f"Entry recording error: {str(e)}")

    def record_exit(self, vehicle_number):
        """Record vehicle exit with slot release"""
        try:
            exit_time = datetime.now()
            idx = self.parking_records[
                (self.parking_records['vehicle_number'] == vehicle_number) & 
                (self.parking_records['status'] == 'parked')
            ].index[0]
            
            slot_number = self.parking_records.loc[idx, 'slot_number']
            slot_info = self.slot_manager.get_slot_info(slot_number)
            
            entry_time = pd.to_datetime(self.parking_records.loc[idx, 'entry_time'])
            duration = (exit_time - entry_time).total_seconds() / 3600
            
            # Calculate charges based on duration
            charges = self.calculate_charges(duration)
            
            # Release the parking slot
            self.slot_manager.release_slot(slot_number)
            
            # Update record
            self.parking_records.loc[idx, 'exit_time'] = exit_time
            self.parking_records.loc[idx, 'duration'] = duration
            self.parking_records.loc[idx, 'charges'] = charges
            self.parking_records.loc[idx, 'status'] = 'exited'
            
            self.save_records()
            
            # Add exit notification
            duration_minutes = duration * 60
            st.success(
                f"Exit recorded: {vehicle_number}\n"
                f"Parking duration: {int(duration_minutes)} minutes\n"
                f"Released Slot: {slot_number} (Section {slot_info['section']}, Floor {slot_info['floor']})"
            )
            
        except Exception as e:
            logging.error(f"Exit recording error: {str(e)}")

    def calculate_charges(self, duration):
        """Calculate parking charges based on duration"""
        if duration <= 24:  # Less than a day
            return math.ceil(duration * self.RATES['hourly'])
        elif duration <= 168:  # Less than a week
            days = math.ceil(duration / 24)
            return days * self.RATES['daily']
        elif duration <= 720:  # Less than a month
            weeks = math.ceil(duration / 168)
            return weeks * self.RATES['weekly']
        else:
            months = math.ceil(duration / 720)
            return months * self.RATES['monthly']

    def generate_receipt(self, vehicle_number, entry_time, exit_time, duration, charges, slot_number):
        """Generate detailed parking receipt with slot information"""
        try:
            slot_info = self.slot_manager.get_slot_info(slot_number)
            gst = charges * self.GST_RATE
            total = charges + gst
            
            receipt_html = f"""
            <div style="font-family: Arial; padding: 20px; border: 2px solid #333;">
                <h2 style="text-align: center;">PARKING RECEIPT</h2>
                <div style="border-bottom: 1px solid #ccc; margin-bottom: 15px;"></div>
                
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p><strong>Vehicle Number:</strong> {vehicle_number}</p>
                        <p><strong>Entry Time:</strong> {entry_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Exit Time:</strong> {exit_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Duration:</strong> {duration:.2f} hours</p>
                    </div>
                    <div>
                        <p><strong>Slot Number:</strong> {slot_number}</p>
                        <p><strong>Section:</strong> {slot_info['section']}</p>
                        <p><strong>Floor:</strong> {slot_info['floor']}</p>
                    </div>
                </div>
                
                <div style="border-top: 1px solid #ccc; margin-top: 15px; padding-top: 15px;">
                    <p><strong>Base Charge:</strong> ₹{charges:.2f}</p>
                    <p><strong>GST (18%):</strong> ₹{gst:.2f}</p>
                    <h3 style="text-align: center; color: #1a73e8;">Total Amount: ₹{total:.2f}</h3>
                </div>
                
                <div style="text-align: center; margin-top: 20px; font-size: 0.8em; color: #666;">
                    Thank you for using our parking service!
                </div>
            </div>
            """
            
            st.markdown(receipt_html, unsafe_allow_html=True)
            
            # Save receipt
            receipt_path = os.path.join(
                self.config['RECEIPT_DIR'],
                f"receipt_{vehicle_number}_{exit_time.strftime('%Y%m%d%H%M%S')}.html"
            )
            with open(receipt_path, 'w') as f:
                f.write(receipt_html)
                
        except Exception as e:
            logging.error(f"Receipt generation error: {str(e)}")

    def show_parking_slots(self):
        """Display parking slots status and visualization"""
        st.header("Parking Slots Status")
        
        try:
            # Get current parking status
            status = self.slot_manager.get_parking_status()
            
            # Display overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Slots", status['total_slots'])
            with col2:
                st.metric("Occupied Slots", status['occupied_slots'])
            with col3:
                st.metric("Available Slots", status['available_slots'])
            with col4:
                st.metric("Occupancy Rate", f"{status['occupancy_rate']:.1f}%")
            
            # Section-wise availability
            st.subheader("Section-wise Availability")
            section_data = self.slot_manager.get_section_availability()
            section_cols = st.columns(len(section_data))
            for i, (section, data) in enumerate(section_data.items()):
                with section_cols[i]:
                    available = data['total'] - data['occupied']
                    st.metric(
                        f"Section {section}",
                        f"{available}/{data['total']} free",
                        f"{(available/data['total']*100):.1f}%"
                    )
            
            # Floor-wise availability
            st.subheader("Floor-wise Availability")
            floor_data = self.slot_manager.get_floor_availability()
            floor_cols = st.columns(len(floor_data))
            for i, (floor, data) in enumerate(floor_data.items()):
                with floor_cols[i]:
                    available = data['total'] - data['occupied']
                    st.metric(
                        f"Floor {floor}",
                        f"{available}/{data['total']} free",
                        f"{(available/data['total']*100):.1f}%"
                    )
            
            # Available slots visualization
            st.subheader("Available Parking Slots")
            empty_slots = self.slot_manager.get_empty_slots()
            
            if empty_slots:
                # Create an interactive grid layout
                slots_per_row = 10
                for i in range(0, len(empty_slots), slots_per_row):
                    cols = st.columns(slots_per_row)
                    for j, slot_num in enumerate(empty_slots[i:i+slots_per_row]):
                        slot_info = self.slot_manager.get_slot_info(slot_num)
                        cols[j].button(
                            f"Slot {slot_num}\nSection {slot_info['section']}\nFloor {slot_info['floor']}",
                            key=f"slot_{slot_num}"
                        )
            else:
                st.warning("No parking slots available!")
            
            # Display parking lot layout
            st.subheader("Parking Lot Layout")
            layout_image = os.path.join(self.config['IMAGES_DIR'], "image.jpeg")
            if os.path.exists(layout_image):
                image = Image.open(layout_image)
                st.image(image, caption="Parking Lot Layout")
            else:
                st.info("Parking lot layout image not available")
            
        except Exception as e:
            logging.error(f"Error displaying parking slots: {str(e)}")
            st.error("Error loading parking slots information")

    def show_parking_records(self):
        """Display comprehensive parking records interface"""
        st.subheader("Parking Records")
        
        try:
            # Current statistics
            current = self.parking_records[self.parking_records['status'] == 'parked']
            today_date = pd.to_datetime('today').normalize()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Currently Parked", len(current))
            with col2:
                total_today = len(self.parking_records[
                    pd.to_datetime(self.parking_records['entry_time']).dt.normalize() == today_date
                ])
                st.metric("Total Today", total_today)
            with col3:
                total_revenue = self.parking_records[
                    self.parking_records['status'] == 'exited'
                ]['charges'].sum()
                st.metric("Total Revenue", f"₹{total_revenue:.2f}")
            with col4:
                if total_today > 0:
                    occupancy = (len(current) / total_today) * 100
                    st.metric("Occupancy Rate", f"{occupancy:.1f}%")
            
            # Currently parked vehicles
            st.subheader("Currently Parked Vehicles")
            if not current.empty:
                current['current_duration'] = (datetime.now() - 
                    pd.to_datetime(current['entry_time'])).dt.total_seconds() / 3600
                display_current = current[[
                    'vehicle_number', 'entry_time', 'current_duration',
                    'slot_number', 'section', 'floor'
                ]]
                st.dataframe(display_current.style.format({
                    'current_duration': '{:.2f} hours'
                }))
            else:
                st.info("No vehicles currently parked")
            
            # Historical records with filtering
            st.subheader("Historical Records")
            col1, col2, col3 = st.columns(3)
            with col1:
                date_filter = st.date_input("Select Date", datetime.now().date())
            with col2:
                status_filter = st.selectbox("Status", ["All", "Parked", "Exited"])
            with col3:
                section_filter = st.selectbox(
                    "Section", ["All"] + sorted(list(set(self.parking_records['section'].dropna())))
                )
            
            if date_filter:
                filter_date = pd.to_datetime(date_filter)
                historical = self.parking_records[
                    pd.to_datetime(self.parking_records['entry_time']).dt.normalize() == filter_date
                ]
                
                # Apply status filter
                if status_filter != "All":
                    historical = historical[
                        historical['status'].str.lower() == status_filter.lower()
                    ]
                
                # Apply section filter
                if section_filter != "All":
                    historical = historical[historical['section'] == section_filter]
                
                if not historical.empty:
                    st.dataframe(
                        historical.style.format({
                            'duration': '{:.2f}',
                            'charges': '₹{:.2f}'
                        })
                    )
                    
                    # Daily statistics
                    st.subheader("Daily Statistics")
                    completed_visits = historical[historical['status'] == 'exited']
                    daily_stats = {
                        'Total Vehicles': len(historical),
                        'Total Revenue': f"₹{completed_visits['charges'].sum():.2f}",
                        'Average Duration': f"{completed_visits['duration'].mean():.2f} hours",
                        'Average Charge': f"₹{completed_visits['charges'].mean():.2f}",
                        'Most Active Section': historical['section'].mode().iloc[0] if not historical['section'].empty else 'N/A',
                        'Most Used Floor': str(historical['floor'].mode().iloc[0]) if not historical['floor'].empty else 'N/A'
                    }
                    st.json(daily_stats)
                    
                    # Visualizations
                    st.subheader("Daily Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Hourly distribution
                        hourly_data = historical['entry_time'].dt.hour.value_counts().sort_index()
                        st.bar_chart(hourly_data)
                        st.caption("Hourly Distribution of Vehicle Entries")
                    
                    with col2:
                        # Section-wise distribution
                        section_data = historical['section'].value_counts()
                        st.bar_chart(section_data)
                        st.caption("Section-wise Vehicle Distribution")
                    
                else:
                    st.info(f"No records found for {date_filter}")
                    
        except Exception as e:
            logging.error(f"Error displaying parking records: {str(e)}")
            st.error("Error loading parking records")

    def show_system_status(self):
        """Display system status and analytics dashboard"""
        st.header("System Status and Analytics")
        
        try:
            # System metrics
            status = self.get_current_status()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Active Vehicles", status['vehicles_parked'])
            with col2:
                st.metric("Total Today", status['total_today'])
            with col3:
                st.metric("Revenue Today", f"₹{status['revenue_today']:.2f}")
            with col4:
                uptime = (datetime.now() - st.session_state.last_update).total_seconds() / 3600
                st.metric("System Uptime", f"{uptime:.1f} hours")
            
            # System components status
            st.subheader("Component Status")
            components = {
                "Vehicle Detection Model": self.vehicle_model is not None,
                "OCR System": hasattr(self, 'reader'),
                "Database": hasattr(self, 'parking_records'),
                "Slot Management": hasattr(self, 'slot_manager'),
                "Dataset": hasattr(self, 'dataset_info')
            }
            
            # Display component status
            status_cols = st.columns(len(components))
            for i, (component, status) in enumerate(components.items()):
                with status_cols[i]:
                    st.markdown(
                        f"### {component}\n"
                        f"{'✅ Active' if status else '❌ Inactive'}"
                    )
            
            # System performance metrics
            st.subheader("System Performance")
            
            # Load recent logs
            log_file = os.path.join(self.config['LOG_DIR'], 'parking_system.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    recent_logs = f.readlines()[-10:]
                
                # Display logs in expandable section
                with st.expander("Recent System Logs"):
                    for log in recent_logs:
                        st.text(log.strip())
            
            # Dataset information
            if hasattr(self, 'dataset_info'):
                st.subheader("Dataset Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Image Statistics")
                    st.json({
                        "Total Images": self.dataset_info['total_images'],
                        "Occupied Spots": self.dataset_info['occupied_spots'],
                        "Empty Spots": self.dataset_info['empty_spots']
                    })
                
                with col2:
                    st.markdown("### Environmental Conditions")
                    st.json({
                        "Weather Conditions": sorted(list(self.dataset_info['weather_conditions'])),
                        "Active Cameras": sorted(list(self.dataset_info['cameras']))
                    })
            
        except Exception as e:
            logging.error(f"Error displaying system status: {str(e)}")
            st.error("Error loading system status")

    def get_current_status(self):
        """Get real-time parking status and statistics"""
        try:
            current = self.parking_records[self.parking_records['status'] == 'parked']
            today_date = pd.to_datetime('today').normalize()
            today_records = self.parking_records[
                pd.to_datetime(self.parking_records['entry_time']).dt.normalize() == today_date
            ]
            
            revenue = today_records[
                today_records['status'] == 'exited'
            ]['charges'].sum()
            
            return {
                'vehicles_parked': len(current),
                'total_today': len(today_records),
                'revenue_today': revenue
            }
        except Exception as e:
            logging.error(f"Status calculation error: {str(e)}")
            return {'vehicles_parked': 0, 'total_today': 0, 'revenue_today': 0}

    def run(self):
        """Main dashboard entry point"""
        st.title("Automated Parking Management System")
        
        # Sidebar configuration
        st.sidebar.title("Navigation")
        menu = st.sidebar.selectbox(
            "Select Option",
            ["Live Detection", "Parking Records", "Parking Slots", "System Status"]
        )
        
        # Display settings in sidebar
        with st.sidebar.expander("Settings"):
            self.RATE_PER_HOUR = st.number_input(
                "Hourly Rate (₹)", 
                min_value=10, 
                value=self.RATE_PER_HOUR
            )
            self.GST_RATE = st.number_input(
                "GST Rate (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=self.GST_RATE * 100
            ) / 100
        
        # Main content area
        if menu == "Live Detection":
            st.header("Live Vehicle Detection")
            
            source = st.radio("Select Source", ["Camera", "Upload Video"])
            
            if source == "Camera":
                camera_id = st.selectbox(
                    "Select Camera", 
                    ["0", "1", "2"],
                    help="Choose camera device ID (0 is usually the built-in camera)"
                )
                if st.button("Start Camera", key="start_camera"):
                    self.process_video_feed(camera_id)
            else:
                video_file = st.file_uploader(
                    "Upload Video",
                    type=['mp4', 'avi', 'mov'],
                    help="Upload a video file for vehicle detection"
                )
                if video_file:
                    self.process_video_feed(video_file)
                    
        elif menu == "Parking Records":
            self.show_parking_records()
        elif menu == "Parking Slots":
            self.show_parking_slots()
        else:
            self.show_system_status()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Automated Parking System",
        page_icon="🅿️",
        layout="wide"
    )
    
    # Initialize and run the dashboard
    dashboard = AutomatedParkingDashboard()
    dashboard.run()
