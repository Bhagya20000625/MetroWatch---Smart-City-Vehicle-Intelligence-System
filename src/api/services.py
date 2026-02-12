"""
Business Logic Layer - Vehicle Analytics Service
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.detection.vehicle_detector import VehicleDetector
from src.ocr.plate_recognizer import LicensePlateRecognizer
from src.utils.province_detector import ProvinceDetector
from src.database.database import SessionLocal
from src.database.models import Vehicle, VehicleLog
from sqlalchemy import func
import cv2
import numpy as np
from datetime import datetime
import tempfile
import os
from src.tracking.vehicle_tracker import VehicleTracker


class VehicleAnalyticsService:
    def __init__(self):
        print("Initializing Vehicle Analytics Service...")
        self.detector = VehicleDetector(model_name='yolov8n.pt')
        self.plate_recognizer = LicensePlateRecognizer()
        self.province_detector = ProvinceDetector()
        print("✓ Service initialized successfully!")
        
    def process_frame(self, image_bytes, db_session=None):
        """
        Process a single frame and store results in database.
        
        Args:
            image_bytes: Image as bytes
            db_session: SQLAlchemy session (optional, creates new if not provided)
        """
        # Convert bytes to cv2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image format")
        
        # Detect vehicles
        vehicles, _ = self.detector.detect_vehicles(img, confidence_threshold=0.3)
        
        # Use provided session or create new one
        close_session = False
        if db_session is None:
            db_session = SessionLocal()
            close_session = True
        
        try:
            results = []
            for vehicle in vehicles:
                # Extract vehicle region
                x1, y1, x2, y2 = vehicle['bbox']
                vehicle_img = img[y1:y2, x1:x2]
                
                # Recognize plate
                plate_info = None
                province = None
                
                if vehicle_img.size > 0:
                    plate_info = self.plate_recognizer.recognize_plate(vehicle_img)
                    
                    # Get province if plate detected
                    if plate_info and plate_info.get('plate_text'):
                        province_info = self.province_detector.detect_province(plate_info['plate_text'])
                        # Extract province name string from dict
                        province = province_info.get('province_name') if province_info else None
                
                # Create vehicle record in database
                db_vehicle = Vehicle(
                    vehicle_type=vehicle['type'],
                    confidence=vehicle['confidence'],
                    bbox={
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2)
                    },
                    plate_text=plate_info['plate_text'] if plate_info else None,
                    province=province,
                    timestamp=datetime.utcnow()
                )
                
                db_session.add(db_vehicle)
                db_session.commit()
                db_session.refresh(db_vehicle)
                
                # Prepare result
                vehicle_data = {
                    'id': db_vehicle.id,
                    'type': db_vehicle.vehicle_type,
                    'confidence': db_vehicle.confidence,
                    'bbox': db_vehicle.bbox,
                    'plate_text': db_vehicle.plate_text,
                    'province': db_vehicle.province,
                    'timestamp': db_vehicle.timestamp.isoformat()
                }
                
                results.append(vehicle_data)
            
            return results
            
        finally:
            if close_session:
                db_session.close()

    def process_video(self, video_bytes, db_session=None):
        """
        Process video with bidirectional traffic monitoring.
        
        Uses line-crossing detection to count:
        - Entry: Vehicles crossing entry line (entering monitored area)
        - Exit: Vehicles crossing exit line (leaving monitored area)
        
        Entry and exit counts represent DIFFERENT vehicles in bidirectional traffic flow.
        """
        # Save video to temp file (OpenCV needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            video_path = tmp_file.name
    
        try:
        # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Unable to open video file")
        
        # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
            print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize tracker
            tracker = VehicleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Set entry/exit zones (25% and 75% of frame height)
            entry_y = int(height * 0.25)
            exit_y = int(height * 0.75)
            tracker.set_entry_line(0, entry_y, width, entry_y)
            tracker.set_exit_line(0, exit_y, width, exit_y)
        
        # Use provided session or create new one
            close_session = False
            if db_session is None:
                db_session = SessionLocal()
                close_session = True
        
            frame_count = 0
            tracked_vehicles = {}  # Store vehicle DB records by track_id
            vehicle_types = {}     # Store vehicle types by track_id
        
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
            
                frame_count += 1
            
            # Process every 2nd frame for speed
                if frame_count % 2 != 0:
                    continue
                
            # Detect vehicles
                detected_vehicles, _ = self.detector.detect_vehicles(frame, confidence_threshold=0.3)
            
            # Prepare detections for tracker (format: [[x1,y1,x2,y2,score], ...])
                detections = []
                detection_types = {}  # Map bbox to vehicle type

                for v in detected_vehicles:
                    x1, y1, x2, y2 = v['bbox']
                    detections.append([x1, y1, x2, y2, v['confidence']])
                    # Store type keyed by bbox for matching after tracking
                    bbox_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                    detection_types[bbox_key] = v['type']
            
            # Update tracker
                if len(detections) > 0:
                    tracks = tracker.update(np.array(detections), timestamp=datetime.utcnow())
                else:
                    tracks = tracker.update(np.empty((0, 5)), timestamp=datetime.utcnow())
            
            # Process each tracked vehicle
                for track in tracks:
                    x1, y1, x2, y2, track_id = track
                    track_id = int(track_id)
                
                    # Find matching vehicle type (approximate match)
                    bbox_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                    vehicle_type = detection_types.get(bbox_key, 'car')
                
                    # Create vehicle record if new track
                    if track_id not in tracked_vehicles:
                        # Extract vehicle region for plate recognition (only on first detection)
                        vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]
                        plate_text = None
                        province = None
                    
                        if vehicle_img.size > 0:
                            plate_info = self.plate_recognizer.recognize_plate(vehicle_img)
                            if plate_info and plate_info.get('plate_text'):
                                plate_text = plate_info['plate_text']
                                province_info = self.province_detector.detect_province(plate_text)
                                province = province_info.get('province_name') if province_info else None
                    
                    # Create vehicle database record
                        db_vehicle = Vehicle(
                            track_id=track_id,
                            vehicle_type=vehicle_type,
                            confidence=float(detections[0][4]) if len(detections) > 0 else 0.9,
                            bbox={'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                            plate_text=plate_text,
                            province=province,
                            timestamp=datetime.utcnow()
                        )
                        db_session.add(db_vehicle)
                        db_session.flush()  # Get the ID without full commit
                    
                        tracked_vehicles[track_id] = db_vehicle
                        vehicle_types[track_id] = vehicle_type
                    
                        print(f"New track {track_id}: {vehicle_type}, plate: {plate_text}")
                
                    # Log vehicle position
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                
                    log = VehicleLog(
                        vehicle_id=tracked_vehicles[track_id].id,
                        track_id=track_id,
                        position_x=float(center_x),
                        position_y=float(center_y),
                        frame_number=frame_count,
                        event_type='tracked',
                        timestamp=datetime.utcnow()
                    )
                    db_session.add(log)
            
                # Progress update
                if frame_count % 30 == 0:
                    stats = tracker.get_stats()
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Tracks: {stats['current_vehicles']} | "
                          f"Entries: {stats['total_entries']} | Exits: {stats['total_exits']}")
        
            # Process actual line-crossing events from tracker
            print("\nProcessing line-crossing events...")
            
            # Save entry events (vehicles crossing INTO city)
            for event in tracker.entry_events:
                track_id = event['track_id']
                if track_id in tracked_vehicles:
                    bbox = event['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    entry_log = VehicleLog(
                        vehicle_id=tracked_vehicles[track_id].id,
                        track_id=track_id,
                        position_x=float(center_x),
                        position_y=float(center_y),
                        frame_number=event['frame'],
                        event_type='entry',
                        timestamp=event['timestamp'] or datetime.utcnow()
                    )
                    db_session.add(entry_log)
            
            # Save exit events (vehicles crossing OUT OF city)
            for event in tracker.exit_events:
                track_id = event['track_id']
                if track_id in tracked_vehicles:
                    bbox = event['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    exit_log = VehicleLog(
                        vehicle_id=tracked_vehicles[track_id].id,
                        track_id=track_id,
                        position_x=float(center_x),
                        position_y=float(center_y),
                        frame_number=event['frame'],
                        event_type='exit',
                        timestamp=event['timestamp'] or datetime.utcnow()
                    )
                    db_session.add(exit_log)
            
            db_session.commit()
            
            print(f"✓ Saved {len(tracker.entry_events)} entry events")
            print(f"✓ Saved {len(tracker.exit_events)} exit events")
        
            cap.release()
        
            # Get final stats from tracker
            final_stats = tracker.get_stats()
        
            print(f"✓ Video processed: {frame_count} frames, {len(tracked_vehicles)} vehicles tracked")
        
            return {
                'total_frames': total_frames,
                'processed_frames': frame_count,
                'total_vehicles': len(tracked_vehicles),
                'total_entries': final_stats['total_entries'],
                'total_exits': final_stats['total_exits'],
                'fps': fps
        }
    
        finally:
            # Clean up temp file
            if os.path.exists(video_path):
                os.remove(video_path)
        
            if close_session:
                db_session.close()
    
    def get_analytics(self, db_session=None):
        """
        Get analytics from database.
        
        Args:
            db_session: SQLAlchemy session (optional)
        """
        close_session = False
        if db_session is None:
            db_session = SessionLocal()
            close_session = True
        
        try:
            # Total vehicles
            total = db_session.query(Vehicle).count()
            
            # Count by vehicle type
            by_type_query = db_session.query(
                Vehicle.vehicle_type,
                func.count(Vehicle.id)
            ).group_by(Vehicle.vehicle_type).all()
            
            by_type = {vtype: count for vtype, count in by_type_query}
            
            # Count by province
            by_province_query = db_session.query(
                Vehicle.province,
                func.count(Vehicle.id)
            ).filter(Vehicle.province.isnot(None)).group_by(Vehicle.province).all()
            
            by_province = {province: count for province, count in by_province_query}
            
            # Count entry/exit events
            entry_count = db_session.query(VehicleLog).filter(
                VehicleLog.event_type == 'entry'
            ).count()
            
            exit_count = db_session.query(VehicleLog).filter(
                VehicleLog.event_type == 'exit'
            ).count()
            
            return {
                'total_vehicles': total,
                'total_entries': entry_count,
                'total_exits': exit_count,
                'total_traffic_flow': entry_count + exit_count,
                'by_type': by_type,
                'by_province': by_province
            }
        finally:
            if close_session:
                db_session.close()
    
    def get_recent_vehicles(self, limit=10, db_session=None):
        """
        Get most recent vehicle detections from database.
        
        Args:
            limit: Number of records to return
            db_session: SQLAlchemy session (optional)
        """
        close_session = False
        if db_session is None:
            db_session = SessionLocal()
            close_session = True
        
        try:
            vehicles = db_session.query(Vehicle).order_by(
                Vehicle.timestamp.desc()
            ).limit(limit).all()
            
            return [{
                'id': v.id,
                'type': v.vehicle_type,
                'confidence': v.confidence,
                'bbox': v.bbox,
                'plate_text': v.plate_text,
                'province': v.province,
                'timestamp': v.timestamp.isoformat()
            } for v in vehicles]
        finally:
            if close_session:
                db_session.close()
    
    def reset_analytics(self, db_session=None):
        """
        Clear all vehicle records from database.
        WARNING: This deletes all data!
        
        Args:
            db_session: SQLAlchemy session (optional)
        """
        close_session = False
        if db_session is None:
            db_session = SessionLocal()
            close_session = True
        
        try:
            # Delete all vehicle logs first (foreign key constraint)
            db_session.query(VehicleLog).delete()
            # Delete all vehicles
            db_session.query(Vehicle).delete()
            db_session.commit()
            
            return {'success': True, 'message': 'Analytics reset successfully'}
        finally:
            if close_session:
                db_session.close()
