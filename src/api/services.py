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
                'current_count': entry_count - exit_count,
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
