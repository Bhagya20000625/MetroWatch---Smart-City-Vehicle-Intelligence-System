"""
Business Logic Layer - Vehicle Analytics Service
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.detection.vehicle_detector import VehicleDetector
from src.ocr.plate_recognizer import LicensePlateRecognizer
from src.utils.province_detector import ProvinceDetector
import cv2
import numpy as np
from datetime import datetime


class VehicleAnalyticsService:
    def __init__(self):
        print("Initializing Vehicle Analytics Service...")
        self.detector = VehicleDetector(model_name='yolov8n.pt')
        self.plate_recognizer = LicensePlateRecognizer()
        self.province_detector = ProvinceDetector()
        
        # In-memory storage (will move to database in Phase 7)
        self.vehicle_history = []
        print("✓ Service initialized successfully!")
        
    def process_frame(self, image_bytes):
        # Convert bytes to cv2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image format")
        
        # Detect vehicles
        vehicles, _ = self.detector.detect_vehicles(img, confidence_threshold=0.3)
        
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
                    province = self.province_detector.detect_province(plate_info['plate_text'])
            
            vehicle_data = {
                'type': vehicle['type'],
                'confidence': vehicle['confidence'],
                'bbox': vehicle['bbox'],
                'plate_text': plate_info['plate_text'] if plate_info else None,
                'province': province
            }
            
            results.append(vehicle_data)
            
            # Store in history
            self.vehicle_history.append({
                **vehicle_data,
                'timestamp': datetime.now()
            })
        
        return results
    
    def get_analytics(self):
        total = len(self.vehicle_history)
        by_type = {}
        by_province = {}
        
        for v in self.vehicle_history:
            # Count by vehicle type
            by_type[v['type']] = by_type.get(v['type'], 0) + 1
            
            # Count by province
            if v.get('province'):
                by_province[v['province']] = by_province.get(v['province'], 0) + 1
        
        return {
            'total_vehicles': total,
            'total_entries': 0,  # Will connect to tracker in future
            'total_exits': 0,
            'current_count': 0,
            'by_type': by_type,
            'by_province': by_province
        }
    
    def get_recent_vehicles(self, limit=10):
        """Get most recent vehicle detections"""
        recent = self.vehicle_history[-limit:] if len(self.vehicle_history) > limit else self.vehicle_history
        return [{
            **v,
            'timestamp': v['timestamp'].isoformat()
        } for v in reversed(recent)]
    
    def reset_analytics(self):
        """Clear vehicle history"""
        self.vehicle_history = []
        return {'success': True, 'message': 'Analytics reset successfully'}
