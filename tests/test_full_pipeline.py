"""
Full Pipeline Integration Test
Tests the complete vehicle analytics system:
1. Vehicle Detection (YOLOv8)
2. License Plate Recognition (EasyOCR)
3. Province Detection

Usage: python tests/test_full_pipeline.py
"""

import sys
import os
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.vehicle_detector import VehicleDetector
from src.ocr.plate_recognizer import LicensePlateRecognizer
from src.utils.province_detector import ProvinceDetector


class VehicleAnalyticsPipeline:
    
    def __init__(self, vehicle_model='yolov8x.pt'):

        print("=" * 80)
        print("Initializing Vehicle Analytics Pipeline")
        print("=" * 80)
        
        # Initialize components
        self.vehicle_detector = VehicleDetector(model_name=vehicle_model)
        self.plate_recognizer = LicensePlateRecognizer()
        self.province_detector = ProvinceDetector()
        
        print("\n✓ All components initialized successfully!")
        print("=" * 80)
    
    def process_image(self, image_path, confidence_threshold=0.3):

        print(f"\n📸 Processing: {image_path}")
        print("-" * 80)
        
        # Detect vehicles
        print("\n[1/3] Detecting vehicles...")
        vehicles, img = self.vehicle_detector.detect_vehicles(
            image_path, 
            confidence_threshold=confidence_threshold
        )
        
        if not vehicles:
            print("❌ No vehicles detected in the image.")
            return []
        
        print(f"✓ Found {len(vehicles)} vehicle(s)")
        results = []
        
        for i, vehicle in enumerate(vehicles, 1):
            print(f"\n[Vehicle {i}] Type: {vehicle['type']}, Confidence: {vehicle['confidence']}")
            
            # Crop vehicle region
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_img = img[y1:y2, x1:x2]
            
            if vehicle_img.size == 0:
                print("  ⚠️ Invalid vehicle crop, skipping...")
                continue
            
            print(f"  [2/3] Recognizing license plate...")
            plate_result = self.plate_recognizer.recognize_plate(vehicle_img)
            
            if plate_result:
                plate_text = plate_result['plate_text']
                plate_confidence = plate_result['confidence']
                print(f"  ✓ Plate: {plate_text} (confidence: {plate_confidence})")
                

                print(f"  [3/3] Detecting province...")
                province_result = self.province_detector.detect_province(plate_text)
                
                print(f"  ✓ Province: {province_result['province_name']} ({province_result['province_code']})")
                
                # Combine all data
                result = {
                    'vehicle_id': i,
                    'vehicle_type': vehicle['type'],
                    'vehicle_confidence': vehicle['confidence'],
                    'vehicle_bbox': vehicle['bbox'],
                    'plate_text': plate_text,
                    'plate_confidence': plate_confidence,
                    'plate_bbox': plate_result['bbox'],
                    'province_code': province_result['province_code'],
                    'province_name': province_result['province_name']
                }
                
                results.append(result)
            else:
                print(f"  ❌ No license plate detected for this vehicle")
        
        return results, img
    
    def visualize_results(self, img, results):
        annotated = img.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['vehicle_bbox']
            
            # Draw vehicle bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Prepare label
            label_lines = [
                f"{result['vehicle_type'].upper()}",
                f"{result['plate_text']}",
                f"{result['province_code']}"
            ]
            
            # Draw label background
            y_offset = y1 - 15
            for line in reversed(label_lines):
                (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y_offset - h - 5), (x1 + w + 10, y_offset + 5), (0, 255, 0), -1)
                cv2.putText(annotated, line, (x1 + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                y_offset -= (h + 10)
        
        return annotated
    
    def save_results(self, results, output_path='data/images/pipeline_result.jpg', annotated_img=None):
        # Save annotated image
        if annotated_img is not None:
            cv2.imwrite(output_path, annotated_img)
            print(f"\n💾 Annotated image saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("PIPELINE RESULTS SUMMARY")
        print("=" * 80)
        
        for result in results:
            print(f"\n🚗 Vehicle #{result['vehicle_id']}:")
            print(f"   Type: {result['vehicle_type']}")
            print(f"   Confidence: {result['vehicle_confidence']}")
            print(f"   License Plate: {result['plate_text']}")
            print(f"   Plate Confidence: {result['plate_confidence']}")
            print(f"   Province: {result['province_name']} ({result['province_code']})")
            print("-" * 80)


# Main execution
if __name__ == "__main__":
    # Test image path
    TEST_IMAGE = r"C:\Users\Bhagya Umayanga\Downloads\Car4.jpg"
    
    # Initialize pipeline
    pipeline = VehicleAnalyticsPipeline(vehicle_model='yolov8x.pt')
    
    # Process image
    results, original_img = pipeline.process_image(TEST_IMAGE, confidence_threshold=0.3)
    
    if results:
        # Visualize results
        annotated_img = pipeline.visualize_results(original_img, results)
        
        # Save results
        pipeline.save_results(results, annotated_img=annotated_img)
        
        # Display result
        cv2.imshow('Vehicle Analytics Pipeline - Results', annotated_img)
        print("\n✨ Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\n✅ Pipeline test completed successfully!")
    else:
        print("\n⚠️ No complete results (vehicle + plate + province) found.")
        print("Tips:")
        print("- Try with an image that has clear, visible license plates")
        print("- Ensure good lighting and front-facing angle")
        print("- Sri Lankan license plates work best with this system")
