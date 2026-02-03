"""
Vehicle Detector using YOLOv8
This script detects vehicles (car, bus, truck, motorcycle, etc.) in images
"""

from ultralytics import YOLO
import cv2
import os


class VehicleDetector:
    """
    A class to detect vehicles in images using YOLOv8
    """
    
    # COCO dataset class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize the vehicle detector
        
        Args:
            model_name: YOLO model to use
                - yolov8n.pt (nano - fastest, less accurate)
                - yolov8s.pt (small - balanced)
                - yolov8m.pt (medium - slower, more accurate)
        """
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")
    
    def detect_vehicles(self, image_path, confidence_threshold=0.5):
        """
        Detect vehicles in an image
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score (0-1)
            
        Returns:
            List of detected vehicles with their details
        """
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
        
        # Load image
        print(f"Processing image: {image_path}")
        img = cv2.imread(image_path)
        
        # Run detection
        results = self.model(img, verbose=False)
        
        # Process results
        detected_vehicles = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Check if it's a vehicle and meets confidence threshold
                if class_id in self.VEHICLE_CLASSES and confidence >= confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    vehicle_info = {
                        'type': self.VEHICLE_CLASSES[class_id],
                        'confidence': round(confidence, 2),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    }
                    
                    detected_vehicles.append(vehicle_info)
        
        print(f"Found {len(detected_vehicles)} vehicle(s)")
        return detected_vehicles, img
    
    def draw_detections(self, img, vehicles):
        """
        Draw bounding boxes on the image
        
        Args:
            img: Original image
            vehicles: List of detected vehicles
            
        Returns:
            Image with bounding boxes drawn
        """
        img_copy = img.copy()
        
        for vehicle in vehicles:
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label = f"{vehicle['type']}: {vehicle['confidence']}"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return img_copy
    
    def save_result(self, img, output_path):
        """
        Save the result image
        
        Args:
            img: Image to save
            output_path: Where to save the image
        """
        cv2.imwrite(output_path, img)
        print(f"Result saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = VehicleDetector(model_name='yolov8n.pt')
    
    # Example: Detect vehicles in an image
    image_path = "data/images/test_image.jpg"
    
    # Detect vehicles
    vehicles, original_img = detector.detect_vehicles(image_path, confidence_threshold=0.5)
    
    if vehicles is not None:
        # Print detection results
        print("\n--- Detection Results ---")
        for i, vehicle in enumerate(vehicles, 1):
            print(f"{i}. Type: {vehicle['type']}, Confidence: {vehicle['confidence']}, BBox: {vehicle['bbox']}")
        
        # Draw bounding boxes
        result_img = detector.draw_detections(original_img, vehicles)
        
        # Save result
        output_path = "data/images/result.jpg"
        detector.save_result(result_img, output_path)
        
        # Optional: Display the result
        cv2.imshow('Vehicle Detection', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
