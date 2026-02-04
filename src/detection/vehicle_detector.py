from ultralytics import YOLO
import cv2
import os


class VehicleDetector: 
    # COCO dataset class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_name='yolov8n.pt'):
        
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")
    
    def detect_vehicles(self, image_path, confidence_threshold=0.5):
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
        cv2.imwrite(output_path, img)
        print(f"Result saved to: {output_path}")


# usage
if __name__ == "__main__":
    # Initialize detector
    detector = VehicleDetector(model_name='yolov8x.pt') 
    
    #  Detect vehicles in an image
    image_path = "data/images/test_image2.jpg" 
    
    # Detect vehicles
    vehicles, original_img = detector.detect_vehicles(image_path, confidence_threshold=0.3)
    
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
        
        # Display the result
        cv2.imshow('Vehicle Detection', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
