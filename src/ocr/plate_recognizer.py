import easyocr
import cv2
import numpy as np
import os
import re
from ultralytics import YOLO


class LicensePlateRecognizer:
    def __init__(self, plate_model_path='models/license_plate_detector.pt'):
        
        print("Initializing License Plate Recognizer...")
        
        # Initialize EasyOCR reader (English language)
        # Set gpu=False for CPU mode (set to True if you have CUDA GPU)
        print("Loading EasyOCR (this takes a moment on first run)...")
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR loaded!")
        self.use_custom_detector = os.path.exists(plate_model_path)
        
        if self.use_custom_detector:
            print(f"Loading custom plate detector: {plate_model_path}")
            self.plate_detector = YOLO(plate_model_path)
        else:
            print("No custom plate detector found. Using region-based approach.")
            self.plate_detector = None
    
    def preprocess_plate(self, plate_img):
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Increase contrast using adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def extract_plate_region(self, img):
        height, width = img.shape[:2]
        
        # License plates are typically in the lower 40% and center 80% of vehicle
        # This is a simple heuristic approach
        y_start = int(height * 0.6)
        y_end = height
        x_start = int(width * 0.1)
        x_end = int(width * 0.9)
        
        # Extract potential plate region
        plate_region = img[y_start:y_end, x_start:x_end]
        
        return [plate_region], [(x_start, y_start, x_end, y_end)]
    
    def detect_plate_with_yolo(self, img):
        if not self.use_custom_detector:
            return self.extract_plate_region(img)
        
        results = self.plate_detector(img, verbose=False)
        plate_regions = []
        bboxes = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Crop plate region
                plate_img = img[y1:y2, x1:x2]
                plate_regions.append(plate_img)
                bboxes.append((x1, y1, x2, y2))
        
        return plate_regions, bboxes
    
    def clean_plate_text(self, text):
        # Remove all non-alphanumeric characters except spaces
        text = re.sub(r'[^A-Z0-9\s]', '', text.upper())
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Sri Lankan format: XX YYYY 9999 or XX YYY 9999 
        return text
    
    def recognize_plate(self, img):
        # Step 1: Detect plate region
        plate_regions, bboxes = self.detect_plate_with_yolo(img)
        
        if not plate_regions:
            return None
        
        results = []
        
        for plate_img, bbox in zip(plate_regions, bboxes):
            if plate_img.size == 0:
                continue
            
            # Step 2: Preprocess plate image
            processed = self.preprocess_plate(plate_img)
            
            # Step 3: Run OCR with EasyOCR
            # EasyOCR returns: [(bbox, text, confidence)]
            ocr_results = self.reader.readtext(plate_img)
            
            if not ocr_results:
                # Try on preprocessed image if first attempt fails
                ocr_results = self.reader.readtext(processed)
            
            if ocr_results:
                # Combine all detected text (sometimes plate is split into multiple detections)
                plate_text = ' '.join([result[1] for result in ocr_results])
                
                # Get average confidence
                avg_confidence = np.mean([result[2] for result in ocr_results])
                
                # Clean text
                cleaned_text = self.clean_plate_text(plate_text)
                
                results.append({
                    'plate_text': cleaned_text,
                    'confidence': round(float(avg_confidence), 2),
                    'bbox': bbox,
                    'raw_text': plate_text
                })
        
        # Return result with highest confidence
        if results:
            return max(results, key=lambda x: x['confidence'])
        
        return None
    
    def recognize_from_file(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
        
        img = cv2.imread(image_path)
        return self.recognize_plate(img)
    
    def draw_plate_detection(self, img, result):
        if result is None:
            return img
        
        img_copy = img.copy()
        bbox = result['bbox']
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle around plate
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label
        label = f"{result['plate_text']} ({result['confidence']:.2f})"
        
        # Draw label background
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_copy, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return img_copy


if __name__ == "__main__":
    # Initialize recognizer
    recognizer = LicensePlateRecognizer()
    
    # Test image path - use an image with a visible license plate
    test_image = "data/images/car_withplate3.jpg"
    
    print(f"\nProcessing: {test_image}")
    
    # Recognize plate
    result = recognizer.recognize_from_file(test_image)
    
    if result:
        print("\n--- License Plate Recognition Result ---")
        print(f"Plate Text: {result['plate_text']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Raw OCR Text: {result['raw_text']}")
        print(f"Bounding Box: {result['bbox']}")
        
        # Load image and draw result
        img = cv2.imread(test_image)
        annotated_img = recognizer.draw_plate_detection(img, result)
        
        # Save result
        output_path = "data/images/plate_result.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"\nResult saved to: {output_path}")
        
        # Display result
        cv2.imshow('License Plate Recognition', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nNo license plate detected or text could not be read.")
        print("Tips:")
        print("- Ensure the image has a clear, visible license plate")
        print("- Try with better lighting conditions")
        print("- Consider training a custom plate detector for better results")
