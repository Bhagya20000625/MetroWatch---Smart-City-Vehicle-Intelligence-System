"""
Train YOLOv8 License Plate Detector
Downloads dataset from Roboflow and trains a custom model
"""

from ultralytics import YOLO
import os


def download_dataset():
    print("=" * 80)
    print("Manual Dataset Download Required")
    print("=" * 80)
    print("\nPlease download the dataset manually from:")
    print("https://universe.roboflow.com/roboflow-universe/license-plate-recognition-rxg4e")
    print("\nExtract to: license-plate-recognition-4/")
    print("=" * 80)
    return None


def train_model(dataset_path, epochs=50):
    print("\n" + "=" * 80)
    print("Training YOLOv8 License Plate Detector")
    print("=" * 80)
    
    model = YOLO('yolov8m.pt')
    
    print("\n📊 Training Configuration:")
    print(f"   Base Model: YOLOv8m")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: 640")
    print(f"   Batch Size: 16 (auto-adjusted based on your system)")
    print(f"   Device: {'GPU' if YOLO.device else 'CPU'}")
    print("=" * 80)
    
    # Train the model
    results = model.train(
        data=f"{dataset_path}/data.yaml", 
        epochs=epochs,                      
        imgsz=640,                          
        batch=16,                           
        name='license_plate_detector',      
        patience=10,                        
        save=True,                         
        plots=True,                         
        verbose=True,
        workers=0                           # Windows compatibility fix
    )
    
    print("\n✓ Training completed!")
    print("=" * 80)
    
    return results


def save_model(model_path='runs/detect/license_plate_detector/weights/best.pt'):
    print("\n" + "=" * 80)
    print("Saving Trained Model")
    print("=" * 80)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Copy best model to models folder
    import shutil
    destination = 'models/license_plate_detector.pt'
    
    if os.path.exists(model_path):
        shutil.copy(model_path, destination)
        print(f"✓ Model saved to: {destination}")
        print(f"   Size: {os.path.getsize(destination) / (1024*1024):.2f} MB")
    else:
        print(f"⚠️ Model not found at: {model_path}")
        print("   Check the training output for the correct path")
    
    print("=" * 80)


def test_model():
    print("\n" + "=" * 80)
    print("Testing Trained Model")
    print("=" * 80)
    
    model_path = 'models/license_plate_detector.pt'
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train first.")
        return
    
    model = YOLO(model_path)
    
    # Test on a sample image if available
    test_images = [
        "data/images/car_withplate2.jpg",
        "data/images/test_image.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📸 Testing on: {img_path}")
            results = model(img_path)
            
            if results[0].boxes:
                print(f"   ✓ Detected {len(results[0].boxes)} plate(s)")
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    print(f"      Confidence: {conf:.2f}")
            else:
                print("   ❌ No plates detected")
            break
    
    print("=" * 80)


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("YOLOv8 LICENSE PLATE DETECTOR TRAINING")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Download license plate dataset (~400MB)")
    print("2. Train YOLOv8n model (15-20 min on CPU, 2-3 min on GPU)")
    print("3. Save trained model to models/license_plate_detector.pt")
    print("\n⚠️  Make sure you have:")
    print("   - Good internet connection (for dataset download)")
    print("   - ~2GB free disk space")
    print("   - Time to wait for training")
    print("=" * 80)
    
    # Check if dataset already downloaded manually
    manual_dataset_path = "license-plate-recognition-4"
    
    if os.path.exists(manual_dataset_path):
        print(f"\n✓ Found manually downloaded dataset at: {manual_dataset_path}")
        dataset_path = manual_dataset_path
    else:
        print("\n❌ Dataset not found!")
        print(f"   Expected location: {os.path.abspath(manual_dataset_path)}")
        print("\nPlease download manually from:")
        print("https://universe.roboflow.com/roboflow-universe/license-plate-recognition-rxg4e")
        print("Extract to the project root as: license-plate-recognition-4/")
        print("=" * 80)
        exit(1)
    
    input("\nPress Enter to start training or Ctrl+C to cancel...")
    
    try:
        # Train model
        print("\n⏳ Starting training... This will take a while!")
        print("   You can monitor progress in the terminal")
        train_model(dataset_path, epochs=50)
        
        # Save model
        save_model()
        
        # Test model
        test_model()
        
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! License Plate Detector is Ready!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. The model is saved at: models/license_plate_detector.pt")
        print("2. Run tests/test_full_pipeline.py to see improved results")
        print("3. The plate_recognizer.py will automatically use this model")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Check internet connection")
        print("- Make sure you have enough disk space")
        print("- Try running again")
