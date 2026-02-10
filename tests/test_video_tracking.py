"""
Video Tracking Test
Tests vehicle tracking on video with entry/exit detection
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker


class VideoTracker:
    
    def __init__(self):
        print("=" * 80)
        print("Initializing Video Tracking System")
        print("=" * 80)
        
        # Initialize detector and tracker
        self.detector = VehicleDetector(model_name='yolov8n.pt')  # Use yolov8n for speed
        self.tracker = VehicleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        print("✓ System initialized!")
        print("=" * 80)
        
    def set_zones(self, frame_width, frame_height):
        """Set entry and exit detection lines"""
        # Entry line at 25% of frame height (horizontal line)
        entry_y = int(frame_height * 0.25)
        self.tracker.set_entry_line(0, entry_y, frame_width, entry_y)
        
        # Exit line at 75% of frame height (horizontal line)
        exit_y = int(frame_height * 0.75)
        self.tracker.set_exit_line(0, exit_y, frame_width, exit_y)
        
        print(f"✓ Entry line set at y={entry_y}")
        print(f"✓ Exit line set at y={exit_y}")
        
    def process_video(self, video_path, output_path='data/videos/tracked_output.mp4'):
        """Process video with tracking"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {video_path}")
            return
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video: {video_path}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Set entry/exit zones
        self.set_zones(width, height)
        
        # Setup video writer
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n🎬 Processing video...")
        print("=" * 80)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            # Detect vehicles
            detected_vehicles, _ = self.detector.detect_vehicles(frame, confidence_threshold=0.3)
            
            # Prepare detections for tracker (format: [[x1,y1,x2,y2,score], ...])
            detections = np.array([
                [v['bbox'][0], v['bbox'][1], v['bbox'][2], v['bbox'][3], v['confidence']]
                for v in detected_vehicles
            ])
            
            # Update tracker
            if len(detections) > 0:
                tracks = self.tracker.update(detections, timestamp)
            else:
                tracks = self.tracker.update(np.empty((0, 5)), timestamp)
            
            # Draw results
            annotated_frame = self._draw_tracking(frame, tracks)
            
            # Write frame
            out.write(annotated_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                stats = self.tracker.get_stats()
                progress = (frame_count / total_frames) * 100
                print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                      f"Tracks: {stats['current_vehicles']} | "
                      f"Entries: {stats['total_entries']} | "
                      f"Exits: {stats['total_exits']}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final stats
        stats = self.tracker.get_stats()
        print("=" * 80)
        print("📊 TRACKING SUMMARY")
        print("=" * 80)
        print(f"Total Frames Processed: {frame_count}")
        print(f"Total Vehicle Tracks: {stats['total_tracks']}")
        print(f"Total Entries: {stats['total_entries']}")
        print(f"Total Exits: {stats['total_exits']}")
        print(f"Net Count: {stats['total_entries'] - stats['total_exits']}")
        print("=" * 80)
        print(f"\n✓ Output saved to: {output_path}")
        
    def _draw_tracking(self, frame, tracks):
        """Draw tracked vehicles and detection lines"""
        annotated = frame.copy()
        
        # Draw entry line (green)
        if self.tracker.entry_line:
            x1, y1, x2, y2 = map(int, self.tracker.entry_line)
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated, 'ENTRY', (10, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw exit line (red)
        if self.tracker.exit_line:
            x1, y1, x2, y2 = map(int, self.tracker.exit_line)
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, 'EXIT', (10, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw tracked vehicles
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            
            # Draw bounding box
            color = self._get_track_color(track_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID: {track_id}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw stats
        stats = self.tracker.get_stats()
        stats_text = [
            f"Tracking: {stats['current_vehicles']} vehicles",
            f"Entries: {stats['total_entries']}",
            f"Exits: {stats['total_exits']}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(annotated, text, (frame.shape[1] - 250, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        return annotated
        
    def _get_track_color(self, track_id):
        """Generate consistent color for each track ID"""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(50, 255, 3)))


if __name__ == "__main__":
    # Test with a video file
    VIDEO_PATH = r"data/videos/test_traffic.mp4"
    
    print("\n" + "=" * 80)
    print("VEHICLE TRACKING TEST")
    print("=" * 80)
    print("\nThis will:")
    print("1. Load a traffic video")
    print("2. Detect vehicles in each frame")
    print("3. Track vehicles with unique IDs")
    print("4. Count entries and exits")
    print("5. Save annotated video")
    print("=" * 80)
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"\n⚠️  Video not found: {VIDEO_PATH}")
        print("\nPlease:")
        print("1. Download a traffic video from YouTube or use your own")
        print("2. Save it to: data/videos/test_traffic.mp4")
        print("3. Or update VIDEO_PATH in this script")
        print("\n" + "=" * 80)
    else:
        tracker = VideoTracker()
        tracker.process_video(VIDEO_PATH)
