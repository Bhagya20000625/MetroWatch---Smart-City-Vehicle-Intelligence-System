"""
Vehicle Tracker using SORT Algorithm
Tracks vehicles across frames, assigns IDs, and detects entry/exit events
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox):
        # State: [x, y, s, r, dx, dy, ds] 
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        # Measurement matrix (we measure position and scale)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        # Measurement noise
        self.kf.R[2:,2:] *= 10.
        
        # Process noise
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """Update tracker with new detection"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        
    def predict(self):
        """Predict next state"""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        """Return current bounding box"""
        return self.convert_x_to_bbox(self.kf.x)
        
    def convert_bbox_to_z(self, bbox):
        """Convert bounding box to measurement format [x,y,s,r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h   
        r = w / float(h)  
        return np.array([x, y, s, r]).reshape((4, 1))
        
    def convert_x_to_bbox(self, x, score=None):
        """Convert state to bounding box [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    
    intersection = w * h
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - intersection
    
    return intersection / union


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
    iou_matrix = iou_batch(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Use Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack([row_ind, col_ind], axis=1)
    else:
        matched_indices = np.empty((0,2), dtype=int)
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)
    
    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
            
    if len(matches) == 0:
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class VehicleTracker:

    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
        # Entry/Exit tracking
        self.entry_line = None  
        self.exit_line = None   
        self.entry_events = []  
        self.exit_events = []    
        self.vehicle_history = {}  
        
    def set_entry_line(self, x1, y1, x2, y2):
        """Set entry detection line"""
        self.entry_line = (x1, y1, x2, y2)
        
    def set_exit_line(self, x1, y1, x2, y2):
        """Set exit detection line"""
        self.exit_line = (x1, y1, x2, y2)
        
    def update(self, detections, timestamp=None):
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :])
            
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i,:])
            self.trackers.append(trk)
            
        # Prepare output
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                track_id = trk.id + 1  # +1 as MOT benchmark requires positive IDs
                ret.append(np.concatenate((d, [track_id])).reshape(1, -1))
                
                # Track position history
                center_x = (d[0] + d[2]) / 2
                center_y = (d[1] + d[3]) / 2
                if track_id not in self.vehicle_history:
                    self.vehicle_history[track_id] = []
                self.vehicle_history[track_id].append((center_x, center_y, self.frame_count))
                
                # Check entry/exit events
                self._check_line_crossing(track_id, d, timestamp)
                
            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
        
    def _check_line_crossing(self, track_id, bbox, timestamp):
        """Check if vehicle crossed entry or exit line"""
        if track_id not in self.vehicle_history or len(self.vehicle_history[track_id]) < 2:
            return
            
        # Get last two positions
        prev_pos = self.vehicle_history[track_id][-2]
        curr_pos = self.vehicle_history[track_id][-1]
        
        # Check entry line
        if self.entry_line:
            if self._line_crossed(prev_pos, curr_pos, self.entry_line):
                self.entry_events.append({
                    'track_id': track_id,
                    'frame': self.frame_count,
                    'timestamp': timestamp,
                    'bbox': bbox,
                    'position': curr_pos
                })
                print(f"✓ Entry detected: Vehicle #{track_id}")
                
        # Check exit line
        if self.exit_line:
            if self._line_crossed(prev_pos, curr_pos, self.exit_line):
                self.exit_events.append({
                    'track_id': track_id,
                    'frame': self.frame_count,
                    'timestamp': timestamp,
                    'bbox': bbox,
                    'position': curr_pos
                })
                print(f"✓ Exit detected: Vehicle #{track_id}")
                
    def _line_crossed(self, prev_pos, curr_pos, line):
        """Check if movement from prev_pos to curr_pos crossed the line"""
        x1, y1, x2, y2 = line
        px, py, _ = prev_pos
        cx, cy, _ = curr_pos
        
        # Check if line segment (prev -> curr) intersects with detection line
        # Using cross product method
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        A = (x1, y1)
        B = (x2, y2)
        C = (px, py)
        D = (cx, cy)
        
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        
    def get_stats(self):
        """Get tracking statistics"""
        return {
            'total_entries': len(self.entry_events),
            'total_exits': len(self.exit_events),
            'current_vehicles': len(self.trackers),
            'total_tracks': len(self.vehicle_history)
        }
        
    def reset(self):
        """Reset tracker"""
        self.trackers = []
        self.frame_count = 0
        self.entry_events = []
        self.exit_events = []
        self.vehicle_history = {}
        KalmanBoxTracker.count = 0
