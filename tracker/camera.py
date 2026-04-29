
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import os
import numpy as np
from .models import Exercise

class VideoCamera:
    def __init__(self, exercise_id=None):
        self.video = cv2.VideoCapture(0)  # 0 is usually the default webcam
        self.exercise_id = exercise_id
        self.activity_type = 'OPEN_CLOSE'
        if self.exercise_id:
            try:
                exercise = Exercise.objects.get(pk=self.exercise_id)
                self.activity_type = exercise.activity_type
            except Exercise.DoesNotExist:
                pass

        self.hand_detected = False
        self.rep_counter = 0
        # Set initial state based on exercise
        initial_states = {
            'OPEN_CLOSE': 'open',
            'PINCH': 'released',
            'SPREAD': 'together',
            'CURL': 'straight'
        }
        self.rep_state = initial_states.get(self.activity_type, 'open')
        
        model_path = os.path.join(os.path.dirname(__file__), 'static', 'tracker', 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        # Sharpening: Increased confidence thresholds to reduce false positives and jitter
        options = HandLandmarkerOptions(
            base_options=base_options, num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # index
            (0, 9), (9, 10), (10, 11), (11, 12),  # middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
            (5, 9), (9, 13), (13, 17)  # palm
        ]

    def __del__(self):
        self.release()

    def get_frame(self):
        success, image = self.video.read()
        if success:
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = self.landmarker.detect(mp_image)
            if result.hand_landmarks:
                self.hand_detected = True
                h, w, _ = image.shape
                for hand_landmarks in result.hand_landmarks:
                    points = []
                    for landmark in hand_landmarks:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        points.append((cx, cy))
                        cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)

                    for start, end in self.connections:
                        cv2.line(image, points[start], points[end], (0, 255, 0), 2)

                # Process exercise logic using World Landmarks (3D metric) for accuracy
                # Only track the first hand to avoid state conflicts
                if result.hand_world_landmarks:
                    self.process_exercise(result.hand_world_landmarks[0])
                elif result.hand_landmarks:
                    self.process_exercise(result.hand_landmarks[0])

                # Display state
                cv2.putText(image, f'State: {self.rep_state.upper()}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                self.hand_detected = False
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        return None

    def release(self):
        # We check isOpened so this is safe to call multiple times (e.g. from __del__)
        if self.video.isOpened():
            self.video.release()
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()

    def is_hand_detected(self):
        return self.hand_detected

    def calculate_angle(self, v1, v2):
        # Using the dot product formula to find the angle between two vectors
        # cos(theta) = (v1 . v2) / (|v1| * |v2|)
        # theta = arccos((v1 . v2) / (|v1| * |v2|))
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return None # Avoid division by zero
        cosine_angle = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
        return np.degrees(cosine_angle)

    def process_exercise(self, landmarks):
        if self.activity_type == 'OPEN_CLOSE':
            self.track_open_close(landmarks)
        elif self.activity_type == 'PINCH':
            self.track_pinch(landmarks)
        elif self.activity_type == 'SPREAD':
            self.track_spread(landmarks)
        elif self.activity_type == 'CURL':
            self.track_curl(landmarks)

    def get_activity_name(self):
        names = {
            'OPEN_CLOSE': 'Open/Close',
            'PINCH': 'Pinch',
            'SPREAD': 'Spread',
            'CURL': 'Curl'
        }
        return names.get(self.activity_type, self.activity_type)

    def _get_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _get_reference_scale(self, landmarks):
        # Distance between Wrist (0) and Middle Finger MCP (9) used for normalization
        return self._get_distance(landmarks[0], landmarks[9])

    def _save_rep(self):
        if self.exercise_id:
            try:
                exercise = Exercise.objects.get(pk=self.exercise_id)
                exercise.reps_count = self.rep_counter
                exercise.save(update_fields=['reps_count'])
            except Exercise.DoesNotExist:
                pass

    def track_open_close(self, landmarks):
        # Logic: Average distance from finger tips to wrist
        tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
        wrist = landmarks[0]        
        avg_dist = np.mean([self._get_distance(landmarks[i], wrist) for i in tips])
        
        # Normalize against hand size (scale invariant)
        scale = self._get_reference_scale(landmarks)
        if scale == 0: return
        ratio = avg_dist / scale

        # Thresholds: ~2.3x scale is open, ~1.3x scale is closed (fist)
        if self.rep_state == 'open' and ratio < 1.3:
            self.rep_state = 'closed'
        elif self.rep_state == 'closed' and ratio > 1.8:
            self.rep_state = 'open'
            self.rep_counter += 1
            self._save_rep()

    def track_pinch(self, landmarks):
        # Logic: Distance between Thumb tip (4) and Index tip (8)
        dist = self._get_distance(landmarks[4], landmarks[8])
        
        # Normalize
        scale = self._get_reference_scale(landmarks)
        if scale == 0: return
        ratio = dist / scale

        # Thresholds: Pinch is usually < 0.3 of palm size
        if self.rep_state == 'released' and ratio < 0.3:
            self.rep_state = 'pinched'
        elif self.rep_state == 'pinched' and ratio > 0.6:
            self.rep_state = 'released'
            self.rep_counter += 1
            self._save_rep()

    def track_spread(self, landmarks):
        # Logic: Distance between Index tip (8) and Pinky tip (20), normalized
        dist = self._get_distance(landmarks[8], landmarks[20])
        scale = self._get_reference_scale(landmarks)
        if scale == 0: return
        ratio = dist / scale

        # Thresholds: Spread > 1.5x palm size, Together < 1.0x
        if self.rep_state == 'together' and ratio > 1.5:
            self.rep_state = 'spread' # Active state
        elif self.rep_state == 'spread' and ratio < 1.0:
            self.rep_state = 'together' # Relaxed state
            self.rep_counter += 1
            self._save_rep()

    def track_curl(self, landmarks):
        # Logic: Average flexion angle of 4 fingers
        flexions = []
        for i in range(5, 18, 4):
            angle = self.calculate_flexion(landmarks[i], landmarks[i+1], landmarks[i+2], landmarks[i+3])
            if angle is not None: flexions.append(angle)
        
        if not flexions: return
        avg_angle = np.mean(flexions)

        # Thresholds: Straight > 160 deg, Curled < 100 deg
        if self.rep_state == 'straight' and avg_angle < 100:
            self.rep_state = 'curled'
        elif self.rep_state == 'curled' and avg_angle > 160:
            self.rep_state = 'straight'
            self.rep_counter += 1
            self._save_rep()

    def calculate_flexion(self, mcp, pip, dip, tip):
        # Calculate angles between joints to determine flexion
        try:
            mcp = np.array([mcp.x, mcp.y, mcp.z])
            pip = np.array([pip.x, pip.y, pip.z])
            dip = np.array([dip.x, dip.y, dip.z])
            tip = np.array([tip.x, tip.y, tip.z])

            # Calculate vectors for interior angles (straight = 180 degrees)
            # Angle at PIP (Proximal Interphalangeal Joint)
            v_pip_mcp = mcp - pip  # Vector back to knuckle
            v_pip_dip = dip - pip  # Vector out to next joint
            angle1 = self.calculate_angle(v_pip_mcp, v_pip_dip)

            # Angle at DIP (Distal Interphalangeal Joint)
            v_dip_pip = pip - dip  # Vector back to PIP
            v_dip_tip = tip - dip  # Vector out to tip
            angle2 = self.calculate_angle(v_dip_pip, v_dip_tip)

            if angle1 is None or angle2 is None:
                return None
            return (angle1 + angle2) / 2  # Average flexion angle
        except Exception as e:
            print(f"Error calculating flexion: {e}")
     
            return None