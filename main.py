import cv2
import mediapipe as mp
import numpy as np

class ExerciseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Exercise counters and states
        self.pushup_count = 0
        self.situp_count = 0
        self.pushup_stage = None
        self.situp_stage = None
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def detect_pushup(self, landmarks):
        """Detect push-up form and count reps"""
        # Get coordinates
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
              landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
               landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        
        # Calculate angles
        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        body_angle = self.calculate_angle(shoulder, hip, knee)
        
        # Check if body is straight (proper plank position)
        body_straight = 160 < body_angle < 200
        
        # Push-up logic
        if elbow_angle > 160 and body_straight:
            self.pushup_stage = "up"
        if elbow_angle < 90 and self.pushup_stage == "up" and body_straight:
            self.pushup_stage = "down"
            self.pushup_count += 1
            
        return elbow_angle, body_angle, body_straight
    
    def detect_situp(self, landmarks):
        """Detect sit-up form and count reps"""
        # Get coordinates
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
              landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
               landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculate angle at hip
        hip_angle = self.calculate_angle(shoulder, hip, knee)
        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        # Sit-up logic
        if hip_angle < 50:  # Sitting up position
            self.situp_stage = "up"
        if hip_angle > 120 and self.situp_stage == "up":  # Lying down position
            self.situp_stage = "down"
            self.situp_count += 1
            
        return hip_angle, knee_angle
    
    def run(self, exercise_type='pushup'):
        """Run the exercise detector"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process pose
            results = self.pose.process(image)
            
            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Process landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                if exercise_type == 'pushup':
                    elbow_angle, body_angle, body_straight = self.detect_pushup(landmarks)
                    
                    # Display info
                    cv2.putText(image, f'Push-ups: {self.pushup_count}', 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Elbow Angle: {int(elbow_angle)}', 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Body Angle: {int(body_angle)}', 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Form: {"Good" if body_straight else "Keep body straight!"}', 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if body_straight else (0, 0, 255), 2)
                    
                elif exercise_type == 'situp':
                    hip_angle, knee_angle = self.detect_situp(landmarks)
                    
                    # Display info
                    cv2.putText(image, f'Sit-ups: {self.situp_count}', 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Hip Angle: {int(hip_angle)}', 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Knee Angle: {int(knee_angle)}', 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            # Display instructions
            cv2.putText(image, f'Mode: {exercise_type.upper()}', 
                       (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, 'Press P for Push-ups | S for Sit-ups | Q to Quit', 
                       (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Exercise Pose Detector', image)
            
            # Key controls
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                exercise_type = 'pushup'
            elif key == ord('s'):
                exercise_type = 'situp'
            elif key == ord('r'):  # Reset counter
                if exercise_type == 'pushup':
                    self.pushup_count = 0
                else:
                    self.situp_count = 0
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ExerciseDetector()
    detector.run(exercise_type='pushup')  # Start with push-ups