import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from threading import Thread
import queue

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
        
        # Feedback flags to avoid repetitive warnings
        self.last_feedback = ""
        self.feedback_cooldown = 0
        
        # Initialize text-to-speech engine in a separate thread
        self.tts_queue = queue.Queue()
        self.tts_thread = Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
    def _tts_worker(self):
        """Worker thread for text-to-speech to avoid blocking main loop"""
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        while True:
            text = self.tts_queue.get()
            if text is None:  # Poison pill to stop thread
                break
            engine.say(text)
            engine.runAndWait()
            
    def speak(self, text):
        """Add text to speech queue"""
        # Only speak if it's different from last feedback (avoid spam)
        if text != self.last_feedback or self.feedback_cooldown == 0:
            self.tts_queue.put(text)
            self.last_feedback = text
            self.feedback_cooldown = 30  # Frames to wait before repeating same feedback
        
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
        
        # Decrease feedback cooldown
        if self.feedback_cooldown > 0:
            self.feedback_cooldown -= 1
        
        # Push-up logic with audio feedback
        if elbow_angle > 160:
            if body_straight:
                self.pushup_stage = "up"
            else:
                # Body not straight in up position
                if self.feedback_cooldown == 0:
                    self.speak("Straighten body")
                    
        elif elbow_angle < 90 and self.pushup_stage == "up":
            if body_straight:
                self.pushup_stage = "down"
                self.pushup_count += 1
                # Announce the count
                self.speak(str(self.pushup_count))
                self.last_feedback = ""  # Reset to allow other feedback
            else:
                # Body not straight while going down
                if self.feedback_cooldown == 0:
                    self.speak("Straighten body")
                    
        elif 90 <= elbow_angle <= 160 and self.pushup_stage == "up":
            # In the lowering phase but not deep enough
            if elbow_angle > 110 and self.feedback_cooldown == 0:
                self.speak("Go lower")
            
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
        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        
        # Calculate angles
        hip_angle = self.calculate_angle(shoulder, hip, knee)
        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        # Calculate distance between elbow and knee (for touch detection)
        elbow_knee_distance = np.sqrt((elbow[0] - knee[0])**2 + (elbow[1] - knee[1])**2)
        
        # Decrease feedback cooldown
        if self.feedback_cooldown > 0:
            self.feedback_cooldown -= 1
        
        # Sit-up logic with audio feedback
        if hip_angle < 50:  # Sitting up position
            # Check if elbow touched knee (distance threshold)
            if elbow_knee_distance < 0.15:  # Threshold for "touch" (normalized coordinates)
                if self.situp_stage == "down":
                    self.situp_stage = "up"
            else:
                # Not going high enough
                if self.situp_stage == "down" and self.feedback_cooldown == 0:
                    self.speak("Go higher")
                    
        elif hip_angle > 140:  # Lying down position (stricter than before)
            if self.situp_stage == "up":
                self.situp_stage = "down"
                self.situp_count += 1
                # Announce the count
                self.speak(str(self.situp_count))
                self.last_feedback = ""  # Reset to allow other feedback
            elif self.situp_stage is None:
                self.situp_stage = "down"
                
        elif 120 <= hip_angle <= 140 and self.situp_stage == "up":
            # Not lying down fully
            if self.feedback_cooldown == 0:
                self.speak("Lie flat down")
            
        return hip_angle, knee_angle, elbow_knee_distance
    
    def run(self, exercise_type='pushup'):
        """Run the exercise detector"""
        cap = cv2.VideoCapture(0)
        
        # Welcome message
        self.speak(f"Starting {exercise_type} mode")
        
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
                    cv2.putText(image, f'Form: {"Good" if body_straight else "Straighten body!"}', 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if body_straight else (0, 0, 255), 2)
                    
                elif exercise_type == 'situp':
                    hip_angle, knee_angle, elbow_knee_dist = self.detect_situp(landmarks)
                    
                    # Display info
                    cv2.putText(image, f'Sit-ups: {self.situp_count}', 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Hip Angle: {int(hip_angle)}', 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Knee Angle: {int(knee_angle)}', 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Elbow-Knee Dist: {elbow_knee_dist:.2f}', 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
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
            cv2.putText(image, 'Press P for Push-ups | S for Sit-ups | R to Reset | Q to Quit', 
                       (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow('Exercise Pose Detector', image)
            
            # Key controls
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                if exercise_type != 'pushup':
                    exercise_type = 'pushup'
                    self.speak("Push-up mode")
            elif key == ord('s'):
                if exercise_type != 'situp':
                    exercise_type = 'situp'
                    self.speak("Sit-up mode")
            elif key == ord('r'):  # Reset counter
                if exercise_type == 'pushup':
                    self.pushup_count = 0
                    self.speak("Push-up counter reset")
                else:
                    self.situp_count = 0
                    self.speak("Sit-up counter reset")
        
        # Cleanup
        self.tts_queue.put(None)  # Stop TTS thread
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ExerciseDetector()
    detector.run(exercise_type='pushup')  # Start with push-ups