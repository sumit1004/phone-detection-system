import cv2
import torch
from ultralytics import YOLO
import pygame
import os
import sys
from threading import Thread
import time

# Initialize pygame mixer for audio playback
pygame.mixer.init()

class PhoneDetectionSystem:
    def __init__(self, warning_audio_path="warning.mp3", confidence_threshold=0.5):
        """
        Initialize the phone detection system.
        
        Args:
            warning_audio_path: Path to the warning audio file
            confidence_threshold: Minimum confidence for detection (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.warning_audio_path = warning_audio_path
        
        # Load the YOLO model (downloads automatically on first run)
        print("[INFO] Loading YOLOv8 model... This may take a moment on first run.")
        self.model = YOLO('yolov8n.pt')  # nano model for better performance
        
        # Check if audio file exists
        self.audio_loaded = False
        self.current_sound = None
        if os.path.exists(warning_audio_path):
            try:
                self.current_sound = pygame.mixer.Sound(warning_audio_path)
                self.audio_loaded = True
                print(f"[INFO] Warning audio loaded: {warning_audio_path}")
            except Exception as e:
                print(f"[WARNING] Could not load audio file: {e}")
                print("[INFO] Audio playback will be disabled.")
        else:
            print(f"[WARNING] Audio file not found: {warning_audio_path}")
            print("[INFO] Audio playback will be disabled.")
        
        # State tracking
        self.phone_detected_previously = False
        self.is_audio_playing = False
        
    def get_class_id_for_phone(self):
        """
        Get the YOLO class ID for 'cell phone' in COCO dataset.
        In COCO, 'cell phone' has class ID 67.
        """
        return 67
    
    def start_warning_audio(self):
        """Start playing warning audio in a loop."""
        if self.audio_loaded and self.current_sound:
            try:
                if not self.is_audio_playing:
                    self.current_sound.play(loops=-1)  # -1 means loop infinitely
                    self.is_audio_playing = True
                    print("[AUDIO] Warning sound started!")
            except Exception as e:
                print(f"[ERROR] Could not play audio: {e}")
    
    def stop_warning_audio(self):
        """Stop playing warning audio."""
        if self.audio_loaded and self.current_sound:
            try:
                pygame.mixer.stop()
                self.is_audio_playing = False
                print("[AUDIO] Warning sound stopped.")
            except Exception as e:
                print(f"[ERROR] Could not stop audio: {e}")
    
    def detect_phones(self, frame):
        """
        Detect phones in a frame using YOLO.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            List of detections (boxes, confidences)
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        phone_class_id = self.get_class_id_for_phone()
        
        # Process results
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if detection is a phone and meets confidence threshold
                    if class_id == phone_class_id and confidence >= self.confidence_threshold:
                        detections.append({
                            'box': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                            'confidence': confidence
                        })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn boxes and labels
        """
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['box'])
            confidence = detection['confidence']
            
            # Draw bounding box (red for phone detected)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Prepare label text
            label_text = f"PHONE DETECTED ({confidence:.2f})"
            
            # Put label background and text
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0], y1), (0, 0, 255), -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """
        Main function to run the phone detection system.
        Press 'Q' to exit.
        """
        print("[INFO] Starting camera and detection system...")
        print("[INFO] Press 'Q' to exit.")
        
        # Open webcam (0 is the default camera)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Could not open webcam. Please check your camera connection.")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[INFO] Camera opened successfully. Detection starting...")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("[ERROR] Failed to read frame from camera.")
                    break
                
                frame_count += 1
                
                # Run phone detection
                detections = self.detect_phones(frame)
                phone_detected = len(detections) > 0
                
                # Handle audio based on phone detection state
                if phone_detected and not self.phone_detected_previously:
                    # Phone detected for the first time
                    self.start_warning_audio()
                    self.phone_detected_previously = True
                
                elif not phone_detected and self.phone_detected_previously:
                    # Phone was detected but now it's gone
                    self.stop_warning_audio()
                    self.phone_detected_previously = False
                
                # Draw detections on frame
                frame = self.draw_detections(frame, detections)
                
                # Add status text
                status_text = "PHONE DETECTED" if phone_detected else "No phone detected"
                status_color = (0, 0, 255) if phone_detected else (0, 255, 0)
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Display the frame
                cv2.imshow('Phone Detection System', frame)
                
                # Check for 'Q' key to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("[INFO] Exit key pressed. Closing application...")
                    break
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        
        finally:
            # Clean up resources
            print("[INFO] Cleaning up resources...")
            self.stop_warning_audio()
            cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            print("[INFO] Application closed successfully.")


def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("       REAL-TIME PHONE DETECTION SYSTEM WITH AUDIO ALERT")
    print("=" * 60)
    
    # Initialize and run the detection system
    detector = PhoneDetectionSystem(
        warning_audio_path="warning.mp3",
        confidence_threshold=0.5
    )
    
    detector.run()


if __name__ == "__main__":
    main()
