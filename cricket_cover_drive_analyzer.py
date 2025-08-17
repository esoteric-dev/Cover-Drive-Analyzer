import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm
import math
from typing import Dict, List, Tuple, Optional

class CricketCoverDriveAnalyzer:
    def __init__(self):
        """Initialize the cricket cover drive analyzer with MediaPipe pose estimation."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose estimation
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Analysis results storage
        self.frame_metrics = []
        self.video_info = {}
        
        # Thresholds for feedback
        self.thresholds = {
            'front_elbow_angle': {'min': 80, 'max': 120, 'optimal': 90},
            'spine_lean': {'min': -15, 'max': 15, 'optimal': 0},
            'head_knee_alignment': {'min': -20, 'max': 20, 'optimal': 0},
            'front_foot_direction': {'min': 0, 'max': 45, 'optimal': 15}
        }
        
        # Colors for visualization
        self.colors = {
            'skeleton': (0, 255, 0),
            'metrics': (255, 255, 255),
            'feedback': (0, 255, 255),
            'warning': (0, 0, 255),
            'good': (0, 255, 0)
        }

    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """Calculate the angle between three points (point2 is the vertex)."""
        if any(p is None for p in [point1, point2, point3]):
            return None
            
        # Convert to numpy arrays
        a = np.array([point1[0], point1[1]])
        b = np.array([point2[0], point2[1]])
        c = np.array([point3[0], point3[1]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def calculate_spine_lean(self, left_shoulder: Tuple[float, float], 
                           right_shoulder: Tuple[float, float],
                           left_hip: Tuple[float, float], 
                           right_hip: Tuple[float, float]) -> float:
        """Calculate spine lean angle relative to vertical."""
        if any(p is None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return None
            
        # Calculate shoulder and hip midpoints
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2,
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2,
                  (left_hip[1] + right_hip[1]) / 2)
        
        # Calculate spine vector
        spine_vector = np.array([shoulder_mid[0] - hip_mid[0], 
                               shoulder_mid[1] - hip_mid[1]])
        
        # Vertical vector (pointing down)
        vertical_vector = np.array([0, 1])
        
        # Calculate angle
        cosine_angle = np.dot(spine_vector, vertical_vector) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        # Determine if leaning forward or backward
        if spine_vector[0] > 0:  # Leaning forward
            return np.degrees(angle)
        else:  # Leaning backward
            return -np.degrees(angle)

    def calculate_head_knee_alignment(self, nose: Tuple[float, float],
                                    left_knee: Tuple[float, float],
                                    right_knee: Tuple[float, float]) -> float:
        """Calculate head-over-knee vertical alignment."""
        if any(p is None for p in [nose, left_knee, right_knee]):
            return None
            
        # Calculate knee midpoint
        knee_mid = ((left_knee[0] + right_knee[0]) / 2,
                   (left_knee[1] + right_knee[1]) / 2)
        
        # Calculate horizontal distance between head and knee
        horizontal_distance = nose[0] - knee_mid[0]
        
        return horizontal_distance

    def calculate_front_foot_direction(self, left_ankle: Tuple[float, float],
                                     right_ankle: Tuple[float, float]) -> float:
        """Calculate front foot direction angle."""
        if any(p is None for p in [left_ankle, right_ankle]):
            return None
            
        # Calculate foot direction vector
        foot_vector = np.array([right_ankle[0] - left_ankle[0],
                              right_ankle[1] - left_ankle[1]])
        
        # Calculate angle relative to horizontal
        angle = np.degrees(np.arctan2(foot_vector[1], foot_vector[0]))
        
        return abs(angle)

    def extract_pose_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract pose landmarks from a frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = (landmark.x, landmark.y, landmark.visibility)
        
        return landmarks

    def calculate_frame_metrics(self, landmarks: Dict, frame_shape: Tuple[int, int]) -> Dict:
        """Calculate all biomechanical metrics for a frame."""
        if not landmarks:
            return None
        
        # Convert normalized coordinates to pixel coordinates
        height, width = frame_shape[:2]
        
        # Extract key points
        nose = landmarks.get(0)
        left_shoulder = landmarks.get(11)
        right_shoulder = landmarks.get(12)
        left_elbow = landmarks.get(13)
        right_elbow = landmarks.get(14)
        left_wrist = landmarks.get(15)
        right_wrist = landmarks.get(16)
        left_hip = landmarks.get(23)
        right_hip = landmarks.get(24)
        left_knee = landmarks.get(25)
        right_knee = landmarks.get(26)
        left_ankle = landmarks.get(27)
        right_ankle = landmarks.get(28)
        
        # Convert to pixel coordinates
        def to_pixels(landmark):
            if landmark is None:
                return None
            return (landmark[0] * width, landmark[1] * height)
        
        # Convert all landmarks to pixel coordinates
        nose_px = to_pixels(nose)
        left_shoulder_px = to_pixels(left_shoulder)
        right_shoulder_px = to_pixels(right_shoulder)
        left_elbow_px = to_pixels(left_elbow)
        right_elbow_px = to_pixels(right_elbow)
        left_wrist_px = to_pixels(left_wrist)
        right_wrist_px = to_pixels(right_wrist)
        left_hip_px = to_pixels(left_hip)
        right_hip_px = to_pixels(right_hip)
        left_knee_px = to_pixels(left_knee)
        right_knee_px = to_pixels(right_knee)
        left_ankle_px = to_pixels(left_ankle)
        right_ankle_px = to_pixels(right_ankle)
        
        # Calculate metrics
        metrics = {}
        
        # Front elbow angle (assuming right-handed batsman)
        metrics['front_elbow_angle'] = self.calculate_angle(
            right_shoulder_px, right_elbow_px, right_wrist_px
        )
        
        # Spine lean
        metrics['spine_lean'] = self.calculate_spine_lean(
            left_shoulder_px, right_shoulder_px, left_hip_px, right_hip_px
        )
        
        # Head-over-knee alignment
        metrics['head_knee_alignment'] = self.calculate_head_knee_alignment(
            nose_px, left_knee_px, right_knee_px
        )
        
        # Front foot direction
        metrics['front_foot_direction'] = self.calculate_front_foot_direction(
            left_ankle_px, right_ankle_px
        )
        
        return metrics

    def get_feedback(self, metric_name: str, value: float) -> str:
        """Generate feedback based on metric values and thresholds."""
        if value is None:
            return "No data"
        
        threshold = self.thresholds[metric_name]
        
        if threshold['min'] <= value <= threshold['max']:
            if abs(value - threshold['optimal']) <= 5:
                return "Excellent"
            else:
                return "Good"
        else:
            if value < threshold['min']:
                return "Too low"
            else:
                return "Too high"

    def draw_overlays(self, frame: np.ndarray, landmarks: Dict, 
                     metrics: Dict, frame_number: int) -> np.ndarray:
        """Draw pose skeleton and metrics on the frame."""
        # Draw pose skeleton
        if landmarks:
            # Create a proper MediaPipe pose landmarks object
            from mediapipe.framework.formats import landmark_pb2
            
            pose_landmarks = landmark_pb2.NormalizedLandmarkList()
            
            for idx in range(33):  # MediaPipe has 33 pose landmarks
                if idx in landmarks:
                    landmark = landmarks[idx]
                    pose_landmarks.landmark.add()
                    pose_landmarks.landmark[idx].x = landmark[0]
                    pose_landmarks.landmark[idx].y = landmark[1]
                    pose_landmarks.landmark[idx].visibility = landmark[2]
                else:
                    # Add empty landmark for missing indices
                    pose_landmarks.landmark.add()
                    pose_landmarks.landmark[idx].x = 0.0
                    pose_landmarks.landmark[idx].y = 0.0
                    pose_landmarks.landmark[idx].visibility = 0.0
            
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw metrics
        y_offset = 30
        for metric_name, value in metrics.items():
            if value is not None:
                feedback = self.get_feedback(metric_name, value)
                
                # Choose color based on feedback
                if feedback == "Excellent":
                    color = self.colors['good']
                elif feedback == "No data":
                    color = self.colors['warning']
                else:
                    color = self.colors['feedback']
                
                # Format metric name for display
                display_name = metric_name.replace('_', ' ').title()
                
                # Draw text
                text = f"{display_name}: {value:.1f}° ({feedback})"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
        
        # Draw frame number
        cv2.putText(frame, f"Frame: {frame_number}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['metrics'], 2)
        
        return frame

    def evaluate_shot(self) -> Dict:
        """Evaluate the overall shot quality and provide feedback."""
        if not self.frame_metrics:
            return {"error": "No metrics available for evaluation"}
        
        # Calculate average metrics
        valid_metrics = {key: [] for key in self.thresholds.keys()}
        
        for frame_metric in self.frame_metrics:
            if frame_metric:
                for metric_name, value in frame_metric.items():
                    if value is not None:
                        valid_metrics[metric_name].append(value)
        
        # Calculate averages
        avg_metrics = {}
        for metric_name, values in valid_metrics.items():
            if values:
                avg_metrics[metric_name] = np.mean(values)
            else:
                avg_metrics[metric_name] = None
        
        # Score each category (1-10)
        scores = {}
        feedback = {}
        
        # Front elbow angle scoring
        if avg_metrics['front_elbow_angle'] is not None:
            elbow_score = 10 - min(abs(avg_metrics['front_elbow_angle'] - 90) / 5, 10)
            scores['swing_control'] = max(1, int(elbow_score))
            if elbow_score >= 8:
                feedback['swing_control'] = "Excellent elbow control throughout the shot"
            elif elbow_score >= 6:
                feedback['swing_control'] = "Good elbow control, minor adjustments needed"
            else:
                feedback['swing_control'] = "Work on maintaining proper elbow angle during swing"
        else:
            scores['swing_control'] = 5
            feedback['swing_control'] = "Unable to assess elbow control"
        
        # Spine lean scoring
        if avg_metrics['spine_lean'] is not None:
            lean_score = 10 - min(abs(avg_metrics['spine_lean']) / 3, 10)
            scores['balance'] = max(1, int(lean_score))
            if lean_score >= 8:
                feedback['balance'] = "Excellent balance and spine alignment"
            elif lean_score >= 6:
                feedback['balance'] = "Good balance, slight adjustments needed"
            else:
                feedback['balance'] = "Focus on maintaining upright spine position"
        else:
            scores['balance'] = 5
            feedback['balance'] = "Unable to assess balance"
        
        # Head position scoring
        if avg_metrics['head_knee_alignment'] is not None:
            head_score = 10 - min(abs(avg_metrics['head_knee_alignment']) / 5, 10)
            scores['head_position'] = max(1, int(head_score))
            if head_score >= 8:
                feedback['head_position'] = "Perfect head position over the ball"
            elif head_score >= 6:
                feedback['head_position'] = "Good head position, minor adjustments needed"
            else:
                feedback['head_position'] = "Keep your head over the ball during the shot"
        else:
            scores['head_position'] = 5
            feedback['head_position'] = "Unable to assess head position"
        
        # Footwork scoring
        if avg_metrics['front_foot_direction'] is not None:
            foot_score = 10 - min(abs(avg_metrics['front_foot_direction'] - 15) / 5, 10)
            scores['footwork'] = max(1, int(foot_score))
            if foot_score >= 8:
                feedback['footwork'] = "Excellent foot positioning and direction"
            elif foot_score >= 6:
                feedback['footwork'] = "Good footwork, slight angle adjustments needed"
            else:
                feedback['footwork'] = "Work on proper front foot positioning and direction"
        else:
            scores['footwork'] = 5
            feedback['footwork'] = "Unable to assess footwork"
        
        # Overall score
        overall_score = int(np.mean(list(scores.values())))
        
        # Follow-through assessment (based on overall performance)
        if overall_score >= 8:
            scores['follow_through'] = 9
            feedback['follow_through'] = "Excellent follow-through and shot completion"
        elif overall_score >= 6:
            scores['follow_through'] = 7
            feedback['follow_through'] = "Good follow-through, maintain consistency"
        else:
            scores['follow_through'] = 5
            feedback['follow_through'] = "Focus on completing the shot with proper follow-through"
        
        return {
            "overall_score": overall_score,
            "category_scores": scores,
            "feedback": feedback,
            "average_metrics": avg_metrics,
            "total_frames": len(self.frame_metrics)
        }

    def analyze_video(self, input_video_path: str, output_dir: str = "output"):
        """Analyze a cricket cover drive video and generate annotated output."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.video_info = {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames
        }
        
        # Setup video writer
        output_video_path = os.path.join(output_dir, "annotated_cover_drive.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        print(f"Analyzing video: {input_video_path}")
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        frame_number = 0
        self.frame_metrics = []
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract pose landmarks
                landmarks = self.extract_pose_landmarks(frame)
                
                # Calculate metrics
                metrics = self.calculate_frame_metrics(landmarks, frame.shape)
                self.frame_metrics.append(metrics)
                
                # Draw overlays
                annotated_frame = self.draw_overlays(frame, landmarks, metrics, frame_number)
                
                # Write frame
                out.write(annotated_frame)
                
                frame_number += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        self.pose.close()
        
        # Evaluate shot
        evaluation = self.evaluate_shot()
        
        # Save evaluation results
        evaluation_path = os.path.join(output_dir, "shot_evaluation.json")
        with open(evaluation_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        # Save detailed metrics
        metrics_path = os.path.join(output_dir, "frame_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                "video_info": self.video_info,
                "frame_metrics": self.frame_metrics
            }, f, indent=2)
        
        print(f"\nAnalysis complete!")
        print(f"Annotated video saved to: {output_video_path}")
        print(f"Evaluation results saved to: {evaluation_path}")
        print(f"Detailed metrics saved to: {metrics_path}")
        
        # Print summary
        print(f"\n=== SHOT EVALUATION SUMMARY ===")
        print(f"Overall Score: {evaluation['overall_score']}/10")
        print(f"\nCategory Scores:")
        for category, score in evaluation['category_scores'].items():
            print(f"  {category.replace('_', ' ').title()}: {score}/10")
            print(f"    Feedback: {evaluation['feedback'][category]}")
        
        return evaluation

def main():
    """Main function to run the cricket cover drive analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cricket Cover Drive Analyzer")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return
    
    # Create analyzer and run analysis
    analyzer = CricketCoverDriveAnalyzer()
    
    try:
        evaluation = analyzer.analyze_video(args.input_video, args.output_dir)
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return

if __name__ == "__main__":
    main()
