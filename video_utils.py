#!/usr/bin/env python3
"""
Video Utilities for Cricket Cover Drive Analyzer

This module provides utilities for video preprocessing, validation,
and format conversion to ensure optimal analysis results.
"""

import cv2
import os
import sys
import argparse
from typing import Tuple, Optional, Dict
import json

class VideoProcessor:
    """Utility class for video preprocessing and validation."""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        self.recommended_codecs = ['h264', 'mp4v', 'XVID']
    
    def validate_video(self, video_path: str) -> Dict:
        """
        Validate a video file for cricket analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing validation results and recommendations
        """
        if not os.path.exists(video_path):
            return {
                'valid': False,
                'error': 'File does not exist',
                'recommendations': []
            }
        
        # Check file extension
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext not in self.supported_formats:
            return {
                'valid': False,
                'error': f'Unsupported file format: {file_ext}',
                'recommendations': [f'Convert to one of: {", ".join(self.supported_formats)}']
            }
        
        # Open video and get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'valid': False,
                'error': 'Could not open video file',
                'recommendations': ['Check if file is corrupted', 'Try converting to MP4 format']
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        # Validate properties
        issues = []
        recommendations = []
        
        if fps < 15:
            issues.append(f'Low frame rate: {fps:.1f} fps (recommended: 30+ fps)')
            recommendations.append('Record at higher frame rate for better analysis')
        
        if width < 640 or height < 480:
            issues.append(f'Low resolution: {width}x{height} (recommended: 1280x720+)')
            recommendations.append('Use higher resolution for better pose detection')
        
        if duration < 1:
            issues.append(f'Very short duration: {duration:.1f}s (recommended: 2-10s)')
            recommendations.append('Record longer video to capture complete shot')
        elif duration > 30:
            issues.append(f'Long duration: {duration:.1f}s (recommended: 2-10s)')
            recommendations.append('Consider trimming to focus on the shot')
        
        if total_frames < 30:
            issues.append(f'Too few frames: {total_frames} (recommended: 60+)')
            recommendations.append('Record longer video or use higher frame rate')
        
        return {
            'valid': len(issues) == 0,
            'properties': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'duration': duration,
                'format': file_ext
            },
            'issues': issues,
            'recommendations': recommendations
        }
    
    def convert_video(self, input_path: str, output_path: str, 
                     target_fps: int = 30, target_resolution: Tuple[int, int] = (1280, 720)) -> bool:
        """
        Convert video to optimal format for analysis.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            target_fps: Target frame rate
            target_resolution: Target resolution (width, height)
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error: Could not open input video: {input_path}")
                return False
            
            # Get input properties
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, target_fps, target_resolution)
            
            print(f"Converting video...")
            print(f"Input: {input_width}x{input_height} @ {input_fps:.1f}fps")
            print(f"Output: {target_resolution[0]}x{target_resolution[1]} @ {target_fps}fps")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame
                resized_frame = cv2.resize(frame, target_resolution)
                
                # Write frame
                out.write(resized_frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")
            
            # Release resources
            cap.release()
            out.release()
            
            print(f"Conversion complete! Output saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            return False
    
    def extract_sample_frames(self, video_path: str, output_dir: str, 
                            num_frames: int = 5) -> bool:
        """
        Extract sample frames from video for preview.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save sample frames
            num_frames: Number of frames to extract
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video: {video_path}")
                return False
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = total_frames // (num_frames + 1)
            
            print(f"Extracting {num_frames} sample frames...")
            
            for i in range(num_frames):
                frame_pos = (i + 1) * frame_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                ret, frame = cap.read()
                if ret:
                    output_path = os.path.join(output_dir, f"sample_frame_{i+1:02d}.jpg")
                    cv2.imwrite(output_path, frame)
                    print(f"Saved: {output_path}")
            
            cap.release()
            print(f"Sample frames saved to: {output_dir}")
            return True
            
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return False
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get detailed information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if error
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            info = {
                'path': video_path,
                'filename': os.path.basename(video_path),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
                'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }
            
            info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
            info['aspect_ratio'] = info['width'] / info['height'] if info['height'] > 0 else 0
            
            cap.release()
            return info
            
        except Exception as e:
            print(f"Error getting video info: {str(e)}")
            return None

def main():
    """Command-line interface for video utilities."""
    parser = argparse.ArgumentParser(description="Video Utilities for Cricket Cover Drive Analyzer")
    parser.add_argument("action", choices=['validate', 'convert', 'info', 'extract'],
                       help="Action to perform")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Output path (for convert/extract actions)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for conversion")
    parser.add_argument("--width", type=int, default=1280, help="Target width for conversion")
    parser.add_argument("--height", type=int, default=720, help="Target height for conversion")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to extract")
    
    args = parser.parse_args()
    
    processor = VideoProcessor()
    
    if args.action == 'validate':
        result = processor.validate_video(args.input_video)
        print(f"\nVideo Validation Results:")
        print(f"Valid: {result['valid']}")
        
        if result['valid']:
            print("✓ Video is suitable for analysis")
        else:
            print(f"✗ {result['error']}")
        
        if 'properties' in result:
            props = result['properties']
            print(f"\nVideo Properties:")
            print(f"  Resolution: {props['width']}x{props['height']}")
            print(f"  Frame Rate: {props['fps']:.1f} fps")
            print(f"  Duration: {props['duration']:.1f} seconds")
            print(f"  Total Frames: {props['total_frames']}")
            print(f"  Format: {props['format']}")
        
        if result['issues']:
            print(f"\nIssues Found:")
            for issue in result['issues']:
                print(f"  ⚠ {issue}")
        
        if result['recommendations']:
            print(f"\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  💡 {rec}")
    
    elif args.action == 'convert':
        if not args.output:
            print("Error: Output path required for conversion")
            return
        
        success = processor.convert_video(
            args.input_video, 
            args.output, 
            args.fps, 
            (args.width, args.height)
        )
        
        if success:
            print("✓ Video conversion completed successfully")
        else:
            print("✗ Video conversion failed")
    
    elif args.action == 'info':
        info = processor.get_video_info(args.input_video)
        if info:
            print(f"\nVideo Information:")
            print(f"  File: {info['filename']}")
            print(f"  Resolution: {info['width']}x{info['height']}")
            print(f"  Frame Rate: {info['fps']:.1f} fps")
            print(f"  Duration: {info['duration']:.1f} seconds")
            print(f"  Total Frames: {info['total_frames']}")
            print(f"  File Size: {info['file_size_mb']:.1f} MB")
            print(f"  Aspect Ratio: {info['aspect_ratio']:.2f}")
        else:
            print("✗ Could not read video information")
    
    elif args.action == 'extract':
        if not args.output:
            args.output = "sample_frames"
        
        success = processor.extract_sample_frames(
            args.input_video, 
            args.output, 
            args.frames
        )
        
        if success:
            print("✓ Frame extraction completed successfully")
        else:
            print("✗ Frame extraction failed")

if __name__ == "__main__":
    main()
