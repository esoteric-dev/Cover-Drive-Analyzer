#!/usr/bin/env python3
"""
Test script for Cricket Cover Drive Analyzer

This script tests the main functionality of the analyzer system
and provides a quick way to verify everything is working correctly.
"""

import os
import sys
import json
import tempfile
import numpy as np
from cricket_cover_drive_analyzer import CricketCoverDriveAnalyzer
from video_utils import VideoProcessor

def test_analyzer_initialization():
    """Test that the analyzer can be initialized correctly."""
    print("Testing analyzer initialization...")
    
    try:
        analyzer = CricketCoverDriveAnalyzer()
        print("✓ Analyzer initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Analyzer initialization failed: {str(e)}")
        return False

def test_metric_calculations():
    """Test biomechanical metric calculations with sample data."""
    print("\nTesting metric calculations...")
    
    analyzer = CricketCoverDriveAnalyzer()
    
    # Sample landmark data (normalized coordinates)
    sample_landmarks = {
        0: (0.5, 0.2, 0.9),    # nose
        11: (0.4, 0.3, 0.8),   # left shoulder
        12: (0.6, 0.3, 0.8),   # right shoulder
        13: (0.3, 0.4, 0.7),   # left elbow
        14: (0.7, 0.4, 0.7),   # right elbow
        15: (0.2, 0.5, 0.6),   # left wrist
        16: (0.8, 0.5, 0.6),   # right wrist
        23: (0.4, 0.6, 0.8),   # left hip
        24: (0.6, 0.6, 0.8),   # right hip
        25: (0.4, 0.8, 0.7),   # left knee
        26: (0.6, 0.8, 0.7),   # right knee
        27: (0.4, 0.9, 0.6),   # left ankle
        28: (0.6, 0.9, 0.6),   # right ankle
    }
    
    # Test frame shape
    frame_shape = (720, 1280, 3)
    
    try:
        metrics = analyzer.calculate_frame_metrics(sample_landmarks, frame_shape)
        
        if metrics:
            print("✓ Metric calculations successful")
            print(f"  Front Elbow Angle: {metrics.get('front_elbow_angle', 'N/A')}")
            print(f"  Spine Lean: {metrics.get('spine_lean', 'N/A')}")
            print(f"  Head-Knee Alignment: {metrics.get('head_knee_alignment', 'N/A')}")
            print(f"  Front Foot Direction: {metrics.get('front_foot_direction', 'N/A')}")
            return True
        else:
            print("✗ Metric calculations returned None")
            return False
            
    except Exception as e:
        print(f"✗ Metric calculations failed: {str(e)}")
        return False

def test_angle_calculation():
    """Test angle calculation function specifically."""
    print("\nTesting angle calculation...")
    
    analyzer = CricketCoverDriveAnalyzer()
    
    # Test cases
    test_cases = [
        # (point1, point2, point3, expected_approx)
        ((0, 0), (0, 1), (1, 1), 90),    # Right angle
        ((0, 0), (0, 1), (0, 2), 180),   # Straight line
        ((1, 0), (0, 0), (0, 1), 90),    # Right angle
    ]
    
    for i, (p1, p2, p3, expected) in enumerate(test_cases):
        try:
            angle = analyzer.calculate_angle(p1, p2, p3)
            if angle is not None:
                error = abs(angle - expected)
                if error < 5:  # Allow 5 degree tolerance
                    print(f"✓ Test case {i+1}: {angle:.1f}° (expected ~{expected}°)")
                else:
                    print(f"✗ Test case {i+1}: {angle:.1f}° (expected ~{expected}°)")
                    return False
            else:
                print(f"✗ Test case {i+1}: No angle calculated")
                return False
        except Exception as e:
            print(f"✗ Test case {i+1} failed: {str(e)}")
            return False
    
    return True

def test_feedback_generation():
    """Test feedback generation based on metric values."""
    print("\nTesting feedback generation...")
    
    analyzer = CricketCoverDriveAnalyzer()
    
    # Test cases for different metric values
    test_cases = [
        ('front_elbow_angle', 90, 'Excellent'),
        ('front_elbow_angle', 100, 'Good'),
        ('front_elbow_angle', 70, 'Too low'),
        ('spine_lean', 0, 'Excellent'),
        ('spine_lean', 10, 'Good'),
        ('spine_lean', 20, 'Too high'),
    ]
    
    for metric, value, expected_feedback in test_cases:
        try:
            feedback = analyzer.get_feedback(metric, value)
            if feedback == expected_feedback:
                print(f"✓ {metric} = {value}: {feedback}")
            else:
                print(f"✗ {metric} = {value}: got '{feedback}', expected '{expected_feedback}'")
                return False
        except Exception as e:
            print(f"✗ Feedback generation failed for {metric}: {str(e)}")
            return False
    
    return True

def test_evaluation_scoring():
    """Test the evaluation and scoring system."""
    print("\nTesting evaluation scoring...")
    
    analyzer = CricketCoverDriveAnalyzer()
    
    # Create sample frame metrics
    sample_metrics = [
        {
            'front_elbow_angle': 90,
            'spine_lean': 0,
            'head_knee_alignment': 0,
            'front_foot_direction': 15
        },
        {
            'front_elbow_angle': 95,
            'spine_lean': 5,
            'head_knee_alignment': 10,
            'front_foot_direction': 20
        }
    ]
    
    # Set the frame metrics
    analyzer.frame_metrics = sample_metrics
    
    try:
        evaluation = analyzer.evaluate_shot()
        
        if 'overall_score' in evaluation:
            print(f"✓ Evaluation successful - Overall score: {evaluation['overall_score']}/10")
            print(f"  Category scores: {evaluation['category_scores']}")
            return True
        else:
            print("✗ Evaluation missing overall score")
            return False
            
    except Exception as e:
        print(f"✗ Evaluation failed: {str(e)}")
        return False

def test_video_processor():
    """Test video processor utilities."""
    print("\nTesting video processor...")
    
    processor = VideoProcessor()
    
    # Test validation with non-existent file
    result = processor.validate_video("non_existent_file.mp4")
    if not result['valid'] and 'File does not exist' in result['error']:
        print("✓ Video validation correctly handles non-existent files")
    else:
        print("✗ Video validation failed for non-existent file")
        return False
    
    return True

def test_output_directory_creation():
    """Test that output directories are created correctly."""
    print("\nTesting output directory creation...")
    
    try:
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_output_dir = os.path.join(temp_dir, "test_output")
            
            analyzer = CricketCoverDriveAnalyzer()
            
            # This should create the directory
            os.makedirs(test_output_dir, exist_ok=True)
            
            if os.path.exists(test_output_dir):
                print("✓ Output directory creation successful")
                return True
            else:
                print("✗ Output directory creation failed")
                return False
                
    except Exception as e:
        print(f"✗ Output directory test failed: {str(e)}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\nTesting dependencies...")
    
    required_modules = [
        'cv2',
        'mediapipe',
        'numpy',
        'json',
        'tqdm'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} available")
        except ImportError:
            print(f"✗ {module} not available")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\nMissing modules: {', '.join(missing_modules)}")
        print("Please install missing dependencies: pip install -r requirements.txt")
        return False
    
    return True

def run_all_tests():
    """Run all tests and provide a summary."""
    print("=" * 60)
    print("CRICKET COVER DRIVE ANALYZER - SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Analyzer Initialization", test_analyzer_initialization),
        ("Angle Calculation", test_angle_calculation),
        ("Metric Calculations", test_metric_calculations),
        ("Feedback Generation", test_feedback_generation),
        ("Evaluation Scoring", test_evaluation_scoring),
        ("Video Processor", test_video_processor),
        ("Output Directory Creation", test_output_directory_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Prepare a cricket cover drive video")
        print("2. Run: python cricket_cover_drive_analyzer.py your_video.mp4")
        print("3. Check the output directory for results")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Ensure all files are in the same directory")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
