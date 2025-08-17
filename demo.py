#!/usr/bin/env python3
"""
Demo script for Cricket Cover Drive Analyzer

This script demonstrates how to use the analyzer with proper error handling
and provides examples of different usage patterns.
"""

import os
import sys
from cricket_cover_drive_analyzer import CricketCoverDriveAnalyzer

def demo_basic_usage():
    """Demonstrate basic usage of the analyzer."""
    print("=== Cricket Cover Drive Analyzer Demo ===\n")
    
    # Check if a video file is provided as command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Use a default video path or prompt user
        video_path = input("Enter the path to your cricket video file: ").strip()
    
    # Validate video file
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("Please provide a valid video file path.")
        return
    
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
        print("Warning: File may not be a supported video format.")
        print("Supported formats: MP4, AVI, MOV, MKV, WMV")
    
    # Create output directory
    output_dir = "demo_output"
    
    try:
        # Initialize analyzer
        print("Initializing Cricket Cover Drive Analyzer...")
        analyzer = CricketCoverDriveAnalyzer()
        
        # Analyze video
        print(f"\nStarting analysis of: {video_path}")
        print(f"Output will be saved to: {output_dir}/")
        
        evaluation = analyzer.analyze_video(video_path, output_dir)
        
        # Display results
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        
        print(f"\nOverall Performance Score: {evaluation['overall_score']}/10")
        
        print("\nDetailed Category Scores:")
        print("-" * 40)
        for category, score in evaluation['category_scores'].items():
            category_name = category.replace('_', ' ').title()
            print(f"{category_name:20} : {score}/10")
        
        print("\nActionable Feedback:")
        print("-" * 40)
        for category, feedback in evaluation['feedback'].items():
            category_name = category.replace('_', ' ').title()
            print(f"{category_name}:")
            print(f"  {feedback}")
            print()
        
        print("\nAverage Biomechanical Metrics:")
        print("-" * 40)
        for metric, value in evaluation['average_metrics'].items():
            if value is not None:
                metric_name = metric.replace('_', ' ').title()
                print(f"{metric_name:25} : {value:.1f}")
            else:
                metric_name = metric.replace('_', ' ').title()
                print(f"{metric_name:25} : No data")
        
        print(f"\nTotal frames analyzed: {evaluation['total_frames']}")
        
        # File locations
        print("\nGenerated Files:")
        print("-" * 40)
        print(f"Annotated Video: {output_dir}/annotated_cover_drive.mp4")
        print(f"Evaluation Report: {output_dir}/shot_evaluation.json")
        print(f"Detailed Metrics: {output_dir}/frame_metrics.json")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the video file is valid and not corrupted")
        print("2. Check that the video shows a clear view of the batsman")
        print("3. Verify good lighting conditions in the video")
        print("4. Make sure the full body is visible throughout the shot")
        return

def demo_programmatic_usage():
    """Demonstrate programmatic usage of the analyzer."""
    print("\n=== Programmatic Usage Example ===\n")
    
    # Example of how to use the analyzer in your own code
    code_example = '''
# Example: Programmatic usage
from cricket_cover_drive_analyzer import CricketCoverDriveAnalyzer

# Create analyzer instance
analyzer = CricketCoverDriveAnalyzer()

# Analyze video
evaluation = analyzer.analyze_video("path/to/video.mp4", "output_dir")

# Access specific results
overall_score = evaluation['overall_score']
swing_control = evaluation['category_scores']['swing_control']
balance_feedback = evaluation['feedback']['balance']

# Use results in your application
if overall_score >= 8:
    print("Excellent shot technique!")
elif overall_score >= 6:
    print("Good technique with room for improvement")
else:
    print("Focus on fundamental improvements")
'''
    
    print("You can also use the analyzer programmatically:")
    print(code_example)

def demo_batch_processing():
    """Demonstrate batch processing of multiple videos."""
    print("\n=== Batch Processing Example ===\n")
    
    batch_example = '''
# Example: Batch processing multiple videos
import os
from cricket_cover_drive_analyzer import CricketCoverDriveAnalyzer

analyzer = CricketCoverDriveAnalyzer()

video_directory = "cricket_videos/"
output_base_dir = "analysis_results/"

for video_file in os.listdir(video_directory):
    if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_directory, video_file)
        output_dir = os.path.join(output_base_dir, video_file.replace('.mp4', ''))
        
        try:
            evaluation = analyzer.analyze_video(video_path, output_dir)
            print(f"{video_file}: {evaluation['overall_score']}/10")
        except Exception as e:
            print(f"{video_file}: Error - {str(e)}")
'''
    
    print("For analyzing multiple videos:")
    print(batch_example)

def main():
    """Main demo function."""
    print("Cricket Cover Drive Analyzer - Interactive Demo")
    print("=" * 50)
    
    # Run basic demo
    demo_basic_usage()
    
    # Show additional usage examples
    demo_programmatic_usage()
    demo_batch_processing()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)
    print("\nFor more information, see the README.md file.")
    print("For technical support, check the troubleshooting section.")

if __name__ == "__main__":
    main()
