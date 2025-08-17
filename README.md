# Cricket Cover Drive Analyzer

A Python-based system for real-time biomechanical analysis of cricket cover drive shots using computer vision and pose estimation. The system processes video frames sequentially, extracts key biomechanical metrics, and provides comprehensive feedback on shot technique.

## Features

- **Real-time Pose Estimation**: Uses MediaPipe for accurate body landmark detection
- **Biomechanical Metrics**: Calculates key angles and alignments for cricket technique
- **Live Video Overlays**: Displays pose skeleton and real-time metrics on video
- **Comprehensive Evaluation**: Provides detailed scoring and actionable feedback
- **Robust Error Handling**: Gracefully handles occlusions and missing joint data
- **Multiple Output Formats**: Generates annotated video, JSON evaluation, and detailed metrics

## Biomechanical Metrics Analyzed

1. **Front Elbow Angle** (shoulder-elbow-wrist)
   - Optimal range: 80-120° (target: 90°)
   - Critical for swing control and power generation

2. **Spine Lean** (hip-shoulder line vs. vertical)
   - Optimal range: -15° to +15° (target: 0°)
   - Essential for balance and shot stability

3. **Head-over-Knee Alignment** (projected distance)
   - Optimal range: -20 to +20 pixels (target: 0)
   - Ensures proper head position over the ball

4. **Front Foot Direction** (ankle alignment)
   - Optimal range: 0-45° (target: 15°)
   - Critical for proper footwork and shot direction

## System Requirements

- Python 3.8 or higher
- Windows 10/11, macOS, or Linux
- Webcam or video file input
- Sufficient RAM for video processing (4GB+ recommended)

## Installation

1. **Clone or download the project files**:
   ```bash
   git clone <repository-url>
   cd cricket-cover-drive-analyzer
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Analyze a cricket cover drive video:

```bash
python cricket_cover_drive_analyzer.py path/to/your/video.mp4
```

### Advanced Usage

Specify a custom output directory:

```bash
python cricket_cover_drive_analyzer.py path/to/your/video.mp4 --output-dir my_analysis_results
```

### Programmatic Usage

```python
from cricket_cover_drive_analyzer import CricketCoverDriveAnalyzer

# Create analyzer instance
analyzer = CricketCoverDriveAnalyzer()

# Analyze video
evaluation = analyzer.analyze_video("path/to/video.mp4", "output_directory")

# Access results
print(f"Overall Score: {evaluation['overall_score']}/10")
print(f"Category Scores: {evaluation['category_scores']}")
```

## Output Files

The system generates the following files in the output directory:

1. **`annotated_cover_drive.mp4`**: Video with pose skeleton and real-time metrics overlay
2. **`shot_evaluation.json`**: Comprehensive evaluation results including:
   - Overall score (1-10)
   - Category scores for each aspect
   - Detailed feedback for improvement
   - Average metrics across all frames
3. **`frame_metrics.json`**: Detailed frame-by-frame metrics for further analysis

### Sample Evaluation Output

```json
{
  "overall_score": 7,
  "category_scores": {
    "swing_control": 8,
    "balance": 7,
    "head_position": 6,
    "footwork": 8,
    "follow_through": 7
  },
  "feedback": {
    "swing_control": "Good elbow control, minor adjustments needed",
    "balance": "Good balance, slight adjustments needed",
    "head_position": "Good head position, minor adjustments needed",
    "footwork": "Excellent foot positioning and direction",
    "follow_through": "Good follow-through, maintain consistency"
  },
  "average_metrics": {
    "front_elbow_angle": 92.5,
    "spine_lean": 3.2,
    "head_knee_alignment": 8.1,
    "front_foot_direction": 18.3
  },
  "total_frames": 150
}
```

## Video Requirements

For optimal analysis, ensure your video meets these criteria:

- **Resolution**: 720p or higher recommended
- **Frame Rate**: 30fps or higher for smooth analysis
- **Lighting**: Good, even lighting to ensure pose detection accuracy
- **Camera Angle**: Side view or 3/4 view of the batsman
- **Duration**: 2-10 seconds covering the complete shot motion
- **Format**: MP4, AVI, MOV, or other common video formats

## Troubleshooting

### Common Issues

1. **"No pose detected" errors**:
   - Ensure good lighting conditions
   - Check that the full body is visible in the frame
   - Try adjusting camera angle or distance

2. **Poor metric accuracy**:
   - Verify video quality and resolution
   - Ensure consistent lighting throughout the video
   - Check that the batsman is clearly visible

3. **Memory issues with large videos**:
   - Process shorter video segments
   - Reduce video resolution if necessary
   - Close other applications to free up RAM

4. **Video codec issues**:
   - Convert video to MP4 format using H.264 codec
   - Use tools like FFmpeg for video conversion

### Performance Optimization

- **GPU Acceleration**: Install CUDA-enabled OpenCV for faster processing
- **Batch Processing**: Process multiple videos sequentially
- **Resolution Scaling**: Reduce video resolution for faster processing

## Technical Details

### Pose Estimation
- Uses MediaPipe Pose with 33 body landmarks
- Model complexity: 2 (high accuracy)
- Detection confidence threshold: 0.5
- Tracking confidence threshold: 0.5

### Metric Calculations
- **Angle Calculations**: Uses vector mathematics for precise angle measurements
- **Coordinate Systems**: Converts normalized MediaPipe coordinates to pixel coordinates
- **Error Handling**: Gracefully handles missing landmarks and occlusions

### Evaluation Algorithm
- **Scoring System**: 1-10 scale based on deviation from optimal ranges
- **Category Weighting**: Equal weighting across all categories
- **Feedback Generation**: Context-aware feedback based on performance levels

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for the excellent pose estimation library
- OpenCV community for computer vision tools
- Cricket coaching community for biomechanical insights

## Support

For support, questions, or feature requests, please open an issue on the project repository or contact the development team.

---

**Note**: This system is designed for educational and training purposes. For professional coaching applications, consider consulting with qualified cricket coaches and biomechanics experts.
