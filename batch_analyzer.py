#!/usr/bin/env python3
"""
Batch Analyzer for Cricket Cover Drive Videos

This script processes multiple cricket videos and generates a comprehensive
summary report comparing different shots or players.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from cricket_cover_drive_analyzer import CricketCoverDriveAnalyzer
from video_utils import VideoProcessor

class BatchAnalyzer:
    """Batch processor for multiple cricket videos."""
    
    def __init__(self):
        self.analyzer = CricketCoverDriveAnalyzer()
        self.processor = VideoProcessor()
        self.results = []
    
    def analyze_video_batch(self, input_directory: str, output_directory: str = "batch_output") -> dict:
        """
        Analyze all videos in a directory.
        
        Args:
            input_directory: Directory containing video files
            output_directory: Directory to save results
            
        Returns:
            Dictionary containing batch analysis results
        """
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(input_directory).glob(f"*{ext}"))
            video_files.extend(Path(input_directory).glob(f"*{ext.upper()}"))
        
        if not video_files:
            print(f"No video files found in {input_directory}")
            return {"error": "No video files found"}
        
        print(f"Found {len(video_files)} video files to analyze")
        
        # Process each video
        successful_analyses = 0
        failed_analyses = 0
        
        for video_file in video_files:
            print(f"\n{'='*60}")
            print(f"Analyzing: {video_file.name}")
            print(f"{'='*60}")
            
            # Create individual output directory
            video_name = video_file.stem
            individual_output = os.path.join(output_directory, video_name)
            
            try:
                # Validate video first
                validation = self.processor.validate_video(str(video_file))
                if not validation['valid']:
                    print(f"⚠ Skipping {video_file.name}: {validation['error']}")
                    failed_analyses += 1
                    continue
                
                # Analyze video
                evaluation = self.analyzer.analyze_video(str(video_file), individual_output)
                
                # Store results
                result = {
                    'filename': video_file.name,
                    'path': str(video_file),
                    'evaluation': evaluation,
                    'validation': validation,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                successful_analyses += 1
                
                print(f"✓ Analysis completed for {video_file.name}")
                print(f"  Overall Score: {evaluation['overall_score']}/10")
                
            except Exception as e:
                print(f"✗ Analysis failed for {video_file.name}: {str(e)}")
                failed_analyses += 1
        
        # Generate batch summary
        summary = self.generate_batch_summary(output_directory)
        
        print(f"\n{'='*60}")
        print("BATCH ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Failed analyses: {failed_analyses}")
        print(f"Total videos: {len(video_files)}")
        
        return summary
    
    def generate_batch_summary(self, output_directory: str) -> dict:
        """Generate a comprehensive summary of all analyses."""
        if not self.results:
            return {"error": "No results to summarize"}
        
        # Calculate statistics
        scores = [r['evaluation']['overall_score'] for r in self.results]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Category analysis
        categories = ['swing_control', 'balance', 'head_position', 'footwork', 'follow_through']
        category_stats = {}
        
        for category in categories:
            category_scores = [r['evaluation']['category_scores'][category] for r in self.results]
            category_stats[category] = {
                'average': sum(category_scores) / len(category_scores),
                'min': min(category_scores),
                'max': max(category_scores)
            }
        
        # Find best and worst performances
        best_video = max(self.results, key=lambda x: x['evaluation']['overall_score'])
        worst_video = min(self.results, key=lambda x: x['evaluation']['overall_score'])
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        summary = {
            'batch_info': {
                'total_videos': len(self.results),
                'analysis_date': datetime.now().isoformat(),
                'output_directory': output_directory
            },
            'overall_statistics': {
                'average_score': round(avg_score, 2),
                'min_score': min_score,
                'max_score': max_score,
                'score_range': max_score - min_score
            },
            'category_statistics': category_stats,
            'best_performance': {
                'filename': best_video['filename'],
                'score': best_video['evaluation']['overall_score'],
                'feedback': best_video['evaluation']['feedback']
            },
            'worst_performance': {
                'filename': worst_video['filename'],
                'score': worst_video['evaluation']['overall_score'],
                'feedback': worst_video['evaluation']['feedback']
            },
            'recommendations': recommendations,
            'detailed_results': self.results
        }
        
        # Save summary
        summary_path = os.path.join(output_directory, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate text report
        self.generate_text_report(summary, output_directory)
        
        return summary
    
    def generate_recommendations(self) -> list:
        """Generate recommendations based on batch analysis."""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # Analyze common issues
        category_scores = {}
        for category in ['swing_control', 'balance', 'head_position', 'footwork', 'follow_through']:
            scores = [r['evaluation']['category_scores'][category] for r in self.results]
            category_scores[category] = sum(scores) / len(scores)
        
        # Find weakest areas
        weakest_category = min(category_scores, key=category_scores.get)
        strongest_category = max(category_scores, key=category_scores.get)
        
        if category_scores[weakest_category] < 6:
            recommendations.append(f"Focus on improving {weakest_category.replace('_', ' ')} - average score: {category_scores[weakest_category]:.1f}/10")
        
        if category_scores[strongest_category] >= 8:
            recommendations.append(f"Excellent {strongest_category.replace('_', ' ')} performance - maintain this standard")
        
        # Overall performance recommendations
        avg_overall = sum(r['evaluation']['overall_score'] for r in self.results) / len(self.results)
        
        if avg_overall >= 8:
            recommendations.append("Overall excellent technique - focus on consistency and fine-tuning")
        elif avg_overall >= 6:
            recommendations.append("Good technique with room for improvement - work on identified weak areas")
        else:
            recommendations.append("Fundamental improvements needed - consider professional coaching")
        
        return recommendations
    
    def generate_text_report(self, summary: dict, output_directory: str):
        """Generate a human-readable text report."""
        report_path = os.path.join(output_directory, "batch_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("CRICKET COVER DRIVE BATCH ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {summary['batch_info']['analysis_date']}\n")
            f.write(f"Total Videos Analyzed: {summary['batch_info']['total_videos']}\n\n")
            
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Score: {summary['overall_statistics']['average_score']}/10\n")
            f.write(f"Score Range: {summary['overall_statistics']['min_score']} - {summary['overall_statistics']['max_score']}\n\n")
            
            f.write("CATEGORY PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            for category, stats in summary['category_statistics'].items():
                category_name = category.replace('_', ' ').title()
                f.write(f"{category_name:15}: {stats['average']:.1f}/10 (range: {stats['min']}-{stats['max']})\n")
            f.write("\n")
            
            f.write("BEST PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            f.write(f"Video: {summary['best_performance']['filename']}\n")
            f.write(f"Score: {summary['best_performance']['score']}/10\n")
            f.write("Feedback:\n")
            for category, feedback in summary['best_performance']['feedback'].items():
                category_name = category.replace('_', ' ').title()
                f.write(f"  {category_name}: {feedback}\n")
            f.write("\n")
            
            f.write("AREAS FOR IMPROVEMENT\n")
            f.write("-" * 20 + "\n")
            f.write(f"Video: {summary['worst_performance']['filename']}\n")
            f.write(f"Score: {summary['worst_performance']['score']}/10\n")
            f.write("Feedback:\n")
            for category, feedback in summary['worst_performance']['feedback'].items():
                category_name = category.replace('_', ' ').title()
                f.write(f"  {category_name}: {feedback}\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(summary['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            f.write("DETAILED RESULTS\n")
            f.write("-" * 20 + "\n")
            for result in summary['detailed_results']:
                f.write(f"\n{result['filename']}:\n")
                f.write(f"  Overall Score: {result['evaluation']['overall_score']}/10\n")
                for category, score in result['evaluation']['category_scores'].items():
                    category_name = category.replace('_', ' ').title()
                    f.write(f"  {category_name}: {score}/10\n")

def main():
    """Command-line interface for batch analysis."""
    parser = argparse.ArgumentParser(description="Batch Cricket Cover Drive Analyzer")
    parser.add_argument("input_directory", help="Directory containing video files to analyze")
    parser.add_argument("--output", "-o", default="batch_output", 
                       help="Output directory for results (default: batch_output)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate videos without analysis")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_directory):
        print(f"Error: Input directory does not exist: {args.input_directory}")
        return
    
    if args.validate_only:
        # Only validate videos
        processor = VideoProcessor()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(args.input_directory).glob(f"*{ext}"))
            video_files.extend(Path(args.input_directory).glob(f"*{ext.upper()}"))
        
        print(f"Validating {len(video_files)} video files...")
        
        for video_file in video_files:
            validation = processor.validate_video(str(video_file))
            print(f"\n{video_file.name}:")
            print(f"  Valid: {validation['valid']}")
            if not validation['valid']:
                print(f"  Error: {validation['error']}")
            if validation['issues']:
                print("  Issues:")
                for issue in validation['issues']:
                    print(f"    - {issue}")
        
        return
    
    # Run batch analysis
    batch_analyzer = BatchAnalyzer()
    
    try:
        summary = batch_analyzer.analyze_video_batch(args.input_directory, args.output)
        
        if 'error' not in summary:
            print(f"\nBatch analysis completed successfully!")
            print(f"Results saved to: {args.output}/")
            print(f"Summary: {args.output}/batch_summary.json")
            print(f"Report: {args.output}/batch_report.txt")
        else:
            print(f"Batch analysis failed: {summary['error']}")
            
    except Exception as e:
        print(f"Error during batch analysis: {str(e)}")

if __name__ == "__main__":
    main()
