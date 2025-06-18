import cv2
import numpy as np
import sys
import time
from process import Process
from video import Video
import argparse
from signal_processing import Signal_processing


def process_video(video_path, window_type='hamming', buffer_size=100, output_file=None):
    video_input = Video()
    video_input.dirname = video_path
    video_input.start()
    
    # Initialize processor with window function parameter
    processor = Process(window_type=window_type, buffer_size=buffer_size)

    timestamps = []
    bpm_results = []
    signal_data = []  # Store raw signal data for later analysis

    print(f"Processing video using window function: {window_type}")
    if buffer_size:
        print(f"Using custom buffer size: {buffer_size}")

    try:
        while True:
            frame = video_input.get_frame()
            if frame is None:
                break  # 视频结束

            processor.frame_in = frame
            success = processor.run()
            if not success:
                continue

            bpm = processor.bpm
            # Get the timestamp from the video position instead of processing time
            timestamp = video_input.get_position_seconds()

            if bpm > 0:
                print(f"Time: {timestamp:.2f}s | BPM: {bpm:.2f}")
                timestamps.append(timestamp)
                bpm_results.append(bpm)
                
                # If processor exposes raw signal data, collect it
                if hasattr(processor, 'data_buffer'):
                    signal_data.append(processor.data_buffer)

    finally:
        video_input.stop()

    # 保存结果
    save_to = output_file if output_file else "bpm_output.csv"
    with open(save_to, "w") as f:
        f.write("time(s),bpm\n")
        for t, b in zip(timestamps, bpm_results):
            f.write(f"{t:.2f},{b:.2f}\n")

    print(f"Done. Results saved to {save_to}")
    
    # If we collected signal data and there's enough of it, perform window function comparison
    if signal_data and len(signal_data) > 10:
        print("Performing window function comparison with the last segment of collected data...")
        sp = Signal_processing()
        last_segment = signal_data[-1]
        # Estimate fps based on timestamps if available
        fps = 30  # Default assumption
        if len(timestamps) > 1:
            fps = len(timestamps) / (timestamps[-1] - timestamps[0])
            
        # Generate comparison of window functions
        results = sp.compare_window_functions(last_segment, fps)
        print(f"Window function comparison saved to window_comparison.png")
        
        # TODO: implement buffer size tradeoff analysis if needed
        # # If buffer_size wasn't specified, analyze different buffer sizes
        # if not buffer_size:
        #     buffer_results = sp.analyze_buffer_size_tradeoff(last_segment, fps)
        #     print(f"Buffer size analysis saved to buffer_size_analysis.png")

    return timestamps, bpm_results

def list_window_functions():
    """List all available window functions"""
    sp = Signal_processing()
    print("\nAvailable window functions:")
    for window_name in sp.window_types.keys():
        print(f"  - {window_name}")
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video to measure heart rate.")
    parser.add_argument("-f", "--video_path", type=str, help="Path to the input video file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV file for BPM results")
    parser.add_argument("-w", "--window", type=str, default="hamming", 
                        help="Window function type for FFT (e.g., hamming, hanning, blackman, kaiser, flattop, none)")
    parser.add_argument("-b", "--buffer_size", type=int, default=None, 
                        help="Buffer size for FFT (powers of 2 recommended for efficiency)")
    parser.add_argument("-l", "--list_windows", action="store_true", 
                        help="List all available window functions")
    args = parser.parse_args()
    
    # Display list of available window functions if requested
    if args.list_windows:
        list_window_functions()
        sys.exit(0)
    
    video_path = args.video_path
    if not video_path:
        print("Please provide a video path using -f or --video_path argument.")
        sys.exit(1)

    output_file = args.output
    if output_file:
        print(f"Results will be saved to {output_file}")
    
    # Process video with specified window function
    process_video(video_path, window_type=args.window, buffer_size=args.buffer_size, output_file=output_file)
