import cv2
import numpy as np
import sys
import time
from process import Process
from video import Video

def process_video(video_path):
    video_input = Video()
    video_input.dirname = video_path
    video_input.start()
    processor = Process()

    timestamps = []
    bpm_results = []

    print("Processing video...")

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
            timestamp = time.time() - processor.start_time

            if bpm > 0:
                print(f"Time: {timestamp:.2f}s | BPM: {bpm:.2f}")
                timestamps.append(timestamp)
                bpm_results.append(bpm)

    finally:
        video_input.stop()

    # 可选：保存结果
    with open("bpm_output.csv", "w") as f:
        f.write("Time(s),BPM\n")
        for t, b in zip(timestamps, bpm_results):
            f.write(f"{t:.2f},{b:.2f}\n")

    print("Done. Results saved to bpm_output.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_heart_rate.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    process_video(video_path)
