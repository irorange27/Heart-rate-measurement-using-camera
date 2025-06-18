import pandas as pd
import matplotlib.pyplot as plt

def read_and_plot(file_path):
    # 读取 csv 文件
    df = pd.read_csv(file_path)

    # 每隔10行采样
    sampled_df = df.iloc[::10]

    # 提取数据
    times = sampled_df["Time(s)"]
    bpms = sampled_df["BPM"]

    # 画散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(times, bpms, color='red', label='Sampled BPM')
    plt.title("Heart Rate Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("BPM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 用法示例
if __name__ == "__main__":
    read_and_plot("bpm_output.csv")
