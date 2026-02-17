import numpy as np
import matplotlib.pyplot as plt

# Create the log-log roofline plot
plt.figure(figsize=(5, 3))
plt.style.use("seaborn-v0_8-whitegrid")

# Plot roofline clusters CPU
labels_cpu_socket = ["AMD EPYC 9124", "Arm Neoverse V2"]
peak_perf_cpu_socket = np.array([192, 1800])
memory_band_cpu_socket = np.array([460, 480])
scale = [2 ,0.95]

for i in range(len(peak_perf_cpu_socket)):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_band_cpu_socket[i], peak_perf_cpu_socket[i])
    line, = plt.plot(x, y, alpha=0.9)

    #  add label
    ridge_x = peak_perf_cpu_socket[i] / memory_band_cpu_socket[i]
    label_x = 50
    label_y = peak_perf_cpu_socket[i] * scale[i]
    plt.text(label_x, label_y, labels_cpu_socket[i], color=line.get_color(), fontsize=9.5, ha="left", va="top")


# Plot roofline clusters GPU
labels_gpu = ["FP64 A2000", "FP32 A2000", "FP64 H200", "FP32 H200"]
peak_performances_gpu = np.array([124.8, 7987.2, 34000, 67000])  # in GFlops/s
memory_bandwidths_gpu = np.array([288, 288, 4000, 4000])  # in GB/s
for i in range(len(peak_performances_gpu)):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_gpu[i], peak_performances_gpu[i])
    line, = plt.plot(x, y, alpha=0.9)

    #  add label
    ridge_x = peak_performances_gpu[i] / memory_bandwidths_gpu[i]
    label_x = 50
    label_y = peak_performances_gpu[i] * 0.9
    plt.text(label_x, label_y, labels_gpu[i], color=line.get_color(), fontsize=9.5, ha="left", va="top")

# add points
aos_I = 0.7397 #flops/byte
aos_P = 432 #flops

# add kenrel lines su3matmat
plt.vlines(aos_I, 0.001, 1e5, linestyles='dashed', colors="black", label="plaq_sum", alpha=0.7, zorder=-1)

# Add labels and legend
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPs/s)')
plt.ylim([1e0, 1e5])
plt.xlim([1e-2, 1e3])
plt.xscale('log')
plt.yscale('log')
plt.legend()

# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig("../output/roofline_clusters_gpus.pdf")
