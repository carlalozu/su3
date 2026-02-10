import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create the log-log roofline plot
plt.figure(figsize=(6, 4))
plt.style.use("seaborn-v0_8-whitegrid")

# Plot roofline CPU
# labels_cpu = ["CPU", "AVX", "AVX2", "AVX512", "16Cores"]
# peak_performances_cpu = np.array([12, 12*2, 12*4, 12*8, 12*16])  # in GFlops/s
# memory_bandwidths_cpu = np.array([39, 39, 39, 39, 460.8])  # in GB/s

labels_cpu = ["CPU 1 core", "CPU 2 cores", "CPU 4 cores", "CPU 8 cores", "CPU 16 cores"]
peak_performances_cpu = np.array([12, 12*2, 12*4, 12*8, 12*16])  # in GFlops/s
memory_bandwidths_cpu = np.array([30, 30*2, 30*4, 30*8, 30*16])  # in GB/s
print("CPU Ridge points:", peak_performances_cpu / memory_bandwidths_cpu)
for i in range(len(peak_performances_cpu)):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_cpu[i], peak_performances_cpu[i])
    line, = plt.plot(x, y, alpha=0.5)

    #  add label
    ridge_x = peak_performances_cpu[i] / memory_bandwidths_cpu[i]
    label_x = 100
    label_y = peak_performances_cpu[i] * 0.95
    plt.text(label_x, label_y, labels_cpu[i], color=line.get_color(), fontsize=10, ha="left", va="top")

# Plot roofline GPU
labels_gpu = ["GPU FP64", "GPU FP32"]
peak_performances_gpu = np.array([124.8, 7987.2])  # in GFlops/s
memory_bandwidths_gpu = np.array([288, 288])  # in GB/s
print("GPU Ridge points:", peak_performances_gpu / memory_bandwidths_gpu)
for i in range(len(peak_performances_gpu)):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_gpu[i], peak_performances_gpu[i])
    line, = plt.plot(x, y, alpha=0.5)

    #  add label
    ridge_x = peak_performances_gpu[i] / memory_bandwidths_cpu[i]
    label_x = 10
    label_y = peak_performances_gpu[i] * 0.95
    plt.text(label_x, label_y, labels_gpu[i], color=line.get_color(), fontsize=10, ha="left", va="top")

# add kenrel lines su3matmat
plt.vlines(0.7397*2, 0.001, 10000, linestyles='dashed', colors="black", label="plaq_sum", alpha=0.5, zorder=-1)
# plt.vlines(0.3724, 0.001, 10000, linestyles=':', colors="black", label="plaq_sum", alpha=0.5)

# add points
aos_I = 0.7397*2 #flops/byte
aos_P = 432 #flops
df_soa = pd.read_csv("../output/volume_float_soa.csv")
compute = df_soa[df_soa["phase"] == "compute"]
compute_gpu = df_soa[df_soa["phase"] == "compute_GPU"]

aos1 = compute[compute["threads"] == 1]
aos16 = compute[compute["threads"] == 8]

aos1["performance"]= aos_P*aos1["vol"]/aos1["avg_s"]*1e-9
aos1["op_int"]= aos_I-0.1

compute_gpu["performance"]= aos_P*compute_gpu["vol"]/compute_gpu["avg_s"]*1e-9
compute_gpu["op_int"]= aos_I+0.2

aos16["performance"]= aos_P*aos16["vol"]/aos16["avg_s"]*1e-9
aos16["op_int"]= aos_I

plt.scatter(aos1["op_int"], aos1["performance"], label="1 thread",  marker="o")
plt.scatter(aos16["op_int"], aos16["performance"], label="8 threads", marker="*", color="red")
plt.scatter(compute_gpu["op_int"], compute_gpu["performance"], label="GPU FP32", marker=">", color="magenta")

for x, y, v in zip(compute_gpu["op_int"], compute_gpu["performance"], compute_gpu["vol"]):
    plt.text(x,y,str(v),fontsize=9,color="magenta",ha="left",va="bottom")

# Add labels and legend
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPs/s)')
plt.ylim([1e-2, 1e4])
plt.xlim([1e-2, 1e3])
plt.xscale('log')
plt.yscale('log')
plt.title("plaq_sum: array of structure")
plt.legend()

# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig("roofline_aos_float.pdf")
