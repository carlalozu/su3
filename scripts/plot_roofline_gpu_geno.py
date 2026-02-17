import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create the log-log roofline plot
plt.figure(figsize=(5, 3))
plt.style.use("seaborn-v0_8-whitegrid")


# Plot roofline CPU
labels_cpu = ["CPU 1 core", "CPU 2 cores", "CPU 4 cores", "CPU 8 cores", "CPU 16 cores"]
colors_cpu = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
peak_performances_cpu = np.array([12, 12*2, 12*4, 12*8, 12*16])  # in GFlops/s
memory_bandwidths_cpu = np.array([30, 30*2, 30*4, 30*8, 460.8])  # in GB/s

for i in [0,3,4]:
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_cpu[i], peak_performances_cpu[i])
    line, = plt.plot(x, y, alpha=0.7, color=colors_cpu[i])

    #  add label
    ridge_x = peak_performances_cpu[i] / memory_bandwidths_cpu[i]
    label_x = 90
    label_y = peak_performances_cpu[i] * 2
    plt.text(label_x, label_y, labels_cpu[i], color=line.get_color(), fontsize=10, ha="left", va="top")



# Plot roofline GPU
labels_gpu = ["FP64 A2000", "FP32 A2000", "FP64 H200", "FP32 H200"]
peak_performances_gpu = np.array([124.8, 7987.2, 34000, 67000])  # in GFlops/s
memory_bandwidths_gpu = np.array([288, 288, 4000, 4000])  # in GB/s
colors_gpu = ["tab:pink", "tab:brown", "tab:grey", "tab:orange"]
for i in range(2):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_gpu[i], peak_performances_gpu[i])
    line, = plt.plot(x, y, alpha=0.9, color=colors_gpu[i])

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

df_soa = pd.read_csv("../output/volume_geno_cpu.csv")
df_soa["performance"]= aos_P*df_soa["vol"]/df_soa["avg_s"]*1e-9
df_soa["op_int"]= aos_I

compute = df_soa[df_soa["phase"] == "compute"]
compute_gpu = df_soa[df_soa["phase"] == "compute_GPU"]

aos1 = compute[compute["threads"] == 1]
aos4 = compute[compute["threads"] == 4]
aos8 = compute[compute["threads"] == 8]
aos16 = compute[compute["threads"] == 16]

plt.scatter(compute_gpu["op_int"]+0.2, compute_gpu["performance"], label="GPU FP64", marker=">", color="tab:pink", zorder=4)

plt.scatter(aos16["op_int"] + 0.2, aos16["performance"], label="16 threads", marker="*", color="tab:purple", zorder=4)
plt.scatter(aos8["op_int"], aos8["performance"], label="8 threads", marker="^", color="tab:red", zorder=4)
plt.scatter(aos4["op_int"], aos4["performance"], label="4 threads", marker="+", color="tab:green", zorder=4)
plt.scatter(aos1["op_int"] - 0.1, aos1["performance"], marker="o", label="1 thread", color="tab:blue", zorder=4)

for x, y, v in zip(aos16["op_int"], aos16["performance"], aos16["vol"]):
    plt.text(x+0.25,y,str(v),fontsize=9,color="tab:purple",ha="left",va="center")

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
plt.savefig("../output/roofline_gpu_geno.pdf")
