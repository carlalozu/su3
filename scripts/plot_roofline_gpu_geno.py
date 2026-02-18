import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


# Create the log-log roofline plot
plt.figure(figsize=(5, 3))
plt.style.use("seaborn-v0_8-whitegrid")

# add points
aos_I = 0.7397*2 #flops/byte
aos_P = 432 #flops

# parameters
threads = [1,8,16]
input_file = "../output/volume_geno_cpu_float.csv"
plot_file = "../output/roofline_gpu_geno_float.pdf"
precision = "float"

perf_1core = 12 # in GFlops/s
memb_1core = 30 # in GB/s
socket_bw = 460.8 # in GB/s

peak_performances_cpu = [perf_1core*t for t in threads]
memory_bandwidths_cpu = [memb_1core*t if memb_1core*t<socket_bw else socket_bw for t in threads]

markers = Line2D.filled_markers

for i, t in enumerate(threads):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_cpu[i], peak_performances_cpu[i])
    line, = plt.plot(x, y, alpha=0.7)

    #  add label
    ridge_x = peak_performances_cpu[i] / memory_bandwidths_cpu[i]
    label_y = peak_performances_cpu[i]*0.5
    plt.text(90, label_y, f"CPU {t} cores", color=line.get_color(), fontsize=9, ha="left", va="bottom")


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
    label_y = peak_performances_gpu[i] * 0.9
    plt.text(90, label_y, labels_gpu[i], color=line.get_color(), fontsize=9, ha="left", va="top")


# add kenrel lines su3matmat
plt.vlines(aos_I, 0.001, 1e5, linestyles='dashed', colors="black", label="plaq_sum", alpha=0.7, zorder=-1)

df_soa = pd.read_csv(input_file)
df_soa["performance"]= aos_P*df_soa["vol"]/df_soa["avg_s"]*1e-9
df_soa["op_int"]= aos_I

compute_gpu = df_soa[df_soa["phase"] == "compute_GPU"]
plt.scatter(compute_gpu["op_int"]+0.2, compute_gpu["performance"], label="GPU FP32", marker=">", color="tab:brown", zorder=4)
# plt.scatter(compute_gpu["op_int"]+0.2, compute_gpu["performance"], label="GPU FP64", marker=">", color="tab:pink", zorder=4)


compute = df_soa[df_soa["phase"] == "compute"]
for i, t in enumerate(threads):
    aost = compute[compute["threads"] == t]
    plt.scatter(
        aost["op_int"]+0.005*t, aost["performance"], 
        label=f"{t} threads", marker=markers[i]
    )
    aost["vol per thread"] = aost["vol"]/aost["threads"]
    # print(aost.head(10))

# for x, y, v in zip(compute_gpu["op_int"], compute_gpu["performance"], compute_gpu["vol"]):
#     plt.text(x,y,str(v),fontsize=9,color="tab:brown",ha="left",va="center")

# Add labels and legend
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPs/s)')
plt.ylim([1e0, 1e5])
plt.xlim([1e-2, 1e3])
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=9)

# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_file)
