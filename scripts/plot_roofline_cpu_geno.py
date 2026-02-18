import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Create the log-log roofline plot
plt.figure(figsize=(5, 3))
plt.style.use("seaborn-v0_8-whitegrid")

# parameters
threads = [1,2,4,8,16]
input_file = "../output/volume_geno_cpu_float.csv"
plot_file = "../output/roofline_cpu_geno_float.pdf"
precision = "float"

perf_1core = 12 # in GFlops/s
memb_1core = 30 # in GB/s
socket_bw = 460.8 # in GB/s

peak_performances_cpu = [perf_1core*t for t in threads]
memory_bandwidths_cpu = [memb_1core*t if memb_1core*t<socket_bw else socket_bw for t in threads]

markers = Line2D.filled_markers
colors_cpu = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

for i, t in enumerate(threads):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_cpu[i], peak_performances_cpu[i])
    line, = plt.plot(x, y, alpha=0.7, color=colors_cpu[i])

    #  add label
    ridge_x = peak_performances_cpu[i] / memory_bandwidths_cpu[i]
    label_x = 90
    label_y = peak_performances_cpu[i]*0.5
    plt.text(label_x, label_y, f"CPU {t} cores", color=line.get_color(), fontsize=10, ha="left", va="bottom")


# add points
aos_I = {
    "double": 0.7397,
    "float": 0.7397*2,
} #flops/byte
aos_P = 432 #flops

# add kenrel lines su3matmat
plt.vlines(aos_I[precision], 0.001, 1e5, linestyles='dashed', colors="black", label="plaq_sum", alpha=0.7, zorder=-1)

df_soa = pd.read_csv(input_file)
df_soa["performance"]= aos_P*df_soa["vol"]/df_soa["avg_s"]*1e-9
df_soa["op_int"]= aos_I[precision]

compute = df_soa[df_soa["phase"] == "compute"]


compute = df_soa[df_soa["phase"] == "compute"]
for i, t in enumerate(threads):
    aost = compute[compute["threads"] == t]
    plt.scatter(
        aost["op_int"]+0.005*t, aost["performance"], 
        label=f"{t} threads", marker=markers[i]
    )
    aost["vol per thread"] = aost["vol"]/aost["threads"]
    print(aost.head(10))

aos16 = compute[compute["threads"] == 16]
for x, y, v in zip(aos16["op_int"], aos16["performance"], aos16["vol"]):
    plt.text(x+0.25,y,str(v),fontsize=9,color="tab:purple",ha="left",va="center")

# Add labels and legend
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPs/s)')
plt.ylim([1e-1, 1e4])
plt.xlim([1e-2, 1e3])
plt.xscale('log')
plt.yscale('log')
plt.legend()

# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_file)
