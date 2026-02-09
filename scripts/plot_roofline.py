import numpy as np
import matplotlib.pyplot as plt

labels_cpu = ["CPU", "AVX", "AVX2", "AVX512", "16Cores"]
labels_gpu = ["GPU FP64", "GPU FP32"]

# Roofline parameters for phase II
peak_performances_cpu = np.array([12, 12*2, 12*4, 12*8, 12*16])  # in GFlops/s
memory_bandwidths_cpu = np.array([39, 39, 39, 39, 460.8])  # in GB/s

peak_performances_gpu = np.array([124.8, 7987.2])  # in GFlops/s
memory_bandwidths_gpu = np.array([288, 288])  # in GB/s

# Operational intensities (FLOPs/Byte)
print("CPU Ridge points:", peak_performances_cpu / memory_bandwidths_cpu)
print("GPU Ridge points:", peak_performances_gpu / memory_bandwidths_gpu)

# Create the log-log roofline plot
plt.figure(figsize=(6, 4))

# Plot roofline CPU
for i in range(len(peak_performances_cpu)):
    x = np.linspace(0.001,  2**10, 100000)

    y = np.minimum(x*memory_bandwidths_cpu[i], peak_performances_cpu[i])
    plt.plot(x, y, label=labels_cpu[i])

# Plot roofline GPU
for i in range(len(peak_performances_gpu)):
    x = np.linspace(0.001,  2**10, 100000)

    y = np.minimum(x*memory_bandwidths_gpu[i], peak_performances_gpu[i])
    plt.plot(x, y, label=labels_gpu[i])

# add su3matmat
# plt.vlines(0.4583, 0.001, 100, linestyles='dashdot', colors="black", label="SU(3) matmul")

# Add labels and legend
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPs/s)')
plt.ylim([1e-2, 1e4])
plt.xlim([1e-2, 1e3])
plt.xscale('log')
plt.yscale('log')
plt.legend()

# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig("roofline.pdf")
