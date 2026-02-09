import numpy as np
import matplotlib.pyplot as plt

labels = ["CPU", "AVX", "AVX2", "AVX512", "GPU FP64", "GPU FP32"]

# Roofline parameters for phase II
peak_performances = np.array([12, 12*2, 12*4, 12*8, 124.8, 7987.2])  # in GFlops/s
memory_bandwidths = np.array([39, 39, 39, 39, 288, 288])  # in GB/s

# Operational intensities (FLOPs/Byte)
ops_intensity = peak_performances / memory_bandwidths
print(ops_intensity)

# Create the log-log roofline plot
plt.figure(figsize=(6, 4))

# Plot roofline
for i in range(len(peak_performances)):
    x = np.linspace(0.001,  2**10, 100000)

    y = np.minimum(x*memory_bandwidths[i], peak_performances[i])
    plt.plot(x, y, label=labels[i])

# add su3matmat
plt.vlines(0.4583, 0.001, 100, linestyles='dashdot', colors="black", label="SU(3) matmul")

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
