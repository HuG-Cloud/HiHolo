import hiholo
import mytools
import matplotlib.pyplot as plt

# Read holograms
input_file = "cali_data.h5"
input_dataset = "holodata"
holo_data = mytools.read_h5_to_double(input_file, input_dataset)
display_data = mytools.scale_display_data(holo_data[0])

direction = 0
# 0 represents vertical average, 1 represents horizontal average
# maxFre: [val1, val2, ...]
# frequencies: [[freq_data1], [freq_data2], ...], x values
# profiles: [[profile_data1], [profile_data2], ...], y values
maxFre, frequencies, profiles = hiholo.computePSDs(holo_data, direction)
print(maxFre)

nz = [20, 10, 40, 60]
wavelength = 1200
pixelSize = 15
stepSize = 0.001098038
# nz: x values
# magnitudes: y values (plot points)
# mag_fits: fitted y values (plot straight line)
# parameters: [source-to-sample, source-to-detector, slope, intercep, error]
nz, magnitudes, mag_fits, parameters = hiholo.calibrateDistance(maxFre, nz, wavelength, pixelSize, stepSize)
print(magnitudes)
print(mag_fits)
print(parameters)

sorted_pairs = sorted(zip(nz, mag_fits))
fit_x = [x for x, _ in sorted_pairs]
fit_y = [y for _, y in sorted_pairs]

plt.figure(figsize=(8, 6))
plt.scatter(nz, magnitudes, color="tab:blue", label="Magnitudes")
plt.plot(fit_x, fit_y, color="tab:red", linewidth=2, label="Linear fit")
plt.xlabel(r"$n_z$", fontsize=16)
plt.ylabel(r"$1/M$", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("calibration_line.png", dpi=300, bbox_inches="tight")
plt.show()
