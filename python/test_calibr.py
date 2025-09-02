import hiholo
import mytools

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