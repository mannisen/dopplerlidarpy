#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from dopplerlidarpy.equations.turbulence_spectra import lomb_scargle_spectrum
from dopplerlidarpy.utilities import general_utils as gu
import matplotlib.pylab as plt
import numpy as np

x = np.arange(0, 100, .1)
x2 = np.arange(0, 200, .2)

sin1 = np.sin(x)
sin2 = np.sin(x2)

sin3 = sin1 + sin2

noise = np.multiply(np.random.rand(np.size(sin3))*10, sin3)

sin3_w_noise = sin3 + noise

print("current bounds: {},{}, new bounds: {},{}".format(np.min(sin3_w_noise), np.max(sin3_w_noise),
                                                        np.min(sin3), np.max(sin3)))

sin3_w_noise = np.array(gu.normalize_between(sin3_w_noise,
                                    (np.min(sin3_w_noise), np.max(sin3_w_noise)),
                                    (np.min(sin3), np.max(sin3))))
print(np.shape(sin3_w_noise))


x_sparse = []
sin3_sparse = []
noise_sparse = []
for i in np.arange(0, len(x)-1, len(x)/10):
    x_sparse = np.hstack((x_sparse, x[int(i):int(i+len(x)/20)]))
    sin3_sparse = np.hstack((sin3_sparse, sin3_w_noise[int(i):int(i+len(x)/20)]))
    noise_sparse = np.hstack((noise_sparse, sin3_w_noise[int(i):int(i+len(x)/20)]))

sin3_sparse_w_noise = sin3_sparse + noise_sparse

sin3_sparse_w_noise = np.array(gu.normalize_between(sin3_sparse_w_noise,
                                    (np.min(sin3_sparse_w_noise), np.max(sin3_sparse_w_noise)),
                                    (np.min(sin3), np.max(sin3))))

# Calculate FFT
fft_spectrum = np.abs(np.fft.rfft(sin3_w_noise))
fft_freq = np.fft.rfftfreq(sin3_w_noise.shape[-1], .1)

# Interpolate and and calculate FFT
sin3_w_noise_interp = np.interp(x, x_sparse, sin3_sparse_w_noise)
fft_spectrum_interp = np.abs(np.fft.rfft(sin3_w_noise_interp))

# Calculate Lomb-Scargle spectrum, estimation of FFT spectrum when gaps in data
ls_freq, ls_spectrum = lomb_scargle_spectrum(x, sin3_w_noise)
ls_freq_irr, ls_spectrum_irr = lomb_scargle_spectrum(x_sparse, sin3_sparse_w_noise)

fig = plt.figure(figsize=[20, 10])

ax1 = plt.subplot2grid((2, 1), (0, 0))
h11 = plt.plot(x, sin3_w_noise, color="gray",marker=".")
h12 = plt.plot(x_sparse, sin3_sparse_w_noise, marker="o", linestyle="None", markerfacecolor="None", color="red")
h13 = plt.plot(x, sin3_w_noise_interp, marker=".", linestyle="None", color="black")
plt.legend([h11[0], h12[0], h13[0]], ['signal regularly', 'signal sparsely', 'signal sparsely interp.'])
ax1.set_xlabel("time (sec)")
ax1.set_ylabel("amplitude (au)")
ax1.set_xlim([0, 100])
plt.grid(which="both")

ax2 = plt.subplot2grid((2, 1), (1, 0))
h22 = plt.plot(ls_freq, ls_spectrum**2, color="gray")
h23 = plt.plot(ls_freq_irr, ls_spectrum_irr**2, color="red")
h24 = plt.plot(fft_freq, fft_spectrum_interp**2, color="black")
h21 = plt.plot(fft_freq, fft_spectrum**2, color="black", linestyle=":")
plt.legend([h22[0], h23[0], h24[0], h21[0]], ['LS spectrum regularly', 'LS spectrum sparsely',
                                             'fft spectrum sparsely interp.', 'fft spectrum regularly'])
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Power")
ax2.set_xscale("log")
ax2.set_xlim([1e-2, 1e0])
plt.grid(which="both")

#ax2.set_yscale("log")


plt.show()
