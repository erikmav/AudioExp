# PlotWav - Plots the wave amplitude (time domain) against the MFCCs.

from MfccWavLoader import MfccWavLoader
import matplotlib.pyplot as plt
import numpy
import sys

if len(sys.argv) < 2:
    print('First param must be a wav file path')
    exit(1)

mfccLoader = MfccWavLoader(sys.argv[1])

plt.subplot(211)
plt.plot(mfccLoader.samples)

plt.subplot(212)
twoDMatrix = mfccLoader.fullFeatureArray[:,:,0]
plt.matshow(twoDMatrix, fignum=False, cmap='bwr', aspect='auto')  # 'coolwarm' pretty good too

plt.show()
