# PlotWav - Plots the wave amplitude (time domain) against the MFCCs.
#
# Usage:
#   python PlotWav.py <wav-file-path>

from MfccWavLoader import MfccWavLoader
import matplotlib.pyplot as plt
import numpy
import sys

if len(sys.argv) < 2:
    print('First param must be a wav file path')
    exit(1)

mfccLoader = MfccWavLoader(sys.argv[1])
mfccWav = mfccLoader.generateMfccs().send(None)

plt.subplot(211)
plt.margins(0)
plt.plot(mfccWav.samples)

plt.subplot(212)
twoDMatrix = mfccWav.fullFeatureArray[:,:,0].T  # Transpose to get MFCCs on Y axis
plt.matshow(twoDMatrix, fignum=False, cmap='bwr', aspect='auto')  # cmap='coolwarm' pretty good too

plt.show()
