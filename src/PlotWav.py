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

mfccLoader = MfccWavLoader(sys.argv[1], produceFirstDerivative=True, produceSecondDerivative=True)
mfccWav = mfccLoader.generateMfccs().send(None)

plt.subplot(611)
plt.title('Amplitude by time')
plt.margins(0)
plt.plot(mfccWav.samples)

plt.subplot(612)
plt.title('1st derivative amplitude by time')
plt.margins(0)
plt.plot(mfccWav.sampleDeltas)

plt.subplot(613)
plt.title('2nd derivative amplitude by time')
plt.margins(0)
plt.plot(mfccWav.sampleDeltaDeltas)

plt.subplot(614)
plt.title('MFCC by time')
mfccTwoDMatrix = mfccWav.fullFeatureArray[:,:,0].T  # Transpose to get MFCCs on Y axis
plt.matshow(mfccTwoDMatrix, fignum=False, cmap='bwr', aspect='auto')  # cmap='coolwarm' pretty good too

plt.subplot(615)
plt.title('1st derivative MFCC by time')
firstDerivTwoDMatrix = mfccWav.fullFeatureArray[:,:,1].T  # Transpose to get MFCCs on Y axis
plt.matshow(firstDerivTwoDMatrix, fignum=False, cmap='bwr', aspect='auto')  # cmap='coolwarm' pretty good too

plt.subplot(616)
plt.title('2nd derivative MFCC by time')
secondDerivTwoDMatrix = mfccWav.fullFeatureArray[:,:,2].T  # Transpose to get MFCCs on Y axis
plt.matshow(secondDerivTwoDMatrix, fignum=False, cmap='bwr', aspect='auto')  # cmap='coolwarm' pretty good too

plt.show()
