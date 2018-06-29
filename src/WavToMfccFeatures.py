# WavToMfccFeatures - Converts a wav file into a comma-separated value list
# of Mel-Frequency Cepstral Coefficients to stdout.

from MfccWavLoader import MfccWavLoader
import sys

if len(sys.argv) < 2:
    print('First param must be a wav file path')
    exit(1)

mfccLoader = MfccWavLoader(sys.argv[1])
mfccLoader.writeFullFeatureArrayToCsvStream(sys.stdout)
