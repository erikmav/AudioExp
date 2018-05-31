# WavToMfccFeatures - Converts a wav file into a comma-separated value list
# of Mel-Frequency Cepstral Coefficients.

from MfccWavLoader import MfccWavLoader
import numpy
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys

if len(sys.argv) < 2:
    print('First param must be a wav file path')
    exit(1)

mfccLoader = MfccWavLoader(sys.argv[1])
mfccLoader.writeFullFeatureArrayToCsvStream(sys.stdout)
