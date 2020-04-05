# AudioExp
Audio stream parsing experiments using Mel-Frequency Cepstral Coefficients (MFCCs) and machine learning.

## Training
See 'usage' section in [TrainKerasTensorFlowForTaggedSounds.py](./src/TrainKerasTensorFlowForTaggedSounds.py).

## Visualization
Compare waveform in amplitude domain to MFCCs graphically. See 'usage' section in [PlotWav.py](./src/PlotWav.py).

## WAV to MFCC CSV
Convert a wav file to a .csv file containing MFCCs using [WavToMfccFeatures.py](./src/WavToMfccFeatures.py).

## Finding samples in a .wav file
When you have a trained model, use [AnalyzeSoundStream.py](./src/AnalyzeSoundStream.py).
