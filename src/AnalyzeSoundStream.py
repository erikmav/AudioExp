import json
from InstrumentLoader import InstrumentLoader
import keras.models
from KerasTensorFlowAnalyzer import KerasTensorFlowAnalyzer
import MfccComparisonAnalyzer
import os
from SoundStreamAnalyzer import SoundStreamAnalyzer
import sys

if len(sys.argv) < 5:
    print('Usage:')
    print('  <wav-file-path> <instruments-folder-path> <model-file-path> <model-params-json-path>')
    exit(1)
wavFilePath = sys.argv[1]
instrumentsFolderPath = sys.argv[2]
modelFilePath = sys.argv[3]
modelParamsPath = sys.argv[4]

instruments = InstrumentLoader(instrumentsFolderPath)

trainedModel = keras.models.load_model(modelFilePath)

# See SoundModelParams.py
f = open(modelParamsPath)
modelParams = json.load(f)

analyzers = [ KerasTensorFlowAnalyzer(trainedModel, modelParams) ] + list(MfccComparisonAnalyzer.constructFromInstruments(instruments))

analyzer = SoundStreamAnalyzer(wavFilePath, instruments, analyzers, 0.9)
analyzer.getMatches()
