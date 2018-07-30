import json
import keras.models
import os
from SoundStreamAnalyzer import SoundStreamAnalyzer
import sys

if len(sys.argv) < 4:
    print('Usage:')
    print('  <wav-file-path> <model-file-path> <model-params-json>')
    exit(1)
wavFilePath = sys.argv[1]
modelFilePath = sys.argv[2]
modelParamsPath = sys.argv[3]

trainedModel = keras.models.load_model(modelFilePath)

f = open(modelParamsPath)
modelParams = json.load(f)

analyzer = SoundStreamAnalyzer(wavFilePath, trainedModel, modelParams, 0.8)
analyzer.getMatches()
