# TrainKerasTensorFlowForTaggedSounds - Loads a set of wav files and their tags,
# analyzes them  using Mel-Frequency Cepstral Coefficients and related derivatives,
# splits the wav set into training and validation sets, and trains a Keras/TensorFlow
# machine learning model to recognize the sounds.

import glob
import json
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
import keras.optimizers
import keras.utils
from MfccWavLoader import MfccWavLoader
from StayAwake import preventComputerFromSleeping
import os
import numpy
import random
import sys

class TaggedSound:
    """Per-sound tag data format stored in the TaggedSoundData.Sounds collection"""
    def __init__(self, soundRelativePath, instrumentTag, tags):
        self.SoundRelativePath = soundRelativePath
        self.InstrumentTag = instrumentTag
        self.Tags = tags

class TaggedSoundData:
    """Top-level data type stored in TaggedSoundData.json"""

    def __init__(self, folderPath):
        self.FolderPath = folderPath
        self.Sounds = []

class SoundTagJsonReader:
    """
    Reads a wav file tag summary format into self.data.
    Data in a JSON form similar to:
    {
        "Sounds": [
            {
                "SoundRelativePath": "snare_sample1.wav",
                "InstrumentTag": "snare",
                "Tags": [ ]
            },
            {
                "SoundRelativePath": "snare_sample2.wav",
                "InstrumentTag": "snare",
                "Tags": [ ]
            },
            {
                "SoundRelativePath": "tom-tom_sample1.wav",
                "InstrumentTag": "tom-tom",
                "Tags": [ "low" ]
            }
        ]
    }
    """

    fileName = "TaggedSoundData.json"

    def __init__(self, folderPath):
        self.folderPath = folderPath
        tagDataPath = os.path.join(folderPath, SoundTagJsonReader.fileName)
        print("Reading tag data from:", tagDataPath)
        f = open(tagDataPath)
        self.data = json.load(f)

if len(sys.argv) < 2:
    print('First param must be a directory containing wav files and ' + SoundTagJsonReader.fileName)
    exit(1)

# Keras / TensorFlow model: A Time-Delay Neural Network in the same vein as those used
# for finding speech phonemes in a stream of Mel-Frequency Cepstral Coefficients
# (along with other information like the first and second derivatives of the MFCCs
# over one or more MFCC window widths). https://en.wikipedia.org/wiki/Time_delay_neural_network
# We treat all of the tagged wav samples as potential phonemes in a large "alphabet."
#
# We model a TDNN as a Convolutional Neural Network using Temporal Pooling.
# https://hal.archives-ouvertes.fr/file/index/docid/287426/filename/A_convolutional_neural_network_approach_for_objective_video_quality_assessment_completefinal_manuscript.pdf
# http://www.cs.toronto.edu/%7Efritz/absps/waibelTDNN.pdf
# http://isl.anthropomatik.kit.edu/cmu-kit/downloads/CP_1991_Review_of_TDNN_(Time-Delay_Neural_Network)_Architectures_for_Speech_Recognition.pdf
# https://www.microsoft.com/en-us/research/wp-content/uploads/2017/08/ms_swbd17-2.pdf
# https://d1ge0kk1l5kms0.cloudfront.net/images/G/01/amazon.jobs/Interspeech_2017_4._CB503635227_.pdf
# https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/TDNN.py
#
# MFCCs and derivatives (M = # of coefficients) are computed at intervals (T) and a set of
# MFCCs are selected into a frame (W = number of MFCC rows covering (W * T) time span).
#
# Per TDNN patterns, all the W * M coefficients in a frame are fully connected to a row of
# perceptrons with a chosen activation function. We use N-neuron rows where the time span
# covers the longest comparison sample (instrument) we have; hence N = ((ceil(maxSampleSpan / T) - W + 1).
#
# The resulting N rows of perceptrons are themselves framed in sets of V rows and fully connected
# each to a row of I perceptrons where I is the number of different "instruments" we are trying
# to detect. Number of rows R = N - V + 1
#
# The resulting R rows are then selected into I columns with each column fully connected to an
# output neuron corresponding to the probabilty of that instrument.
#
# |M|M|                |
# |F|F|                | 
# |C|C|                M coefficients per row
# |C|C|                |
# |s|s|                |
# |1|2|3|4|5|6|...     |
# ---W---
#   ---W---
#
#  |       floor(M / 2) fully connected layer, one per W.
#  V |
#    V
# +-+-+-+-+   +-+-+
# | | | | |   | | |
# | | | | |...| | |        N rows of meurons
# | | | | |   | | |
# +-+-+-+-+   +-+-+
# ---V---
#   ---V---
#
#   |
#   V
#
# +-+-+-+-+   +-+         R rows of neurons
# | | | | |   | |   ->   Instrument1 classification neuron
# | | | | |...| |   ->   Instrument2 classification neuron
# +-+-+-+-+   +-+        ...
#
# We vary the following parameters to find the best accuracy.
# - MFCC computation time intervals: 5ms, 10ms, 20ms, 25ms (matching various of the intervals in papers above)
# - MFCC intervals in a sliding window presented to the neural network: 3, 4, 5
#   (i.e. 15ms up to 125ms when multiplied by the time intervals).
# - Training batch size
# - Number of epochs

soundTagJsonReader = SoundTagJsonReader(sys.argv[1])

# We split the data set into training and test sets.
# Since we don't know until we evaluate any globs in the JSON how many
# sample files we have, we roll the dice as we load each sound.
trainingProbNumerator = 4
trainingProbDenom = 5

# We need the number of instruments for various calculations.
instrumentIndexMap = { }
currentIndex = 0
instrumentMfccData = []
instrumentLabelIndexes = []
testInstrumentMfccData = []
testInstrumentLabelIndexes = []
maxMfccRows = 0
minMfccRows = 100000000
for soundData in soundTagJsonReader.data["Sounds"]:
    instrumentTag = soundData["InstrumentTag"]
    label = instrumentIndexMap.get(instrumentTag)
    if label is None:
        label = currentIndex
        instrumentIndexMap[instrumentTag] = currentIndex
        currentIndex += 1

    fullGlob = os.path.join(soundTagJsonReader.folderPath, soundData["SoundRelativePath"])
    for soundPath in glob.glob(fullGlob):
        mfccLoader = MfccWavLoader(soundPath)
        mfccRows = mfccLoader.fullFeatureArray
        shape = numpy.shape(mfccRows)
        numMfccRows = shape[0]
        print(soundPath, "shape", shape)
        maxMfccRows = max(maxMfccRows, numMfccRows)
        minMfccRows = min(minMfccRows, numMfccRows)

        dieRoll = random.randint(1, trainingProbDenom)
        if dieRoll <= trainingProbNumerator:
            instrumentMfccData.append(mfccRows)
            instrumentLabelIndexes.append(label)
        else:
            testInstrumentMfccData.append(mfccRows)
            testInstrumentLabelIndexes.append(label)

print("Max, min MFCC rows across all instruments: ", maxMfccRows, minMfccRows)

# Zero-pad all sounds to the max number of rows, and expand with a 3rd dimension.
# TODO: Or do we create multiple TDNNs trained at each row length?
def zeroPadTo3d(mfccRows):
    shape = numpy.shape(mfccRows)
    numMfccRows = shape[0]
    if (numMfccRows < maxMfccRows):
        mfccRows = numpy.append(mfccRows, numpy.zeros(((maxMfccRows - numMfccRows), shape[1])), axis=0)
    return mfccRows[..., numpy.newaxis]  # Makes NxM into NxMx1
for i in range(len(instrumentMfccData)):
    instrumentMfccData[i] = zeroPadTo3d(instrumentMfccData[i])
for i in range(len(testInstrumentMfccData)):
    testInstrumentMfccData[i] = zeroPadTo3d(testInstrumentMfccData[i])

numInstruments = len(instrumentIndexMap)
print("Num instruments: ", numInstruments)

# Reformat the resulting lists of training and test data into a 4D tensor
# required by the Conv2D Keras layers. This is "channels_last" format,
# (batch, height, width, channels). channels=1, width is the number of
# MFCC columns, height is the number of rows, batch is the total set of
# training or test.
mfccTensors = numpy.stack(instrumentMfccData)
print("Training tensor shape:", numpy.shape(mfccTensors))
testMfccTensors = numpy.stack(testInstrumentMfccData)
print("Testing tensor shape:", numpy.shape(testMfccTensors))

oneHotLabelsByInstrumentOrdinal = keras.utils.to_categorical(instrumentLabelIndexes, numInstruments)
testOneHotLabelsByInstrumentOrdinal = keras.utils.to_categorical(testInstrumentLabelIndexes, numInstruments)

# For the first convolutional layer, the number of convolutional filters
# that are trained to find patterns amongst the input MFCCs.
# TODO: Experiment with this value - hence an array
numConv1FiltersValues = [ numInstruments, numInstruments * 2, numInstruments * 4 ]

# For the first convolutional layer, the size of the kernel that implies the size of the filters.
# TODO: Experiment with this value - hence an array. Some entries are non-square to experiment with
# wider spans across MFCC ranges or wider spans across time.
conv1KernelSizeValues = [ 1, 2, 3, 4, 5, 6, 7, (1,2), (2,1), (2,3), (3,2), (4,3), (3,4), (1,3), (3,1), (1, MfccWavLoader.numColumns), (2, MfccWavLoader.numColumns), (3, MfccWavLoader.numColumns) ]

# For the second convolutional layer, the number of convolutional filters
# that are trained to find patterns amongst the results of the first conv layer.
# TODO: Experiment with this value - hence an array
numConv2FiltersValues = [ numInstruments, numInstruments * 2, numInstruments * 4 ]

# For the second convolutional layer, the size of the kernel that implies the size of the filters.
# TODO: Experiment with this value - hence an array. Some entries are non-square to experiment with
# wider spans across MFCC ranges or wider spans across time.
conv2KernelSizeValues = [ 1, 2, 3, 4, 5 ]

# TODO: Experiment with this value - hence an array
numFullyConnectedPerceptronsLastLayerValues = [ numInstruments * 2, numInstruments * 3, numInstruments * 4, numInstruments * 8, numInstruments * 16 ]


def TrainAndValidateModel(numConv1Filters, conv1KernelSize, numConv2Filters, conv2KernelSize, numFullyConnectedPerceptronsLastLayer, batchSize = 16):
    print("TrainAndValidateModel:")
    print("  numConv1Filters:", numConv1Filters)
    print("  conv1KernelSize:", conv1KernelSize)
    print("  numConv2Filters:", numConv2Filters)
    print("  conv2KernelSize:", conv1KernelSize)
    print("  numFullyConnectedPerceptronsLastLayer:", numFullyConnectedPerceptronsLastLayer)

    model = Sequential([
        # Layer 1: W rows of MFCC, MFCC derivative, MFCC double derivative, log-filterbank-energy
        # row information (51 columns) leading to the first convolutional layer.
        Conv2D(numConv1Filters, conv1KernelSize, kernel_initializer='TruncatedNormal', activation='relu', input_shape=(maxMfccRows, MfccWavLoader.numColumns, 1)),
        
        # Layer 2: Convolution over results from conv layer 1. This provides an integration over a wider time period,
        # using the features extracted from the first layer.
        Conv2D(numConv2Filters, conv2KernelSize, kernel_initializer='TruncatedNormal', activation='relu'),

        # Reduce dimensionality before connecting to fully connected layers.
        Flatten(),

        # Layer 3: Fully connected layer with ReLU activation.
        Dense(numFullyConnectedPerceptronsLastLayer, activation='relu'),

        # Outputs: SoftMax activation to get probabilities by instrument.
        Dense(numInstruments, activation='softmax')
    ])

    print(model.summary())

    # Compile for categorization.
    model.compile(
        optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 1e-6, nesterov = True),
        loss = 'categorical_crossentropy',
        metrics = [ 'accuracy' ])

    # TODO: Experiment with epochs (or move to dynamic epochs by epsilon gain)
    # TODO: Experiment with batch size
    model.fit(mfccTensors, oneHotLabelsByInstrumentOrdinal, epochs=10, batch_size=batchSize)

    score = model.evaluate(testMfccTensors, testOneHotLabelsByInstrumentOrdinal, batch_size=batchSize)
    print("Score:", model.metrics_names, score)

    return {
        model.metrics_names[0]: score[0],
        model.metrics_names[1]: score[1],
        "numConv1Filters": numConv1Filters,
        "conv1KernelSize": conv1KernelSize,
        "numConv2Filters": numConv2Filters,
        "conv1KernelSize": conv1KernelSize,
        "numFullyConnectedPerceptronsLastLayer": numFullyConnectedPerceptronsLastLayer
     }

results = []

preventComputerFromSleeping(True)
try:
    for numConv1Filters in numConv1FiltersValues:
        for conv1KernelSize in conv1KernelSizeValues:
            for numConv2Filters in numConv2FiltersValues:
                for conv2KernelSize in conv2KernelSizeValues:
                    for numFullyConnectedPerceptronsLastLayer in numFullyConnectedPerceptronsLastLayerValues:
                        results.append(TrainAndValidateModel(numConv1Filters, conv1KernelSize, numConv2Filters, conv2KernelSize, numFullyConnectedPerceptronsLastLayer))
finally:
    preventComputerFromSleeping(False)

resultMinLoss = None
minLoss = 100000
resultMaxAccuracy = None
maxAccuracy = 0
for result in results:
    print(result)
    loss = result["loss"]
    accuracy = result["acc"]
    if loss < minLoss:
        minLoss = loss
        resultMinLoss = result
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        resultMaxAccuracy = result

print("Result with min loss:", resultMinLoss)
print("Result with max accuracy:", resultMaxAccuracy)
