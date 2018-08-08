# TrainKerasTensorFlowForTaggedSounds - Loads a set of wav files and their tags,
# analyzes them using Mel-Frequency Cepstral Coefficients and related derivatives,
# splits the wav set into training and validation sets, and trains a Keras/TensorFlow
# machine learning model to recognize the sounds.
#
# The resulting model can be used to scan a stream of similarly-calculated MFCC
# rows from a longer or continuous wav stream, performing recognition within
# each overlapped window of MFCCs, predicting whether each instrument is at the
# start of that window.

from datetime import datetime
import glob
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
import keras.optimizers
import keras.utils
from MfccWavLoader import MfccWavLoader
import os
import numpy
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from SoundModelParams import SoundModelParams
from SoundTagJsonReader import SoundTagJsonReader
from StayAwake import preventComputerFromSleeping
import sys

if len(sys.argv) < 2:
    print('First param must be a directory containing wav files and ' + SoundTagJsonReader.fileName)
    exit(1)
soundTagFileName = sys.argv[1]

# Avoid loss of experimental results during long runs
resultFileName = "results.out"
if len(sys.argv) >= 3:
    resultFile = sys.argv[3]
resultFile = open(resultFileName, "w")

# Tee specific outputs to both the result file and stdout for safekeeping.
def Log(*objects):
    print(*objects)
    print(*objects, file=resultFile)

startDateTime = datetime.now()
Log("Start:", startDateTime)

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
#  |       conv1 layer of varying kernel shapes and filter counts
#  V |
#    V
# +-+-+-+-+   +-+-+
# | | | | |   | | |
# | | | | |...| | |        Reduced sets of rows, one version per conv filter
# | | | | |   | | |
# +-+-+-+-+   +-+-+
# ---V---
#   ---V---
#
#   |     conv2 layer of varying kernel shapes and filter counts
#   V
#
# +-+-+-+-+   +-+         
# | | | | |   | |
# | | | | |...| |
# +-+-+-+-+   +-+
# ---------------
#       |
#       V
#
#    ********           Fully connected layer
#       |
#       V
#
#      ABC              Per-instrument classification neurons (outputs)
#
# We vary the following parameters to find the best accuracy.
# - MFCC computation time intervals: 5ms, 10ms, 20ms, 25ms (matching various of the intervals in papers above)
# - MFCC intervals in a sliding window presented to the neural network: 3, 4, 5
#   (i.e. 15ms up to 125ms when multiplied by the time intervals).
# - Training batch size
# - Number of epochs

soundTagJsonReader = SoundTagJsonReader(soundTagFileName)

# To ensure that all wavs generate comparable MFCCs, we need to ensure the top end
# of the MFCC bucketing range is consistent. The default MFCC generation takes
# the wav's rateHz / 2. We have 44.1KHz and 48KHz samples so we set the max range
# to half the min, and assert below that we're not loading samples with even lower rates.
wavMinAllowedHz = 44100
mfccMaxRangeHz = wavMinAllowedHz / 2

# We need the number of instruments for various calculations.
currentIndex = 0
allInstrumentMfccData = []
allInstrumentLabels = []
maxMfccRows = 0
minMfccRows = 100000000
minWavHz = 10000000
for soundData in soundTagJsonReader.data["Sounds"]:
    instrumentTag = soundData["InstrumentTag"]

    fullGlob = os.path.join(soundTagJsonReader.folderPath, soundData["SoundRelativePath"])
    for soundPath in glob.glob(fullGlob):
        mfccLoader = MfccWavLoader(soundPath, mfccMaxRangeHz)
        mfccLayers = mfccLoader.fullFeatureArray
        shape = numpy.shape(mfccLayers)
        numMfccRows = shape[0]
        print(soundPath, "shape", shape)
        maxMfccRows = max(maxMfccRows, numMfccRows)
        minMfccRows = min(minMfccRows, numMfccRows)
        minWavHz = min(minWavHz, mfccLoader.rateHz)

        allInstrumentMfccData.append(mfccLayers)
        allInstrumentLabels.append(instrumentTag)

Log("Max, min MFCC rows across all instruments: ", maxMfccRows, minMfccRows)

if minWavHz < wavMinAllowedHz:
    print("ERROR: One or more wav files found with rate in Hz less than configured minimum. Min found:", minWavHz, " allowed min:", wavMinAllowedHz)
    exit(1)

# Zero-pad all sounds to the max number of rows. Assumes layot of (rows, cols, channels) where channels
# can be just the MFCCs (dimension height of 1) or the MFCCs plus its derivatives (dimension height of 2 or more).
# TODO: Or do we create multiple TDNNs trained at each row length?
numMfccLayers = 1
numMfccColumns = 12
def zeroPad(mfccLayers):
    shape = numpy.shape(mfccLayers)
    numMfccRows = shape[0]
    numMfccColumns = shape[1]
    numMfccLayers = shape[2]
    if (numMfccRows < maxMfccRows):
        mfccLayers = numpy.append(mfccLayers, numpy.zeros(((maxMfccRows - numMfccRows), numMfccColumns, numMfccLayers)), axis=0)
    return (mfccLayers, numMfccLayers, numMfccColumns)
for i in range(len(allInstrumentMfccData)):
    allInstrumentMfccData[i], numMfccLayers, numMfccColumns = zeroPad(allInstrumentMfccData[i])
print("numMfccLayers:", numMfccLayers)

# Binarize the labels (convert to 1-hot arrays from text labels/tags).
# Text labels for each array position in the classes_ list on the binarizer.
labelBinarizer = LabelBinarizer()
oneHotLabels = labelBinarizer.fit_transform(allInstrumentLabels)
numInstruments = oneHotLabels.shape[1]
Log("Num instruments:", numInstruments, ":", labelBinarizer.classes_)
soundModelParams = SoundModelParams(maxMfccRows, labelBinarizer.classes_.tolist())

# Partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing.
(instrumentMfccData, testInstrumentMfccData, instrumentOneHotLabels, testInstrumentOneHotLabels) = train_test_split(allInstrumentMfccData,
	oneHotLabels, test_size=0.2, random_state=42)

# Reformat the resulting lists of training and test data into a 4D tensor
# required by the Conv2D Keras layers. This is "channels_last" format,
# (batch, height, width, channels). channels is the number of MFCC layers (just the coefficients
# or the coefficients and their derivatives), width is the number of
# MFCC columns, height is the number of rows, batch is the total set of
# training or test.
mfccTensors = numpy.stack(instrumentMfccData)
print("Training tensor shape:", numpy.shape(mfccTensors))
testMfccTensors = numpy.stack(testInstrumentMfccData)
print("Testing tensor shape:", numpy.shape(testMfccTensors))

# For the first convolutional layer, the number of convolutional filters
# that are trained to find patterns amongst the input MFCCs.
# Experimentation shows that high numbers of these produce the best results.
numConv1FiltersValues = [ numInstruments * 32 ]

# For the first convolutional layer, the size of the kernel that implies the size of the filters.
# Other values are OK but 5x5 seems pretty good based on experimentation.
conv1KernelSizeValues = [ 5 ]
#conv1KernelSizeValues = [ 3, 5, 7, (3,5), (5,3), (2, numMfccColumns), (3, numMfccColumns), (4, numMfccColumns) ]

# For the first convolutional layer, the size of the kernel that implies the size of the filters.
# Other values are valid but 5x5 seems pretty good based on experimentation.
conv1KernelSizeValues = [ 5 ]
#conv1KernelSizeValues = [ 3, 5, 7, (3,5), (5,3), (2, numMfccColumns), (3, numMfccColumns), (4, numMfccColumns) ]

# For the second convolutional layer, the number of convolutional filters
# that are trained to find patterns amongst the results of the first conv layer.
# A tip at https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
# recommends more filters here than in the conv1 layer.
# Experimentation however showed good results with a higher number of conv1 filters and about half as many for conv2.
numConv2FiltersValues = [ numInstruments * 16 ]

# For the second convolutional layer, the size of the kernel that implies the size of the filters.
# Experimentation showed 5x5 and 3x6 (3 rows by whole width) having good results. We keep 5x5 for now.
conv2KernelSizeValues = [ 5 ]
#conv2KernelSizeValues = [ 3, 5, (3,5), (5,3), (3,6), (5,6), (7,6) ]

# Other values can be more optimal but setting this value based on experimentation.
numFullyConnectedPerceptronsLastLayerValues = [ numInstruments * 16 ]
#numFullyConnectedPerceptronsLastLayerValues = [ numInstruments * 2, numInstruments * 3, numInstruments * 4, numInstruments * 8, numInstruments * 16, numInstruments * 32, numInstruments * 64, numInstruments * 128 ]

# Settings here based on experiments (see results\ directory).
conv1DropoutValues = [ 0 ]
conv2DropoutValues = [ 0.25 ]
fullyConnectedDropoutValues = [ 0.5 ] 

# Saves the model to the file path (if specified). Returns a summary result.
def TrainAndValidateModel(
    numConv1Filters,
    conv1KernelSize,
    numConv2Filters,
    conv2KernelSize,
    numFullyConnectedPerceptronsLastLayer,
    batchSize = 16,
    epochs = 32,
    conv1Dropout = 0,
    conv2Dropout = 0,
    fullyConnectedDropout = 0,
    modelSavePath = None):
    
    print("TrainAndValidateModel:")
    print("  numConv1Filters:", numConv1Filters)
    print("  conv1KernelSize:", conv1KernelSize)
    print("  numConv2Filters:", numConv2Filters)
    print("  conv2KernelSize:", conv1KernelSize)
    print("  numFullyConnectedPerceptronsLastLayer:", numFullyConnectedPerceptronsLastLayer)
    print("  conv1Dropout:", conv1Dropout)
    print("  conv2Dropout:", conv2Dropout)
    print("  fullyConnectedDropout:", fullyConnectedDropout)

    model = Sequential([
        # Layer 1: Inputs of MFCCs leading to the first convolutional layer.
        Conv2D(numConv1Filters, conv1KernelSize, kernel_initializer='TruncatedNormal', activation='relu', input_shape=(maxMfccRows, numMfccColumns, numMfccLayers), padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(conv1Dropout),

        # Layer 2: Convolution over results from conv layer 1. This provides an integration over a wider time period,
        # using the features extracted from the first layer.
        Conv2D(numConv2Filters, conv2KernelSize, kernel_initializer='TruncatedNormal', activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(conv2Dropout),

        # Reduce dimensionality before connecting to fully connected layers.
        Flatten(),

        # Layer 3: Fully connected layer with ReLU activation.
        Dense(numFullyConnectedPerceptronsLastLayer, activation='relu'),
        Dropout(fullyConnectedDropout),

        # Outputs: SoftMax activation to get probabilities by instrument.
        Dense(numInstruments, activation='softmax')
    ])

    print(model.summary())

    # Compile for categorization.
    model.compile(
        optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 1e-6, nesterov = True),
        loss = 'categorical_crossentropy',
        metrics = [ 'accuracy' ])

    history = model.fit(mfccTensors, instrumentOneHotLabels, epochs=epochs, batch_size=batchSize)

    score = model.evaluate(testMfccTensors, testInstrumentOneHotLabels, batch_size=batchSize)
    print("Score:", model.metrics_names, score)

    result = {
        "training_" + model.metrics_names[0]: history.history[model.metrics_names[0]][epochs - 1],
        "training_" + model.metrics_names[1]: history.history[model.metrics_names[1]][epochs - 1],
        "testdata_" + model.metrics_names[0]: score[0],
        "testdata_" + model.metrics_names[1]: score[1],
        "batchSize": batchSize,
        "epochs": epochs,
        "maxpoolsize": (2, 2),
        "numConv1Filters": numConv1Filters,
        "conv1KernelSize": conv1KernelSize,
        "numConv2Filters": numConv2Filters,
        "conv2KernelSize": conv2KernelSize,
        "numFullyConnectedPerceptronsLastLayer": numFullyConnectedPerceptronsLastLayer,
        "conv1Dropout": conv1Dropout,
        "conv2Dropout": conv2Dropout,
        "fullyConnectedDropout": fullyConnectedDropout,
        "numInstruments": numInstruments,
        "numTrainingSamples": len(instrumentMfccData),
        "numTestSamples": len(testInstrumentMfccData),
        "modelSavePath": modelSavePath
    }

    if not modelSavePath is None:
        model.save(modelSavePath)
        soundModelParams.save(modelSavePath + ".params.json")

    # Memory usage grows without bound if we don't delete as we go.
    del model

    return result

results = []

saveModelToPath = "./Model.h5"

preventComputerFromSleeping(True)
try:
    for numConv1Filters in numConv1FiltersValues:
        for conv1KernelSize in conv1KernelSizeValues:
            for numConv2Filters in numConv2FiltersValues:
                for conv2KernelSize in conv2KernelSizeValues:
                    for numFullyConnectedPerceptronsLastLayer in numFullyConnectedPerceptronsLastLayerValues:
                        for conv1Dropout in conv1DropoutValues:
                            for conv2Dropout in conv2DropoutValues:
                                for fullyConnectedDropout in fullyConnectedDropoutValues:
                                    result = TrainAndValidateModel(numConv1Filters, conv1KernelSize, numConv2Filters, conv2KernelSize, numFullyConnectedPerceptronsLastLayer, conv1Dropout=conv1Dropout, conv2Dropout=conv2Dropout, fullyConnectedDropout=fullyConnectedDropout, modelSavePath=saveModelToPath)
                                    Log(result)
                                    results.append(result)
finally:
    preventComputerFromSleeping(False)

resultMinLoss = None
minLoss = 100000
resultMaxAccuracy = None
maxAccuracy = 0
for result in results:
    print(result)
    loss = result["testdata_loss"]
    accuracy = result["testdata_acc"]
    if loss < minLoss:
        minLoss = loss
        resultMinLoss = result
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        resultMaxAccuracy = result

Log("Result with min loss:", resultMinLoss)
Log("Result with max accuracy:", resultMaxAccuracy)

endDateTime = datetime.now()
Log("Started:", startDateTime, "; ended:", endDateTime)
Log("Elapsed:", endDateTime - startDateTime)
