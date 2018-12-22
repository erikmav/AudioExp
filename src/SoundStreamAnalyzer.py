from MfccWavLoader import MfccWavLoader, normalizeMfccArray
import numpy
from SoundModelParams import SoundModelParams
from SoundStreamEventJsonReader import SoundStreamEventJsonReader

class SoundStreamAnalyzer:
    """
    Opens a wav file or microphone stream and runs a trained model searching
    for instrument events. Implements a Generator (iterator, enumerator) pattern
    for returning detected results.
    """

    def __init__(self, wavFilePath, trainedMfccModel, modelParams, minDetectionCertainty):
        """
        modelParams: An instance of SoundModelParams.
        minDetectionCertainty: A value in the range [0, 1] for the minimum certainty
          of a match required from the detecetor.
        """
        self.model = trainedMfccModel
        self.modelParams = modelParams
        self.minDetectionCertainty = minDetectionCertainty

        self.mfccs = MfccWavLoader(wavFilePath).fullFeatureArray
        shape = numpy.shape(self.mfccs)
        print("Loaded", wavFilePath, "producing", shape[0], "rows; full shape", shape)

    def getMatches(self):
        return self.getMatchesForSamples(self.mfccs)

    def getMatchesForSamples(self, mfccs):
        shape = numpy.shape(mfccs)
        windowRows = self.modelParams["mfccRows"]
        labels = self.modelParams["instruments"]
        print("Window length in MFCC rows", windowRows)
        numWindows = shape[0] - windowRows + 1
        print("numWindows", numWindows, shape)
        currentRow = 0
        while currentRow < numWindows:
            analysisArray = mfccs[currentRow : currentRow + windowRows, :]
            normalizeMfccArray(analysisArray)

            # Input to model needs to be an array of samples to match (1), each with
            # 3 dimensions (rows, columns, layers=1).
            analysisArray = analysisArray[numpy.newaxis, ...]

            predictions = self.model.predict(analysisArray, verbose=1)  # Predictions shape (1, numInstruments)
            for i in range(predictions.shape[1]):
                if predictions[0][i] >= self.minDetectionCertainty:
                    print("Found", labels[i], "at MFCC row", currentRow)

            currentRow += 1
