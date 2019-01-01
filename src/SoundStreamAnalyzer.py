from InstrumentLoader import InstrumentLoader
from MfccWavLoader import MfccWavLoader, normalizeMfccArray
import numpy
from SoundModelParams import SoundModelParams
from SoundStreamEventJsonReader import SoundStreamEventJsonReader

class SoundStreamAnalyzer:
    """
    Opens a wav file or microphone stream and searches for instrument events utilizing
    available analysis tools including distance comparisons and trained models.
    Implements a Generator (iterator, enumerator) pattern for returning detected results.
    """

    def __init__(self, wavFilePath, instruments, trainedMfccModel, modelParams, minDetectionCertainty):
        """
        wavFilePath: The path to the input file to analyze.
        instruments: An InstrumentLoader with all instruments loaded.
        trainedMfccModel: Instance of a Keras trained model.
        modelParams: An instance of SoundModelParams.
        minDetectionCertainty: A value in the range [0, 1] for the minimum certainty
          of a match required from the detecetor.
        """
        self.instruments = instruments
        self.model = trainedMfccModel
        self.modelParams = modelParams
        self.minDetectionCertainty = minDetectionCertainty

        # TODO: Allow None value for wav path to open the primary microphone instead.
        # Implies refactoring to allow streaming windows of max needed length into various
        # detection algos.
        mfccLoader = MfccWavLoader(wavFilePath)
        self.mfccs = mfccLoader.generateMfccs().send().fullFeatureArray
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
