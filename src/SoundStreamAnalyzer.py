from InstrumentLoader import InstrumentLoader
from MfccWavLoader import MfccWavLoader, normalizeMfccArray, windowStepLengthSec
import numpy
from SoundStreamEventJsonReader import SoundStreamEventJsonReader
from StayAwake import preventComputerFromSleeping

class SoundStreamAnalyzer:
    """
    Opens a wav file or microphone stream and searches for instrument events utilizing
    available analysis tools including distance comparisons and trained models.
    Implements a Generator (iterator, enumerator) pattern for returning detected results.
    """

    def __init__(self, wavFilePath, instruments, analyzers, minDetectionCertainty):
        """
        wavFilePath: The path to the input file to analyze.
        instruments: An InstrumentLoader with all instruments loaded.
        analyzers: An array of AnalyzerBase derived class instances.
        minDetectionCertainty: A value in the range [0, 1] for the minimum certainty
          of a match required from the detecetor.
        """
        self.instruments = instruments
        self.minDetectionCertainty = minDetectionCertainty

        self.analyzers = analyzers
        self.analyzersByLen = { }
        for analyzer in analyzers:
            len = analyzer.getWindowInMfccRows()
            if len not in self.analyzersByLen:
                self.analyzersByLen[len] = []
            a = self.analyzersByLen[len]
            a.append(analyzer)

        # TODO: Allow None value for wav path to open the primary microphone instead.
        # Implies refactoring to allow streaming windows of max needed length into various
        # detection algos.
        mfccLoader = MfccWavLoader(wavFilePath)
        self.mfccs = mfccLoader.generateMfccs('analysis-stream', []).send(None).fullFeatureArray
        shape = numpy.shape(self.mfccs)
        print("Loaded", wavFilePath, "producing", shape[0], "rows; full shape", shape)

    def getMatches(self):
        return self.getMatchesForSamples(self.mfccs)

    def getMatchesForSamples(self, mfccs):
        preventComputerFromSleeping(True)
        try:
            shape = numpy.shape(mfccs)
            labels = self.instruments.orderedResultInstrumentLabels

            minAnalyzerRequiredMfccRows = min(self.analyzersByLen.keys())
            print("Min analyzer window length in MFCC rows", minAnalyzerRequiredMfccRows)

            numWindows = shape[0] - minAnalyzerRequiredMfccRows + 1
            print("numWindows", numWindows, shape)

            for currentRow in range(numWindows):
                if int(currentRow % 10) == 0:
                    print("Row", currentRow)
                tagsFound = {}

                for numMfccRows, analyzers in self.analyzersByLen.items():
                    if (currentRow + numMfccRows) >= shape[0]:
                        # Analyzers need more samples than we have remaining in the tail.
                        continue

                    analysisArray = mfccs[currentRow : currentRow + numMfccRows, :]
                    normalizeMfccArray(analysisArray)

                    # Input to analyzers needs to be an array of samples to match (1), each with
                    # 3 dimensions (rows, columns, layers=1).
                    analysisArray = analysisArray[numpy.newaxis, ...]

                    for analyzer in analyzers:
                        # Predictions shape (1, numInstruments)
                        predictions = analyzer.analyze(analysisArray)
                        for i in range(predictions.shape[1]):
                            if predictions[0][i] >= self.minDetectionCertainty:
                                tagsFound[labels[i]] = True
                    
                if len(tagsFound) > 0:
                    print("Found", tagsFound.keys(), "at offset", windowStepLengthSec * currentRow, " sec (MFCC row", currentRow, ")")

        finally:
            preventComputerFromSleeping(False)
