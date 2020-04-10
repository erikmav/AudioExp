from AnalyzerBase import AnalyzerBase
import math
import numpy
from scipy.spatial.distance import cdist

class MfccComparisonAnalyzer(AnalyzerBase):
    """Analyzer wrapping an MFCC array to directly compare to normalized MFCC-converted inputs."""

    def __init__(self, mfccWav, instrumentIndex, numInstruments):
        self.mfccWav = mfccWav
        self.instrumentIndex = instrumentIndex
        self.numInstruments = numInstruments

    def name(self):
        return "MFCComp"

    def getWindowInMfccRows(self):
        return self.mfccWav.numMfccRows

    def analyze(self, mfccRows):
        # mfccRows is in the shape (1, N, 12, 1), slice out the middle 2 dimensions as the
        # MFCC rows.
        mfccRows2D = mfccRows[0, :, :, 0]

        # Similarly our own WAV data has the shape (N, 12, M).
        instrumentMfccRows2D = self.mfccWav.fullFeatureArray[:, :, 0]

        # MFCCs are normalized in [-1,1], this difference results in [-2,2].
        # Check for all diff results to be close to zero
        # Normalize to [0,1] and subtract from 1 to get likelihood.
        diff = instrumentMfccRows2D - mfccRows2D
        absDiff = numpy.abs(diff)
        avg = numpy.average(absDiff)
        normAvg = avg / 2.0
        convNormAvg = 1.0 - normAvg
        #sqrConvNormAvg = convNormAvg * convNormAvg
        dist = convNormAvg
        if dist >= 0.9:
            print("dist:", dist, "min:", numpy.min(absDiff), "max:", numpy.max(absDiff), "avg:", numpy.average(absDiff), "sqrt(avg/2):", math.sqrt(numpy.average(absDiff) / 2))

        # MFCCs are normalzed in [-1,1] so Euclidean distance will be in the range [0, 2].
        # Normalize to [0,1] and subtract from 1 to get likelihood.
        #euclideanDist = cdist(mfccRows2D, instrumentMfccRows2D, 'euclidean')
        #dist = 1.0 - numpy.min(euclideanDist) / 2.0
        #if dist >= 0.8:
        #    print("eucDist:", euclideanDist)
        predictions = numpy.full((1, self.numInstruments), 0)  # Predictions shape (1, numInstruments)
        predictions[0][self.instrumentIndex] = dist
        return predictions


def constructFromInstruments(instruments):
    """
    Factory Generator method that converts instruments into instances.
    instruments: An InstrumentLoader with all instruments loaded.
    """
    numInstruments = len(instruments.allInstrumentMfccWavs)
    for i in range(numInstruments):
        yield MfccComparisonAnalyzer(instruments.allInstrumentMfccWavs[i], i, numInstruments)
