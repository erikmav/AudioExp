from AnalyzerBase import AnalyzerBase
import math
import numpy
from scipy.spatial.distance import cdist

class MfccComparisonAnalyzer(AnalyzerBase):
    """Analyzer wrapping an MFCC array to directly compare to normalized MFCC-converted inputs."""

    def __init__(self, mfccWav, instrumentNameToResultIndexMap):
        self.mfccWav = mfccWav
        self.instrumentNameToResultIndexMap = instrumentNameToResultIndexMap

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
        normAvg = avg / 2.0  # Normalize from [0,2] to [0,1]
        convNormAvg = 1.0 - normAvg
        dist = convNormAvg
        predictions = numpy.full((1, len(self.instrumentNameToResultIndexMap)), 0.0)  # Predictions shape (1, numInstruments)
        for label in self.mfccWav.tags:
            predictions[0][self.instrumentNameToResultIndexMap[label]] = dist
        return predictions


def constructFromInstruments(instruments):
    """
    Factory Generator method that converts instruments into instances.
    instruments: An InstrumentLoader with all instruments loaded.
    """
    for mfccWav in instruments.allInstrumentMfccWavs:
        yield MfccComparisonAnalyzer(mfccWav, instruments.instrumentNameToResultIndexMap)
