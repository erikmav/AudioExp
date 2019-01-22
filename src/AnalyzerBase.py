class AnalyzerBase:
    """
    Abstract base class for MFCC audio analyzers.
    """

    def name(self):
        """Returns the name of this analyzer for logging purposes"""
        raise NotImplementedError()

    def getWindowInMfccRows(self):
        """Returns the number of MFCC rows this analyzer needs."""
        raise NotImplementedError()

    def analyze(self, mfccRows):
        """
        Given a NumPy array of MFCC rows, returns a NumPy array in the shape (1, numInstruments)
        of confidence values for each possible instrument type, with confidence being in the range [0, 1].
        """
        raise NotImplementedError()
