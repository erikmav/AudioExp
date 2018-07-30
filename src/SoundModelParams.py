import json

class SoundModelParams:
    """
    Hyperparameters (or perhaps mega-hyperparameters or hyper-hyperparameters)
    related to a trained MFCC model. Including:

    mfccRows: The window size of MFCC rows needed to run the model against.
      The window must be normalized with MfccWavLoader.normalizeMfccArray().

    instrumentsArray: An ordered array of instrument/tag names correcponding
      to the one-hot array generated during loading of instrument samples.
    """

    def __init__(self, mfccRows, instrumentsArray):
        self.mfccRows = mfccRows
        self.instruments = instrumentsArray

    def save(self, filePath):
        with open(filePath, 'w') as j:
            json.dump({ 'mfccRows': self.mfccRows, 'instruments': self.instruments }, j)
