from AnalyzerBase import AnalyzerBase

class KerasTensorFlowAnalyzer(AnalyzerBase):
    """Analyzer wrapping a trained Keras+TensorFlow model"""

    def __init__(self, trainedMfccModel, modelParams):
        self.model = trainedMfccModel
        self.modelParams = modelParams

    def name(self):
        return "KerasTF"

    def getWindowInMfccRows(self):
        print(self.modelParams)
        return self.modelParams["mfccRows"]

    def analyze(self, mfccRows):
        predictions = self.model.predict(mfccRows, verbose=1)  # Predictions shape (1, numInstruments)
        return predictions
