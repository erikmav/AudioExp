import glob
from MfccWavLoader import MfccWavLoader
import numpy
import os
from SoundTagJsonReader import SoundTagJsonReader

class InstrumentLoader:
    """
    Wraps loading WAV files referenced in a TaggedSoundData.json and directory structure
    while calculating MFCCs.
    """

    allInstrumentMfccData = []
    allInstrumentLabels = []
    mfccLenToSamplesMap = {}

    def __init__(self, samplesDirPath, mfccMaxRangeHz):
        soundTagJsonReader = SoundTagJsonReader(samplesDirPath)

        self.maxMfccRows = 0
        self.minMfccRows = 100000000
        self.minWavHz = 10000000
        for soundData in soundTagJsonReader.data["Sounds"]:
            # Add instrument label along with any additional tags into a list for multi-label binarizing.
            instrumentTag = soundData["InstrumentTag"]
            additionalTagsList = soundData["Tags"] or []
            allTags = [ instrumentTag ] + additionalTagsList

            fullGlob = os.path.join(soundTagJsonReader.folderPath, soundData["SoundRelativePath"])
            for soundPath in glob.glob(fullGlob):
                mfccLoader = MfccWavLoader(soundPath, mfccMaxRangeHz)
                for mfccWav in mfccLoader.generateMfccs():
                    mfccLayers = mfccWav.fullFeatureArray
                    shape = numpy.shape(mfccLayers)
                    numMfccRows = shape[0]
                    print(soundPath, "shape", shape)
                    self.maxMfccRows = max(self.maxMfccRows, numMfccRows)
                    self.minMfccRows = min(self.minMfccRows, numMfccRows)
                    self.minWavHz = min(self.minWavHz, mfccWav.rateHz)

                    self.allInstrumentMfccData.append(mfccLayers)
                    self.allInstrumentLabels.append(allTags)

                    sampleList = self.mfccLenToSamplesMap.get(numMfccRows)
                    if sampleList is None:
                        sampleList = []
                        self.mfccLenToSamplesMap[numMfccRows] = sampleList
                    sampleList.append(mfccWav)
