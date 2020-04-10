import glob
from MfccWavLoader import MfccWavLoader
import numpy
import os
from SoundTagJsonReader import SoundTagJsonReader

# To ensure that all wavs generate comparable MFCCs, we need to ensure the top end
# of the MFCC bucketing range is consistent. The default MFCC generation takes
# the wav's rateHz / 2. We have 44.1KHz and 48KHz samples so we set the max range
# to half the min, and assert below that we're not loading samples with even lower rates.
wavMinAllowedHz = 44100
mfccMaxRangeHz = wavMinAllowedHz / 2

class InstrumentLoader:
    """
    Wraps loading WAV files referenced in a TaggedSoundData.json and directory structure
    while calculating MFCCs.
    """

    allInstrumentMfccWavs = []
    allInstrumentMfccData = []
    allInstrumentLabels = []
    mfccLenToSamplesMap = {}

    def __init__(self, samplesDirPath):
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
                for mfccWav in mfccLoader.generateMfccs(instrumentTag, allTags):
                    numMfccRows = mfccWav.numMfccRows
                    # print(soundPath, "shape", shape)
                    self.maxMfccRows = max(self.maxMfccRows, numMfccRows)
                    self.minMfccRows = min(self.minMfccRows, numMfccRows)
                    self.minWavHz = min(self.minWavHz, mfccWav.rateHz)

                    self.allInstrumentMfccWavs.append(mfccWav)
                    self.allInstrumentMfccData.append(mfccWav.fullFeatureArray)
                    self.allInstrumentLabels.append(allTags)

                    sampleList = self.mfccLenToSamplesMap.get(numMfccRows)
                    if sampleList is None:
                        sampleList = []
                        self.mfccLenToSamplesMap[numMfccRows] = sampleList
                    sampleList.append(mfccWav)
