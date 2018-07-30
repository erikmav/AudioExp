import json
import os

class TaggedSound:
    """Per-sound tag data format stored in the TaggedSoundData.Sounds collection"""
    def __init__(self, soundRelativePath, instrumentTag, tags):
        self.SoundRelativePath = soundRelativePath
        self.InstrumentTag = instrumentTag
        self.Tags = tags

class TaggedSoundData:
    """Top-level data type stored in TaggedSoundData.json"""

    def __init__(self):
        self.Sounds = []

class SoundTagJsonReader:
    """
    Reads a wav file tag summary format into self.data.
    Data in a JSON form similar to:
    {
        "Sounds": [
            {
                "SoundRelativePath": "snare_sample1.wav",
                "InstrumentTag": "snare",
                "Tags": [ ]
            },
            {
                "SoundRelativePath": "snare_sample2.wav",
                "InstrumentTag": "snare",
                "Tags": [ ]
            },
            {
                "SoundRelativePath": "tom-tom_sample1.wav",
                "InstrumentTag": "tom-tom",
                "Tags": [ "low" ]
            }
        ]
    }
    """

    fileName = "TaggedSoundData.json"

    def __init__(self, folderPath):
        self.folderPath = folderPath
        tagDataPath = os.path.join(folderPath, SoundTagJsonReader.fileName)
        print("Reading tag data from:", tagDataPath)
        f = open(tagDataPath)
        self.data = json.load(f)
