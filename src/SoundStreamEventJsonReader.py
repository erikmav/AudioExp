import json
import os

class SoundStreamInstrumentEvent:
    """Data for a single tagged event in a sound stream"""

    def __init__(self, timeSec, instrumentTags):
        self.TimeSec = timeSec
        self.InstrumentTags = instrumentTags

class SoundStream:
    """Data related to one wav file containing a sound sequence for matching"""
    def __init__(self, relativePath, instrumentTag, tags):
        self.RelativePath = relativePath
        self.InstrumentEvents = []

class SoundStreamEventData:
    """Top-level data type stored in SoundStreamEventData.json"""

    def __init__(self):
        self.SoundStreams = []

class SoundStreamEventJsonReader:
    """
    Reads a wav file event declaration format into self.data.
    Data in a JSON form similar to:
    {
        "SoundStreams": [
            {
                "RelativePath": "beatbox_1.wav",
                "InstrumentEvents": [
                    {
                        "TimeSec": 1.0556,
                        "InstrumentTags": [ "snare" ],
                    }
                ]
            },
        ]
    }
    """

    fileName = "SoundStreamEventData.json"

    def __init__(self, folderPath):
        self.folderPath = folderPath
        tagDataPath = os.path.join(folderPath, SoundStreamEventJsonReader.fileName)
        print("Reading tag data from:", tagDataPath)
        f = open(tagDataPath)
        self.data = json.load(f)
