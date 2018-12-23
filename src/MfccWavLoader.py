import numpy
from os import path
from python_speech_features import delta, logfbank, mfcc
import re
import scipy.io.wavfile as wav
import sys

# python_speech_features defaults unless we find a different approach.
frameWindowSec = 0.025  # 25 ms
windowStepLengthSec = 0.01  # 10 ms

class MfccWav:
    """
    Wav data and related Mel-Frequency Cepstral Coefficients and related numbers,
    serving the resulting NumPy arrays for use in matching or training.
    """

    logFbankHeader = 'logFbank0,logFbank1,logFbank2,logFbank3,logFbank4,logFbank5,logFbank6,logFbank7,logFbank8,logFbank9,logFbank10,logFbank11'
    mfccHeader = 'mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,mfcc13'
    mfccDerivativeHeader = 'mfccd2,mfccd3,mfccd4,mfccd5,mfccd6,mfccd7,mfccd8,mfccd9,mfccd10,mfccd11,mfccd12,mfccd13'
    mfcc2ndDerivativeHeader = 'mfcc2d2,mfcc2d3,mfcc2d4,mfcc2d5,mfcc2d6,mfcc2d7,mfcc2d8,mfcc2d9,mfcc2d10,mfcc2d11,mfcc2d12,mfcc2d13'

    def __init__(self, wavPath, samples, rateHz, mfccMaxRangeHz, produceLogFbank, produceFirstDerivative, produceSecondDerivative):
        self.wavPath = wavPath
        self.rateHz = rateHz
        self.samples = samples
        self.producedLogFbank = produceLogFbank

        # TODO: Look for python_speech_features package >0.6 containing
        # https://github.com/jameslyons/python_speech_features/pull/76 and
        # https://github.com/jameslyons/python_speech_features/pull/77
        # Once that is available, remove this nfft variable and calculate_nfft()
        # and let the default None value to mfcc() use the same code in python_speech_features.
        nfft = calculate_nfft(rateHz, frameWindowSec)

        # Calculate the MFCC features. https://github.com/jameslyons/python_speech_features#mfcc-features
        # We keep many defaults: 13 cepstral coefficients calculated, 26 filters in the MFCC
        # filterbank, 0 Hz low frequency, rateHz/2 high frequency, 0.97 pre-emphasis filter,
        # 22 lifter on final cepstral coefficients.
        #
        # We avoid the appendEnergy parameter to ensure our use of convolutional filters over the 2D array
        # of MFCCs across time can find patterns in comparably scaled numbers.
        #
        # We get back a NumPy array of 13 cepstral coefficients per row by a number of rows matching
        # the number of windows across the steps in the wave samples. We drop the first column per
        # common implementations of MFCC machine learning for voice.
        self.mfccFeatures = mfcc(samples, rateHz, winlen=frameWindowSec, winstep=windowStepLengthSec, nfft=nfft, highfreq=mfccMaxRangeHz)[:,1:13]

        if produceFirstDerivative:
            # Calculate the deltas (first derivative, velocity) as additional feature info. '2' is number of MFCC rows
            # before and after the current row whose samples are averaged to get the delta. 13 columns.
            self.mfccDeltas = delta(self.mfccFeatures, 2)

            if produceSecondDerivative:
                # Also useful is the delta-delta (second derivative, acceleration) calculated on the deltas. 13 columns.
                self.mfccDeltaDeltas = delta(self.mfccDeltas, 2)

        # Now that we're done with derivatives, normalize the original MFCC coefficients.
        # This is the same algorithm each frame of the MFCCs of an input sound stream
        # will need to use to match against these normalized values, producing far better results.
        normalizeMfccArray(self.mfccFeatures)

        if produceLogFbank:
            # Calculate log-MFCC-filterbank features from the original samples.
            # We keep many defaults: 26 filters in the MFCC filterbank, 0 Hz low frequency,
            # rateHz/2 high frequency, 0.97 pre-emphasis filter, 22 lifter on final cepstral coefficients.
            # We get back a NumPy array of 26 log(filterbank) entries. We keep skip the low coefficient
            # and take the 2nd through 13th, as later banks measure fast-changing harmonics in the
            # high frequencies.
            logFbankFeatures = logfbank(samples, rateHz, frameWindowSec, windowStepLengthSec, nfft=nfft)
            self.logFbankFeatures = logFbankFeatures[:,1:13]
            self.fullFeatureArray = numpy.stack([ self.logFbankFeatures ], axis=-1)
            self.csvHeader = MfccWav.logFbankHeader

        else:
            toStack = [ self.mfccFeatures ]
            self.csvHeader = MfccWav.mfccHeader
            if produceFirstDerivative:
                toStack.append(self.mfccDeltas)
                self.csvHeader += "," + MfccWav.mfccDerivativeHeader
                if produceSecondDerivative:
                    toStack.append(self.mfccDeltaDeltas)
                    self.csvHeader += "," + MfccWav.mfcc2ndDerivativeHeader

            # Nx12xM
            self.fullFeatureArray = numpy.stack(toStack, axis=-1)

    def writeFullFeatureArrayToCsvStream(self, outStream):
        twoDMatrix = self.fullFeatureArray[:,:,0]
        numpy.savetxt(sys.stdout, twoDMatrix, delimiter=',', header=self.csvHeader, comments='')

class MfccWavLoader:
    """
    Generates one or more MfccWav instances from an input wave file,
    depending on file metadata.
    """

    stepSuffixRegex = re.compile('.*-(\d\.\d\d\d)$', re.IGNORECASE)

    def __init__(self, wavPath, mfccMaxRangeHz=None, produceLogFbank=False, produceFirstDerivative=False, produceSecondDerivative=False):
        self.wavPath = wavPath
        self.mfccMaxRangeHz = mfccMaxRangeHz
        self.produceLogFbank = produceLogFbank
        self.produceFirstDerivative = produceFirstDerivative
        self.produceSecondDerivative = produceSecondDerivative

    def generateMfccs(self):
        # Convert the WAV file into monaural samples in a NumPy array.
        (rateHz, samples) = wav.read(self.wavPath)
        print("Loaded", self.wavPath, rateHz, "Hz")

        # Yield the full set of samples as the first output.
        yield MfccWav(self.wavPath, samples, rateHz, self.mfccMaxRangeHz, self.produceLogFbank, self.produceFirstDerivative, self.produceSecondDerivative)

        # Generate subsets from special filename formats.
        fileName = path.basename(self.wavPath)
        baseFileName, _ = path.splitext(fileName)

        # Sounds with base filenames ending in "-x.yyy" generate one additional sample
        # at each step length seconds after x.yyy seconds into the full sample.
        # This allows easier handling for samples where the prefix before x.yyy
        # is the most important but we can get more data at more sample lengths
        # using successive portions of the tail.
        match = self.stepSuffixRegex.match(baseFileName)
        if match:
            beginTimeCode = float(match.group(1))
            samplesPerStep = rateHz * windowStepLengthSec
            beginSample = beginTimeCode * rateHz
            currentEndSample = samples.shape[0] - samplesPerStep  # Full array was emitted above.
            while currentEndSample >= beginSample:
                yield MfccWav(self.wavPath, samples[0:int(currentEndSample)], rateHz, self.mfccMaxRangeHz, self.produceLogFbank, self.produceFirstDerivative, self.produceSecondDerivative)
                currentEndSample -= samplesPerStep

def normalizeMfccArray(mfccs):
    # Per http://www.cs.toronto.edu/%7Efritz/absps/waibelTDNN.pdf : Subtract from each coefficient
    # the average coefficient energy computed over all frames, then normalize each coefficient
    # to lie in [-1, 1]. This is the same algorithm each frame of the MFCCs of an input sound stream
    # will need to run to match against these normalized values, and produces far better results.
    avg = numpy.average(mfccs)
    numpy.subtract(mfccs, avg)
    mfccs /= numpy.max(numpy.abs(mfccs))

def calculate_nfft(samplerate, winlen):
    """Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.
    
    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.

    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft
