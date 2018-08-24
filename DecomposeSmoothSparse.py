#   Import Libraries
import essentia.standard as es
import numpy as np
import os, sys
from librosa.core import stft, istft                    # Librosa Version 0.6.0
from essentia import Pool, array                        # Essentia Version 2.1-dev
import time

# --------------------------------------- Utility Function: Timer --------------------------------------#
# Used for Calculating Execution Times of different functions in algorithm


def timing(f, timer_enabled=False):       # set timer_enabled True to time functions or set False to disable timer
    def wrap(*args):
        if timer_enabled:
            time1 = time.time()
        ret = f(*args)
        if timer_enabled:
            time2 = time.time()
            print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap
# ------------------------------------------------------------------------------------------------------#


# --------------------------------------- Class definition: Separator ----------------------------------#

class HPSS:

    # Harmonic + Percussive separator based on [1]
    #
    # References:
    #       [1] Canadas-Quesada, Francisco Jesus, et al. "Percussive/harmonic sound separation by non-negative matrix
    #           factorization with smoothness/sparseness constraints." EURASIP Journal on Audio, Speech, and Music
    #           Processing 2014.1 (2014): 26.
    #       [2] Kameoka, Hirokazu. "Non-negative matrix factorization and its variants for audio signal processing."
    #           Applied Matrix and Tensor Variate Data Analysis. Springer, Tokyo, 2016. 25.
    #

    def __init__(self, x, **options):
        #
        #   x (1-D Array )      :   Time-domain mono audio signal
        #
        #   Options:
        #       directory(str)  :   directory to save the results
        #                           (exp: directory = "data/separated/")
        #                           (default = "")
        #       filename (str)  :   filename to save the results
        #                           (exp: filename = "trackname")
        #                           (default = "")
        #       fs (float)      :   sampling rate for saving the separated signals
        #       beta (float)    :   constant used in beta divergence cost (default = 1)
        #                           if 0: Itakura-Saito (IS) divergence
        #                           if 1: Kullback-Leibler (KL) divergence
        #                           if 2: Euclidean-Distance divergence
        #       Rp (int)        :   Number of percussive components (default = 150)
        #       Rh (int)        :   Number of harmonic components   (default = 150)
        #       fftSize (int)   :   FFT size for stft/istft calculations    (default = 150)
        #       frameSize (int) :   Frame size for stft/istft calculations    (default = 150)
        #       hopSize (int)   :   Hop size for stft/istft calculations
        #       winType (str)   :   Type of window applied to each frame for stft/istft calculations
        #                           (based on window types in scipy.signal.get_window)
        #       maxIter (int)   :   Maximum number of iterations to update basis vectors and activations (default = 100)
        #       K_SSM (float)   :   Spectral smoothness weight  (default = .1)
        #       K_SSP (float)   :   Spectral sparseness weight  (default = .1)
        #       K_TSM (float)   :   Temporal smoothness weight  (default = .1)
        #       K_TSP (float)   :   Temporal sparseness weight  (default = .1)
        #       StatusBar       :   pyQt status bar for outputing messages (optional)

        #   Initialize object parameters
        self.directory = ""                     # Directory for writing the separated audio files
        self.filename = ""                      # Filename for writing the separated audio files

        self.format = "wav"                     # Format of separated files to be saved to file
        self.formatIsSpecified = "False"        # Flag used for notifying user if default value is used

        self.fs = 44100                         # Sample rate to write the separated files
        self.fsIsSpecified = "False"            # Flag used for notifying user if default value is used

        self.fftSize = 2048                     # fft size used for stft/istft operations
        self.fftSizeIsSpecified = "False"       # Flag used for notifying user if default value is used

        self.frameSize = 2048                   # Frame size used for stft/istft operations
        self.frameSizeIsSpecified = "False"     # Flag used for notifying user if default value is used

        self.hopSize = 512                      # hop size used for stft/istft operations
        self.hopSizeIsSpecified = "False"       # Flag used for notifying user if default value is used

        self.winType = "hann"                   # window type used for stft/istft operations
        self.winTypeIsSpecified = "False"       # Flag used for notifying user if default value is used

        self.beta = 1.5                         # beta divergence coefficient

        self.Rp = 150                           # Number of percussive components
        self.Rh = 150                           # Number of harmonic components (Remember that (Rp+Rh) < min(F, T)) [2]
        self.maxIter = 100.0                    # Number of iterations for updating basis and activation matrices

        self.K_SSM = .1                         # Spectral smoothness weight in cost function
        self.K_SSP = .1                         # Spectral sparseness weight in cost function
        self.K_TSM = .1                         # Temporal smoothness weight in cost function
        self.K_TSP = .1                         # Temporal sparseness weight in cost function

        # pyqt status bar for outputing messages
        self.StatusBar = None

        #   Check parameters to be updated
        for option in options:

            if option == "directory":
                self.directory = str(options.get(option))
                if self.directory[-1] != "/":
                    self.directory += "/"

            if option == "filename":
                self.filename = str(options.get(option))

            if option == "format":
                self.format = str(options.get(option))
                self.formatIsSpecified = "True"

            if option == "fs":
                self.fs = str(options.get(option))
                self.fsIsSpecified = "True"

            if option == "fftSize":
                self.fftSize = int(options.get(option))
                self.fftSizeIsSpecified = "True"

            if option == "frameSize":
                self.frameSize = int(options.get(option))
                self.frameSizeIsSpecified = "True"

            if option == "hopSize":
                self.hopSize = int(options.get(option))
                self.hopSizeIsSpecified = "True"

            if option == "winType":
                self.hopSize = str(options.get(option))
                self.hopSizeIsSpecified = "True"

            if option == "beta":
                self.beta = np.float(options.get(option))

            if option == "Rp":
                self.Rp = np.int(options.get(option))

            if option == "Rh":
                self.Rh = np.int(options.get(option))

            if option == "maxIter":
                self.maxIter = np.int(options.get(option))

            if option == "K_SSM":
                self.K_SSM = np.float(options.get(option))

            if option == "K_SSP":
                self.K_SSP = np.float(options.get(option))

            if option == "K_TSM":
                self.K_TSM = np.float(options.get(option))

            if option == "K_TSP":
                self.K_TSP = np.float(options.get(option))

            if option == "StatusBar":
                self.StatusBar = options.get(option)

        #   Initialize separator object matrices and vectors

        self.xOriginal = x                      # Original time series signal

        self.Xc = self.stft(self.xOriginal)     # complex spectrogram of xOriginal/ Xc  <F(freq bins)xT(Time frames)>

        self.Spec = np.abs(self.Xc)             # magnitude spectrogram of xOriginal    <F(freq bins)xT(Time frames)>

        self.F = np.size(self.Xc, 0)            # number of frequency bins
        self.T = np.size(self.Xc, 1)            # number of time frames

        self.X_nBeta = self.beta_normalize()    # magnitude normalized spectrogram

        self.W_P = np.ones([self.F, self.Rp])           # Percussive basis vectors (numpy matrix)           <FxRp>
        self.W_P = np.random.rand(self.F, self.Rp)      # Percussive basis vectors (numpy matrix)
        self.W_P = np.matrix(self.W_P)

        self.H_P = np.ones([self.Rp, self.T])           # Percussive temporal activations (numpy matrix)    <RpxT>
        self.H_P = np.random.rand(self.Rp, self.T)      # Percussive temporal activations (numpy matrix)
        self.H_P = np.matrix(self.H_P)

        self.W_H = np.ones([self.F, self.Rh])           # Harmonic basis vectors (numpy matrix)             <FxRh>
        self.W_H = np.random.rand(self.F, self.Rh)      # Harmonic basis vectors (numpy matrix)
        self.W_H = np.matrix(self.W_H)

        self.H_H = np.ones([self.Rh, self.T])           # Harmonic temporal activations (numpy matrix)      <RhxT>
        self.H_H = np.random.rand(self.Rh, self.T)      # Harmonic temporal activations (numpy matrix)
        self.H_H = np.matrix(self.H_H)

        self.A = []                                     # $W_{P}H_{P}+W_{H}H_{H}$                       (numpy matrix)
        self.A1 = []                                    # $(W_{P}H_{P}+W_{H}H_{H})^(\beta-1)$
        self.A2 = []                                    # $(W_{P}H_{P}+W_{H}H_{H})^{(\beta-2)}\odot{X_{n\beta}}$

        self.M_P = []                                   # Percussive mask (numpy matrix)                    <FxT>
        self.M_H = []                                   # Harmonic mask   (numpy matrix)                    <FxT>

        self.X_P = []                                   # Percussive Spectrogram (numpy matrix)             <FxT>
        self.X_H = []                                   # Harmonic Spectrogram   (numpy matrix)             <FxT>

        self.x_p = []                                   # Percussive signal time domain (numpy array)
        self.x_h = []                                   # Harmonic signal time domain   (numpy array)

    @timing
    def beta_normalize(self):
        #   Following function normalizes the magnitude spectrogram
        #   returns magnitude normalized spectrogram X_nBeta
        #   refer to equation (2) in reference
        SpecPowered = np.power(self.Spec, self.beta)
        denom = np.sum(np.sum(SpecPowered, 0))
        denom = denom/(float(self.F*self.T))
        denom = np.power(denom, 1.0/self.beta)
        X_nBeta = SpecPowered / denom
        return X_nBeta

    @timing
    def update_bases_and_activations(self):
        #   This function updates the basis vectors W_P, H_P, W_H, and H_H for a single step of iteration

        self.calc_common_factors()    # calculate commonly used matrices

        next_W_P = np.multiply(self.W_P, (self.eq19()+self.K_SSM*self.eq21())/(self.eq20()+self.K_SSM*self.eq22()))
        next_H_P = np.multiply(self.H_P, (self.eq23()+self.K_TSP*self.eq25())/(self.eq24()+self.K_TSP*self.eq26()))
        next_W_H = np.multiply(self.W_H, (self.eq27()+self.K_SSP*self.eq29())/(self.eq28()+self.K_SSP*self.eq30()))
        next_H_H = np.multiply(self.H_H, (self.eq31()+self.K_TSM*self.eq33())/(self.eq32()+self.K_TSM*self.eq34()))

        self.W_P = next_W_P
        self.H_P = next_H_P
        self.W_H = next_W_H
        self.H_H = next_H_H

    @timing
    def create_masks(self):
        #   This function creates the masks M_P, M_H
        self.X_P = self.W_P*self.H_P
        self.X_H = self.W_H*self.H_H

        self.M_P = np.power(self.X_P, 2)/(np.power(self.X_P, 2)+np.power(self.X_H, 2))  #Equation 15 in reference
        self.M_H = np.power(self.X_H, 2)/(np.power(self.X_P, 2)+np.power(self.X_H, 2))  #Equation 16 in reference

    @timing
    def stft(self, audio=[]):
        # This function takes the stft of an input signal
        # Messages to be output
        if self.fftSizeIsSpecified == "False":
            print("fft size is not specified for STFT calculation. Assumed frameSize is " + str(self.fftSize))

        if self.frameSizeIsSpecified == "False":
            print("frameSize is not specified for STFT calculation. Assumed frameSize is " + str(self.frameSize))

        if self.hopSizeIsSpecified == "False":
            print("hopSize is not specified for STFT calculation. Assumed hopSize is " + str(self.hopSize))

        if self.winTypeIsSpecified == "False":
            print("Window type is not specified for STFT calculation. Assumed hopSize is " + str(self.winType))

        # Take the stft using Librosa.core.stft
        if audio == []:
            _x = self.xOriginal     # if an array is not specified use xOriginal
        else:
            _x = audio              # otherwise, use the specified signal

        _stft = stft(_x, n_fft=self.fftSize, hop_length=self.hopSize, win_length=self.frameSize, window=self.winType)

        return _stft

    @timing
    def istft(self, _stft=[]):
        # Take the stft using Librosa.core.stft
        # returns numpy array
        if _stft == []:
            X = self.Xc            # if an array is not specified use Xc (stft of xOriginal)
        else:
            X = _stft              # otherwise, use the specified signal

        _x = istft(X, hop_length=self.hopSize, win_length=self.frameSize)

        return _x

    @timing
    def spectral_to_temporal_using_masks(self, normalize=False):
        self.x_p = self.istft(np.asarray(np.multiply(self.M_P, self.Xc)))
        self.x_h = self.istft(np.asarray(np.multiply(self.M_H, self.Xc)))
        if normalize:
            gain = np.max([np.max(np.abs(self.x_p)), np.max(np.abs(self.x_h))])
            self.x_p = self.x_p / gain
            self.x_h = self.x_h / gain

    @timing
    def separate(self):
        # Separates harmonic / percussive signals by iteration - Steps 3 to 10 in Algorithm 1 detailed in reference
        for i in range(int(self.maxIter)):
            print("Iteration %i out of %i" % (i+1, self.maxIter))
            self.next_iteration()

        self.create_masks()
        self.spectral_to_temporal_using_masks()

    def next_iteration(self):
        self.update_bases_and_activations()


    @timing
    def save_separated_audiofiles(self):
        # check and see if directory exists
        if self.directory != "" and ("/" in self.directory):
            directoryLevels = self.directory.split("/")
            for ixLevel, directoryLevel in enumerate(directoryLevels):
                if ixLevel == 0:
                    LevelPath = directoryLevel
                else:
                    LevelPath += "/" + directoryLevel
                if not os.path.isdir(LevelPath):
                    os.mkdir(LevelPath)

        # Create audio writer object
        if self.fsIsSpecified == "False":   # Notify user if sampling rate not specified
            print("Sample Rate not specified for writing the audio files. Assumed Fs (Hz) is " + str(self.fs))

        if self.formatIsSpecified == "False":   # Notify user if format not specified
            print("File format not specified for writing the audio files. Assumed format is " + str(self.format))

        MonoWriter = es.MonoWriter(sampleRate=self.fs, format=self.format)
        MonoWriter.configure(filename=self.directory+self.filename+"_percussive."+self.format)
        MonoWriter(array(self.x_p))

        MonoWriter = es.MonoWriter(sampleRate=self.fs, format=self.format)
        MonoWriter.configure(filename=self.directory + self.filename + "_harmonic." + self.format)
        MonoWriter(array(self.x_h))

    #   Functions to calculate EqXX in appendix section of reference (EqXX refers to Equation XX in appendix)
    @timing
    def calc_common_factors(self):
        # Calculates and stores the matrix operations repeatedly used in the following equations
        # Stores the results in self.A, self.A1 and self.A2
        self.A = self.W_P*self.H_P+self.W_H*self.H_H+np.finfo(float).eps
        self.A1 = np.power(self.A, self.beta-1)
        self.A2 = np.multiply(np.power(self.A, self.beta-2), self.X_nBeta)

    @timing
    def eq19(self):
        Eq19 = self.A2 * self.H_P.transpose()
        return Eq19

    @timing
    def eq20(self):
        return self.A1 * self.H_P.transpose()

    @timing
    def eq21(self):
        Eq21 = np.zeros_like(self.W_P)
        for rp in range(self.Rp):
            termA = np.sum(np.power(self.W_P[1:, rp]-self.W_P[:-1, rp], 2))
            termB = np.sum(np.power(self.W_P[:, rp], 2))
            termC = termB**2
            termA_divBy_C = termA / termC

            W_P_minus1 = np.append(np.array([[self.W_P[0, rp]]]), self.W_P[:-1, rp], axis=0)
            W_P_plus1 = np.append(self.W_P[1:, rp], np.array([[self.W_P[-1, rp]]]), axis=0)

            term1 = 2 * self.F / termB * (W_P_minus1+W_P_plus1)
            term2 = 2 * self.F * self.W_P[:, rp] * termA_divBy_C
            Eq21[:, rp] = term1 + term2

            '''
            for f in range(self.F):
                if f == 0:
                    ix0 = 0
                else:
                    ix0 = f-1
                if f == self.F-1:
                    ix1 = self.F-1
                else:
                    ix1 = f+1
                term1 = 2*self.F*(self.W_P[ix0, rp]+self.W_P[ix1, rp])/termB
                term2 = 2*self.F*self.W_P[f, rp]*termA_divBy_C
                Eq21[f, rp] = term1+term2
            '''
        return Eq21

    @timing
    def eq22(self):
        Eq22 = np.matrix(np.zeros_like(self.W_P))
        for rp in range(self.Rp):
            Eq22[:, rp] = 4*self.F*self.W_P[:, rp]/np.sum(np.power(self.W_P[:, rp], 2))
        return Eq22

    @timing
    def eq23(self):
        Eq23 = self.W_P.transpose() * self.A2
        return Eq23

    @timing
    def eq24(self):
        return self.W_P.transpose() * self.A1

    @timing
    def eq25(self):
        Eq25 = np.zeros_like(self.H_P)
        for rp in range(self.Rp):
            sum1 = np.sum(self.H_P[rp, :])
            sum2 = np.sum(np.power(self.H_P[rp, :], 2))

            Eq25[rp, :] = self.T**.5*(self.H_P[rp, :]*sum1) / sum2**1.5
            '''
            for t in range(self.T):
                Eq25[rp, t] = self.T**.5*(self.H_P[rp, t]*sum1) \
                              / sum2**1.5
            '''
        return Eq25

    @timing
    def eq26(self):
        Eq26 = np.zeros_like(self.H_P)
        for rp in range(self.Rp):
            sum1 = np.sum(np.power(self.H_P[rp, :], 2))/np.float(self.T)

            Eq26[rp, :] = np.ones_like(self.H_P[rp, :]) / sum1 ** .5
            '''
            for t in range(self.T):
                Eq26[rp, t] = 1.0/sum1**.5
            '''
        return Eq26

    @timing
    def eq27(self):
        Eq27 = self.A2 * (self.H_H.transpose())
        return Eq27

    @timing
    def eq28(self):
        Eq28 = self.A1 * self.H_H.transpose()
        return Eq28

    @timing
    def eq29(self):
        Eq29 = np.zeros_like(self.W_H)
        for rh in range(self.Rh):
            sum1 = np.sum(self.W_H[:, rh])
            sum2 = np.sum(np.power(self.W_H[:, rh], 2))**1.5
            Eq29[:, rh] = self.F**.5*sum1/sum2*self.W_H[:, rh]
            '''
            sum1_dividedBy_sum2 = sum1/sum2
            for f in range(self.F):
                Eq29[f, rh] = self.F**.5*(self.W_H[f, rh])*sum1_dividedBy_sum2
            '''
        return Eq29

    @timing
    def eq30(self):
        Eq30 = np.zeros_like(self.W_H)
        for rh in range(self.Rh):
            sum1 = (np.sum(np.power(self.W_H[:, rh], 2))/np.float(self.F)) ** 0.5

            Eq30[:, rh] = np.ones_like(self.W_H[:, rh]) / sum1

            '''
            for f in range(self.F):
                Eq30[f, rh] = 1.0/sum1
            '''
        return Eq30

    @timing
    def eq31(self):
        Eq31 = (self.W_H.transpose()) * self.A2
        return Eq31

    @timing
    def eq32(self):
        return self.W_H.transpose() * self.A1

    @timing
    def eq33(self):
        Eq33 = np.zeros_like(self.H_H)
        for rh in range(self.Rh):
            sum1 = np.sum(np.power(self.H_H[rh, :], 2))
            sum2 = np.sum(np.power(self.H_H[rh, 1:]-self.H_H[rh, :-1], 2))

            H_H_minus1 = np.append(np.array([[self.H_H[rh, 0]]]), self.H_H[rh, :-1])
            H_H_plus1 = np.append(self.H_H[rh, 1:], np.array([[self.H_H[rh, -1]]]), axis=1)

            Eq33[rh, :] = 2 * self.T * ((H_H_minus1+H_H_plus1) / sum1) + \
                          (2 * self.T * self.H_H[rh, :] * sum2) / sum1 ** 2

            '''
            for t in range(self.T):
                if t == 0:
                    ix0 = 0
                else:
                    ix0 = t-1
                if t == self.T-1:
                    ix1 = self.T-1
                else:
                    ix1 = t+1

                Eq33[rh, t] = 2*self.T*((self.H_H[rh, ix0]+self.H_H[rh, ix1])/sum1) + \
                              (2*self.T*self.H_H[rh, t]*sum2) / sum1**2
            '''
        return Eq33

    @timing
    def eq34(self):
        Eq34 = np.zeros_like(self.H_H)
        for rh in range(self.Rh):
            sum1 = np.sum(np.power(self.H_H[rh, :], 2))

            Eq34[rh, :] = 4.0 * self.T * self.H_H[rh, :] / sum1
            '''
            for t in range(self.T):
                Eq34[rh, t] = 4.0*self.T*self.H_H[rh, t]/sum1
            '''
        return Eq34

'''
filename = "data/tracks/Mr.FingersMysteryofLove.mp3"

frameSize = 2048
hopSize = 1024
fftSize = 2048

monoLoader = es.MonoLoader(filename=filename, sampleRate=44100)
x = monoLoader()[2:6*44100]
print(x)
hpss = HPSS(
            x,
            directory="data/SparseSmoothSeparated",
            filename="Mr.FingersMysteryofLove",
            format="mp3",
            beta=1.5,
            frameSize=frameSize,
            hopSize=hopSize,
            fftSize=fftSize,
            Rp=150,
            Rh=150,
            maxIter=100.0,
            K_SSM=.1,           # Percussive Spectral Smoothness
            K_TSP=.1,           # Percussive Temporal Smoothness
            K_SSP=.1,           # Harmonic Spectral Smoothness
            K_TSM=.1,           # Harmonic Temporal Smoothness

)

hpss.separate()
hpss.save_separated_audiofiles()

'''