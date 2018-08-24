# harmonic-percussive-separator
An implementation of harmonic/percussive source separation based on smoothness/sparseness constraints proposed in the following reference:

###### Canadas-Quesada, Francisco Jesus, et al. "Percussive/harmonic sound separation by non-negative matrix factorization with smoothness/sparseness constraints." EURASIP Journal on Audio, Speech, and Music Processing 2014.1 (2014): 26.

# Dependancies
1. Essentia: http://essentia.upf.edu/
2. Librosa:  https://librosa.github.io/librosa/

# How to use
1. Load the separator module
```
from DecomposeSmoothSparse import HPSS

```
2. Locate file name
```
filename = "data/tracks/Mr.FingersMysteryofLove.mp3"

```

3. Specify analysis parameters:
```
frameSize = 2048
hopSize = 1024
fftSize = 2048
```
4. Load Audio using Essentia Monoloader module
```
monoLoader = es.MonoLoader(filename=filename, sampleRate=44100)
x = monoLoader()

```

5. Create a separator object
```
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
```
6. Separate audio file into harmonic/percussive components
```
hpss.separate()
```

7. Save the separated components
```
hpss.save_separated_audiofiles()
```
