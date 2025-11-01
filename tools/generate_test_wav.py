import numpy as np
from scipy.io import wavfile

sr = 44100
duration = 5.0
t = np.linspace(0, duration, int(sr*duration), endpoint=False)
# two-tone signal
sig = 0.5 * np.sin(2*np.pi*440*t) + 0.25 * np.sin(2*np.pi*880*t)
# scale to float32
sig = sig.astype(np.float32)
wavfile.write('test_stft.wav', sr, sig)
print('Wrote test_stft.wav', sig.shape)
