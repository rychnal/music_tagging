import numpy as np
import librosa as lbr
from librosa.display import specshow
import matplotlib.pyplot as plt

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
        'pop', 'reggae', 'rock']
WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 256
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

def load_track(filename, enforce_shape=None):
    '''vytvoreni melspectrogramu z pisnikcy a vytvoreni grafu'''

    new_input, sample_rate = lbr.load(filename, duration=30, mono=True)

    features = lbr.feature.melspectrogram(y=new_input, **MEL_KWARGS)

    if enforce_shape is not None:
        if features.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - features.shape[0],
                    enforce_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > enforce_shape[0]:
            features = features[: enforce_shape[0], :]

        if features.shape[1] < enforce_shape[1]:
            delta_shape = (enforce_shape[0],
                           enforce_shape[1]- features.shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=1)
        elif features.shape[1] > enforce_shape[1]:
            features = features[: enforce_shape[1], :]

        '''plt.figure(figsize=(10, 4))
        specshow(lbr.power_to_db(features,ref = np.max),y_axis = 'mel', fmax = 8000,x_axis = 'time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.savefig('spectogram_final_128mel\\%s' % os.path.basename(filename)[:-2] + 'png', format='png')
        plt.close()'''

    features[features == 0] = 1e-6
    return (np.log(features), float(new_input.shape[0]) / sample_rate)
