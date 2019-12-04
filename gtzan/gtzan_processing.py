import librosa.display
import numpy as np
import glob
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

# fft parameters
sampling_rate = 12000
num_fft = 2048
hop_length_cus = 512
duration_in_seconds = 29.12  # to make it 1366 frame (1366 = 12000 * 29.12 / 256)


def compute_ffts():
    with open('./test_songs_gtzan_list.txt', 'rt') as file:
        for line in file:
            audio_path = '/content/drive/My Drive/' + line.strip().replace("au", "wav")
            fft_path = audio_path + '-fft.npy'
            src, sc = librosa.load(audio_path)
            fft = np.abs(librosa.stft(src, n_fft=num_fft, hop_length=hop_length_cus)[0:(int(num_fft / 2)) + 1])
            np.save(fft_path, fft)


# mel-spectrogram parameters
n_fft = 512
n_mels = 96
hop_length = 256


def compute_mel():
    with open('./test_songs_gtzan_list.txt', 'rt') as file:
        for line in file:
            audio_path = '/content/drive/My Drive/' + line.strip().replace("au", "wav")
            mg_path = audio_path + '-mg.npy'
            src, sr = librosa.load(audio_path, sr=sampling_rate)  # whole signal
            n_sample = src.shape[0]
            n_sample_fit = int(duration_in_seconds * sampling_rate)

            if n_sample < n_sample_fit:  # if too short
                src = np.hstack((src, np.zeros((int(duration_in_seconds * sampling_rate) - n_sample,))))
            elif n_sample > n_sample_fit:  # if too long
                src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
            logam = librosa.core.amplitude_to_db
            melgram = librosa.feature.melspectrogram
            mg = logam(melgram(y=src, sr=sampling_rate, hop_length=hop_length,
                               n_fft=n_fft, n_mels=n_mels) ** 2, ref=1.0)
            np.save(mg_path, mg)


def display_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()


def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))
    return np.ndarray.flatten(mfcc)[:25000]


def generate_features_and_labels():
    all_features = []
    all_labels = []

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for genre in genres:
        sound_files = glob.glob("/gtzan/genres/" + genre + '/*.wav')
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    return np.stack(all_features), onehot_labels


features, labels = generate_features_and_labels()

np.save("./data/flatten_mfcc.npy", features)
np.save("./data/one_hot_labels.npy", labels)