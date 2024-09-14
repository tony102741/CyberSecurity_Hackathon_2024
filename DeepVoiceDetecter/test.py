# import module
import librosa
import numpy as np
import os
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.python import keras

# #### parameters ####
SAMPLE_RATE = 48000
FRAME_SIZE = 2048
HOP_LENGTH = 512
WIN_LENGTH = None
DATA_TYPE = np.float64
N_FFT = 2048
AUDIO_FILE_CUT_LENGTH = 128
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12

# file location
model_file_location = os.path.join("models", "model")
test_voice_location = "VoiceFile(DeepVoice)\\test4.wav"

test_voice_converted = list()

# After loading the file, convert it to Mel Spectogram.
signal, samp_r = librosa.load(test_voice_location, sr=SAMPLE_RATE, mono=True, dtype=DATA_TYPE)
S = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

# normalization
norm = normalize(librosa.power_to_db(S, ref=np.max))

# Divide the length of the mel spectogram into equal sizes
# Data fragments smaller than the specified length are padded with 0.
if norm.shape[1] > AUDIO_FILE_CUT_LENGTH:
    splited = np.array_split(norm, np.ceil(norm.shape[1]/AUDIO_FILE_CUT_LENGTH), 1)
    for part in splited:
        if part.shape[1] < AUDIO_FILE_CUT_LENGTH:
            test_voice_converted.append(np.pad(part, ((0, 0), (0, AUDIO_FILE_CUT_LENGTH-part.shape[1])), 'constant', constant_values=0))
        else: test_voice_converted.append(part)
elif norm.shape[1] < AUDIO_FILE_CUT_LENGTH: norm = np.pad(norm, ((0, 0), (0, AUDIO_FILE_CUT_LENGTH-norm.shape[1])), 'constant', constant_values=0)
else: test_voice_converted.append(norm)

# A weak solution for solving the CNN input dimensionality problem.
test_data = np.asarray(test_voice_converted, dtype=np.float64)
test_data = tf.expand_dims(test_data, axis=-1)

# load model
model = keras.models.load_model(model_file_location, compile=False)

# prediction
y_pred = model.predict(test_data)

if y_pred.mean() > 0.5:
    # As a result of the prediction, this voice is a voice generated through deep learning artificial intelligence.
    print("AI")
else:
    # As a result of the prediction, this voice is a human voice.
    print("Human")