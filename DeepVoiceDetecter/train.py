# import module
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint
from keras import backend as K

# #### parameters ####
# Mel Spectogram
SAMPLE_RATE = 48000
FRAME_SIZE = 2048
HOP_LENGTH = 512
WIN_LENGTH = None
DATA_TYPE = np.float64
N_FFT = 2048
AUDIO_FILE_CUT_LENGTH = 128
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 100

# Set Directory
deep_voice_record_file_dir = "VoiceFile(DeepVoice)"
normal_voice_record_file_dir = "VoiceFile(Normal)"
model_save_dir = os.path.join("models", "model")

# Find and load the wav file to use for training
# Excluding special directories in Unix-like
deep_voice_record_file_list = os.listdir(deep_voice_record_file_dir)
if('.' in deep_voice_record_file_list): deep_voice_record_file_list.remove('.')
if('..' in deep_voice_record_file_list): deep_voice_record_file_list.remove('..')
normal_voice_record_file_list = os.listdir(normal_voice_record_file_dir)
if('.' in normal_voice_record_file_list): normal_voice_record_file_list.remove('.')
if('..' in normal_voice_record_file_list): normal_voice_record_file_list.remove('..')
deep_voice_converted = list()
normal_voice_converted = list()

# Apply preprocessing to each file
# (Deep learning artificial intelligence generated voice file)
for deep_voice_file in deep_voice_record_file_list:
    # After loading the file, convert it to Mel Spectogram.
    signal, samp_r = librosa.load(os.path.join(deep_voice_record_file_dir, deep_voice_file), sr=SAMPLE_RATE, mono=True, dtype=DATA_TYPE)
    S = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    
    # normalization
    norm = normalize(librosa.power_to_db(S, ref=np.max))
    
    # Divide the length of the mel spectogram into equal sizes
    # Data fragments smaller than the specified length are padded with 0.
    if norm.shape[1] > AUDIO_FILE_CUT_LENGTH:
        splited = np.array_split(norm, np.ceil(norm.shape[1]/AUDIO_FILE_CUT_LENGTH), 1)
        for part in splited:
            if part.shape[1] < AUDIO_FILE_CUT_LENGTH:
                deep_voice_converted.append(np.pad(part, ((0, 0), (0, AUDIO_FILE_CUT_LENGTH-part.shape[1])), 'constant', constant_values=0))
            else: deep_voice_converted.append(part)
    elif norm.shape[1] < AUDIO_FILE_CUT_LENGTH: norm = np.pad(norm, ((0, 0), (0, AUDIO_FILE_CUT_LENGTH-norm.shape[1])), 'constant', constant_values=0)
    else: deep_voice_converted.append(norm)
    
# Convert to NumPy array
deep_voice_converted = np.asarray(deep_voice_converted, dtype=np.float64)

# Apply preprocessing to each file
# Real human voice file
for normal_voice_file in normal_voice_record_file_list:
    # After loading the file, convert it to Mel Spectogram.
    signal, samp_r = librosa.load(os.path.join(normal_voice_record_file_dir, normal_voice_file), sr=SAMPLE_RATE, mono=True, dtype=DATA_TYPE)
    S = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    # normalization
    norm = normalize(librosa.power_to_db(S, ref=np.max))
    
    # Divide the length of the mel spectogram into equal sizes
    # Data fragments smaller than the specified length are padded with 0.
    if norm.shape[1] > AUDIO_FILE_CUT_LENGTH:
        splited = np.array_split(norm, np.ceil(norm.shape[1]/AUDIO_FILE_CUT_LENGTH), 1)
        for part in splited:
            if part.shape[1] < AUDIO_FILE_CUT_LENGTH:
                normal_voice_converted.append(np.pad(part, ((0, 0), (0, AUDIO_FILE_CUT_LENGTH-part.shape[1])), 'constant', constant_values=0))
            else: normal_voice_converted.append(part)
    elif norm.shape[1] < AUDIO_FILE_CUT_LENGTH: norm = np.pad(norm, ((0, 0), (0, AUDIO_FILE_CUT_LENGTH-norm.shape[1])), 'constant', constant_values=0)
    else: normal_voice_converted.append(norm)
    
# Convert to NumPy array
normal_voice_converted = np.asarray(normal_voice_converted, dtype=np.float64)

# Generate labels as many files as loaded
label = np.concatenate((np.ones(deep_voice_converted.shape[0]), np.zeros(normal_voice_converted.shape[0])))

# Merging generated and real human voices
data = np.concatenate((deep_voice_converted, normal_voice_converted), axis=0)

# Shuffle and split training data and test data
train_images, test_images, train_labels, test_labels = train_test_split(data, label)

# dimension expansion
train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)

# Linear convolutional neural network definition(binary classifications)
# Since it is a randomly defined neural network,
# it is a good idea to replace this part if there is a neural network definition
# with better performance.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# How to get recall, accuracy, and F1 score in Keras
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Use checkpoints
checkpoint = ModelCheckpoint("weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

# model compile
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', f1_m, precision_m, recall_m])

# Training a neural network model
history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels), callbacks=[CSVLogger(filename="history.csv"), checkpoint])

# evaluation
loss, accuracy, f1_score, precision, recall = model.evaluate(test_images, test_labels)

# model save
model.save(model_save_dir)