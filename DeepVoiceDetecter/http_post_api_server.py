# import module
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import librosa
import numpy as np
import os
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.python import keras
from pydub import AudioSegment

# flask
app = Flask(__name__)

# #### parameters ####
# web Server
UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'aac', 'ogg', 'flac', 'wma', 'aiff', 'aif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
EPOCHS = 12

# load model
model = keras.models.load_model(os.path.join("models", "model"), compile=False)

# Extension check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert to wav file
def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_path, format='wav')
    return wav_path

# prediction
def predict(sound_file_location):
    test_voice_converted = list()
    
    # After loading the file, convert it to Mel Spectogram.
    signal, samp_r = librosa.load(sound_file_location, sr=SAMPLE_RATE, mono=True, dtype=DATA_TYPE)
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
    
    # Return prediction results
    return model.predict(test_data)

# The content below is related to the Flask web page. Because it is simple code, there are no detailed comments.
@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'no File!'}), 400
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = predict(filepath)
        if result.mean() > 0.5:
            return jsonify({'result': 1})
        else:
            return jsonify({'result': 0})
    else:
        return jsonify({'error': 'not support Ext.'}), 400
    
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=8080)