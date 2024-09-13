from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import speech_recognition as sr
import os
import torch
from transformers import BertForSequenceClassification
from kobert_transformers import get_tokenizer
from pydub import AudioSegment

app = Flask(__name__)

UPLOAD_FOLDER = './voice_data'
TEXT_DATA_FOLDER = './text_data'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'aac', 'ogg', 'flac', 'wma', 'aiff', 'aif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEXT_DATA_FOLDER'] = TEXT_DATA_FOLDER

# KoBERT Tokenizer 및 모델 로드
tokenizer = get_tokenizer()
model_path = './model/epoch_5'  # 마지막 에포크에서 저장된 모델 체크포인트 경로
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

def allowed_file(filename):
    """파일 확장자가 허용된 형식인지 확인합니다."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(file_path):
    """음성 파일을 WAV 형식으로 변환합니다."""
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_path, format='wav')
    return wav_path

def audio_to_text(filepath):
    """음성 파일을 텍스트로 변환합니다."""
    if not filepath.lower().endswith('.wav'):
        filepath = convert_to_wav(filepath)
        
    recognizer = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='ko-KR')
        except sr.UnknownValueError:
            text = "음성을 인식할 수 없습니다."
        except sr.RequestError:
            text = "음성 인식 서비스에 접근할 수 없습니다."
    return text

def get_kobert_embeddings(text):
    """입력 텍스트를 KoBERT로 임베딩합니다."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

def predict(text):
    """입력 텍스트가 보이스피싱인지 예측합니다."""
    embeddings = get_kobert_embeddings(text)
    pred = torch.argmax(embeddings, dim=1).item()
    return 0 if pred == 1 else 1

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 및 예측 결과 반환 API."""
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            text = audio_to_text(filepath)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        text_file = os.path.join(app.config['TEXT_DATA_FOLDER'], f'{filename}.txt')
        with open(text_file, 'w', encoding='utf-8') as file:
            file.write(text)
        
        result = predict(text)

        # 결과만 반환
        return jsonify({'result': result})

    else:
        return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(TEXT_DATA_FOLDER):
        os.makedirs(TEXT_DATA_FOLDER)
    app.run(host='0.0.0.0', port=5001, debug=True)