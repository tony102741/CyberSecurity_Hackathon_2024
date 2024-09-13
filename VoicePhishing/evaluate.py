import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertForSequenceClassification
from kobert_transformers import get_tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('./training_data/voicephishing.csv')

# NaN 값 처리
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].fillna('')

# 데이터 전처리
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted_labels.cpu().numpy())
    
    # 정확도, 정밀도, 재현율, F1 점수 계산
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return accuracy, precision, recall, f1

def load_and_evaluate_models(model_dir, dataset, tokenizer, max_len=128, batch_size=8):
    # 모델 디렉토리를 epoch 숫자로 정렬
    model_paths = sorted(
        [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))],
        key=lambda x: int(x.split('_')[-1])
    )
    models = {}
    
    for path in model_paths:
        epoch = os.path.basename(path)
        print(f"Loading model from {path}...")
        try:
            model = BertForSequenceClassification.from_pretrained(path)
            model.eval()
            models[epoch] = model
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")

    # 데이터셋을 학습 데이터와 평가 데이터로 분할
    total_size = len(dataset)
    eval_size = int(0.1 * total_size)  # 전체 데이터의 10%를 평가 데이터로 사용
    train_size = total_size - eval_size

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # 평가 수행
    device = torch.device('cpu')  # GPU 사용하지 않음
    
    # 평가 결과 저장용 리스트
    epochs = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for epoch, model in models.items():
        model.to(device)
        accuracy, precision, recall, f1 = evaluate_model(model, eval_loader, device)
        print(f"Epoch {epoch}: Validation Accuracy = {accuracy:.4f}")
        print(f"Epoch {epoch}: Validation Precision = {precision:.4f}")
        print(f"Epoch {epoch}: Validation Recall = {recall:.4f}")
        print(f"Epoch {epoch}: Validation F1 Score = {f1:.4f}")

        # 결과 저장
        epochs.append(int(epoch.split('_')[-1]))
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracies, label='Accuracy', marker='o')
    plt.plot(epochs, precisions, label='Precision', marker='o')
    plt.plot(epochs, recalls, label='Recall', marker='o')
    plt.plot(epochs, f1_scores, label='F1 Score', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics per Epoch')
    plt.legend()
    plt.grid(True)

    # 한글 폰트 설정 (MacOS)
    plt.rcParams['font.family'] = 'AppleGothic' # Mac에서 AppleGothic 사용
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 설정
    
    # x축의 눈금을 정수로 설정
    plt.xticks(epochs)  # epoch 리스트를 사용하여 x축 눈금 설정

    plt.show()

# KoBERT Tokenizer 로드
tokenizer = get_tokenizer()

# 전체 데이터셋
texts = df['text'].tolist()
labels = df['label'].tolist()
dataset = TextDataset(texts, labels, tokenizer, max_len=128)

# 모델 로드 및 평가
model_dir = './model'
load_and_evaluate_models(model_dir, dataset, tokenizer)