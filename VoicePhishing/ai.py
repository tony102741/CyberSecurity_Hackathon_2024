import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertForSequenceClassification
from kobert_transformers import get_tokenizer
from tqdm import tqdm
import os

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

# KoBERT Tokenizer 및 모델 로드
tokenizer = get_tokenizer()
model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)

# 데이터 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.1
)

# 데이터 로더 생성
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 옵티마이저 및 학습 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device('cpu')  # GPU 사용하지 않음
model.to(device)

# 최근 모델 체크포인트 찾기
def get_latest_epoch(model_dir='./model'):
    if not os.path.exists(model_dir):
        return 0
    checkpoints = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not checkpoints:
        return 0
    checkpoints.sort(key=lambda d: int(d.split('_')[-1]))  # 최신 에포크 찾기
    return int(checkpoints[-1].split('_')[-1])

last_epoch = get_latest_epoch()

def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training Epoch", unit="batch"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader):
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

# 학습 루프
num_epochs = 5
for epoch in range(last_epoch + 1, num_epochs + 1):
    avg_loss = train_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch}: Average Loss = {avg_loss}")

    # 모델 체크포인트 저장
    model_save_path = f'./model/epoch_{epoch}'
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)

    # 토크나이저 저장
    tokenizer.save_vocabulary(model_save_path)

    # 모델 평가
    accuracy, precision, recall, f1 = evaluate_model(model, val_loader)
    print(f"Epoch {epoch}: Validation Accuracy = {accuracy:.4f}")
    print(f"Epoch {epoch}: Validation Precision = {precision:.4f}")
    print(f"Epoch {epoch}: Validation Recall = {recall:.4f}")
    print(f"Epoch {epoch}: Validation F1 Score = {f1:.4f}")