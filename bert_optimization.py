# ========== 1. Carregar e preparar os dados ==========
import pandas as pd
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

df_fake = pd.read_csv("Data/Fake.csv")
df_true = pd.read_csv("Data/True.csv")

df_fake['label'] = 0
df_true['label'] = 1

def limpar_prefixo_agencia(texto):
    return re.sub(r"^(.*?\(Reuters\)[\s\-–]*)", "", texto)

df_true['text'] = df_true['text'].apply(limpar_prefixo_agencia)

df = pd.concat([df_fake, df_true], ignore_index=True)
df = shuffle(df).drop_duplicates(subset='text').reset_index(drop=True)

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

# ========== 2. Tokenização BERT ==========
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

train_encodings = tokenize_data(train_texts)
val_encodings = tokenize_data(val_texts)

# ========== 3. Dataset PyTorch ==========
import torch
from torch.utils.data import Dataset, DataLoader

class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = FakeNewsDataset(train_encodings, train_labels)
val_dataset = FakeNewsDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ========== 4. Modelo ==========
import torch.nn as nn
from transformers import BertModel

class FakeBERT_CNN(nn.Module):
    def __init__(self, dropout=0.3, out_channels=128, kernel_size=3):
        super(FakeBERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.conv1 = nn.Conv1d(768, out_channels, kernel_size=kernel_size, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)

# ========== 5. Treino e avaliação ==========
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_eval(model, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()

    for batch in tqdm(train_loader, desc="Treino", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            preds += (probs > 0.5).cpu().tolist()
            targets += labels.cpu().tolist()

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    return acc, f1

# ========== 6. Testar várias combinações ==========
import itertools

dropouts = [0.1, 0.3]
out_channels_list = [64, 128]
kernel_sizes = [3, 5]
learning_rates = [2e-5, 5e-5]

melhor_config = None
melhor_f1 = 0

print("\n🔍 A testar combinações de hiperparâmetros...")

for dropout, out_channels, kernel_size, lr in itertools.product(dropouts, out_channels_list, kernel_sizes, learning_rates):
    print(f"\n➡️ dropout={dropout}, out_channels={out_channels}, kernel={kernel_size}, lr={lr}")
    model = FakeBERT_CNN(dropout=dropout, out_channels=out_channels, kernel_size=kernel_size)
    acc, f1 = train_and_eval(model, lr)
    print(f"📊 Accuracy: {acc:.4f} | F1: {f1:.4f}")

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_config = (dropout, out_channels, kernel_size, lr)

print(f"\n🏆 Melhor configuração encontrada:")
print(f"   dropout={melhor_config[0]}, out_channels={melhor_config[1]}, kernel={melhor_config[2]}, lr={melhor_config[3]}")
print(f"   Melhor F1: {melhor_f1:.4f}")
