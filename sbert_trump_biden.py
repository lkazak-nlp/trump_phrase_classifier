import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5

df = pd.read_csv("trump_biden_balanced.csv")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, return_tensors="pt")
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class SBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits

model = SBERTClassifier(MODEL_NAME).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# === Оценка ===
from sklearn.metrics import classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=["Not Trump", "Trump"]))


def predict_trump_quote(text):
    model.eval()
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        confidence = probs[pred]

        print(f"Фраза: {text}")
        print(f"Вердикт: {'Трамп' if pred == 1 else 'Не Трамп'} (уверенность: {confidence:.2f})")


# Примеры
predict_trump_quote("We will make USA great again!")
predict_trump_quote("We will make America great again!")
predict_trump_quote("We will make yankees cool!")

predict_trump_quote("Fuck Hillary")
predict_trump_quote("Hillary is a good girl")

predict_trump_quote("Hillary is a bad girl")
predict_trump_quote("Hillary is a bad president")

predict_trump_quote("Putin said I'm a genius!!!")

predict_trump_quote("We lead not by the example of our power, but by the power of our example.")
predict_trump_quote("Folks, this is not who we are as Americans.")
predict_trump_quote("We’re in a battle for the soul of this nation.")
