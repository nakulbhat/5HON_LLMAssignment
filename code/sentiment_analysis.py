import math
import random
import csv
import os
from typing import List, Tuple, Dict
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import csv

INPUT_FILE = "../data/raw_data/sent_analyis/test.csv"
OUTPUT_FILE = "../data/processed_data/sent_analysis.csv"

def preprocess_data():
    with open(INPUT_FILE, "r", encoding="latin-1") as infile, open(OUTPUT_FILE, "w", encoding="latin-1") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            if row[2] =="sentiment":
                writer.writerow([row[1], row[2]])
            else:
                writer.writerow([row[1], 0 if row[2] == "negative" or row[2] == "neutral" else 1])

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    punct = ".,!?;:()[]\"'"
    for p in punct:
        text = text.replace(p, f' {p} ')
    tokens = text.strip().split()
    return tokens

class Vocab:
    def __init__(self, min_freq: int = 1, max_size: int = None, specials: List[str] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.counter = Counter()
        self.itos = []
        self.stoi = {}
        self.specials = specials or ['<pad>', '<unk>']
        for s in self.specials:
            self.add_token_to_map(s)

    def add_token_to_map(self, token):
        if token not in self.stoi:
            idx = len(self.itos)
            self.itos.append(token)
            self.stoi[token] = idx

    def build_from_texts(self, texts: List[str]):
        for text in texts:
            tokens = simple_tokenize(text)
            self.counter.update(tokens)
        items = [t for t, c in self.counter.items() if c >= self.min_freq]
        items.sort(key=lambda x: (-self.counter[x], x))
        if self.max_size:
            items = items[: self.max_size - len(self.specials)]
        for token in items:
            self.add_token_to_map(token)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi['<unk>']) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if i < len(self.itos) else '<unk>' for i in ids]


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocab, max_len: int = 128):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = simple_tokenize(self.texts[idx])
        token_ids = self.vocab.encode(tokens)[: self.max_len]
        label = int(self.labels[idx])
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_batch(batch):
    token_lists, label_list = zip(*batch)
    lengths = [len(t) for t in token_lists]
    max_len = max(lengths) if lengths else 0
    pad_idx = 0  # index of <pad> in Vocab by construction
    batch_size = len(token_lists)
    src = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, t in enumerate(token_lists):
        l = t.size(0)
        src[i, :l] = t
        mask[i, :l] = 1
    labels = torch.stack(label_list)
    return src, labels, mask  # mask: True for real tokens


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)  # not a parameter

    def forward(self, x: torch.Tensor):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation='relu', batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        device = src.device
        batch_size, seq_len = src.size()
        emb = self.embedding(src) * math.sqrt(self.embed_dim)  # (batch, seq, embed)
        emb = emb.transpose(0, 1)  # (seq, batch, embed)
        emb = self.pos_encoder(emb)  # seq, batch, embed

        padding_mask = ~src_mask  # True for padding
        memory = self.transformer_encoder(emb, src_key_padding_mask=padding_mask)
        memory = memory.transpose(0, 1)  # (batch, seq, embed)

        mask_float = src_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        summed = (memory * mask_float).sum(dim=1)  # (batch, embed)
        lengths = mask_float.sum(dim=1).clamp(min=1e-6)  # (batch, 1)
        pooled = summed / lengths  # (batch, embed)

        logits = self.fc(pooled)  # (batch, num_classes)
        return logits


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, average: str = 'binary'):
    if preds.dim() == 2:
        pred_labels = preds.argmax(dim=1)
    else:
        pred_labels = preds
    labels = labels.to(pred_labels.device)
    correct = (pred_labels == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total if total > 0 else 0.0

    num_classes = int(max(pred_labels.max().item(), labels.max().item()) + 1)
    precisions = []
    recalls = []
    f1s = []
    for c in range(num_classes):
        tp = ((pred_labels == c) & (labels == c)).sum().item()
        fp = ((pred_labels == c) & (labels != c)).sum().item()
        fn = ((pred_labels != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    macro_prec = sum(precisions) / num_classes
    macro_rec = sum(recalls) / num_classes
    macro_f1 = sum(f1s) / num_classes

    return {
        'accuracy': accuracy,
        'precision_macro': macro_prec,
        'recall_macro': macro_rec,
        'f1_macro': macro_f1
    }

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(dataloader, desc='train', leave=False)
    for src, labels, mask in pbar:
        src = src.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits = model(src, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * src.size(0)
        all_preds.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(preds, labels)
    metrics['loss'] = avg_loss
    return metrics

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='eval', leave=False)
        for src, labels, mask in pbar:
            src = src.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            logits = model(src, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * src.size(0)
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(preds, labels)
    metrics['loss'] = avg_loss
    return metrics

def load_csv_dataset(path: str, text_col='text', label_col='sentiment') -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row[text_col])
            labels.append(int(row[label_col]))
    return texts, labels

def run_training(
    csv_path: str = None,
    text_col='text',
    label_col='sentiment',
    out_dir: str = './model_out',
    num_epochs: int = 6,
    batch_size: int = 64,
    max_len: int = 128,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 512,
    num_layers: int = 2,
    learning_rate: float = 2e-4,
    num_classes: int = 2,
    device: str = None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    if csv_path and os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}")
        texts, labels = load_csv_dataset(csv_path, text_col=text_col, label_col=label_col)
    else:
        raise Exception("file not found")

    # Split
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts = list(texts)
    labels = list(labels)
    n = len(texts)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_texts = texts[:n_train]
    train_labels = labels[:n_train]
    val_texts = texts[n_train:n_train+n_val]
    val_labels = labels[n_train:n_train+n_val]
    test_texts = texts[n_train+n_val:]
    test_labels = labels[n_train+n_val:]

    # Build vocab from train texts
    vocab = Vocab(min_freq=1, max_size=30000)
    vocab.build_from_texts(train_texts)
    print(f"Vocab size: {len(vocab)}")

    train_ds = TextClassificationDataset(train_texts, train_labels, vocab, max_len=max_len)
    val_ds = TextClassificationDataset(val_texts, val_labels, vocab, max_len=max_len)
    test_ds = TextClassificationDataset(test_texts, test_labels, vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = TransformerClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.1,
        max_len=max_len,
        pad_idx=vocab.stoi['<pad>'],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['loss'])

        print(f"  train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.4f} f1: {train_metrics['f1_macro']:.4f}")
        print(f"  val   loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.4f} f1: {val_metrics['f1_macro']:.4f}")


    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_metrics['loss']:.4f} acc: {test_metrics['accuracy']:.4f} f1: {test_metrics['f1_macro']:.4f}")
    return model, vocab, test_metrics

if __name__ == '__main__':
    model, vocab, test_metrics = run_training(
        csv_path=OUTPUT_FILE,

        # --- Make training VERY fast and VERY bad ---
        num_epochs=1,          # Only one epoch
        batch_size=32,         # Smaller batches
        max_len=32,            # Short sequences
        embed_dim=16,          # Tiny embeddings
        num_heads=1,           # Minimal attention
        ff_dim=32,             # Tiny feed-forward
        num_layers=1,          # Only one Transformer layer
        learning_rate=1e-3,    # Fast, sloppy training
        num_classes=2,
        out_dir='./model_out_fast_test'
    )
