import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_+.#/&-]*")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def tokenize_frame(df):
    resume_tokens = [tokenize(text) for text in df["resume_text"].fillna("")]
    job_tokens = [tokenize(text) for text in df["job_text"].fillna("")]
    return resume_tokens, job_tokens


def build_vocab(token_lists, max_vocab_size: int = 30000, min_freq: int = 1):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_vocab_size:
            break
    return vocab


def encode_tokens(tokens, vocab, max_len: int):
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[:max_len]]
    length = len(ids)
    if length < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - length))
    return ids, length


def lexical_overlap_features(resume_tokens, job_tokens):
    resume_set = set(resume_tokens)
    job_set = set(job_tokens)
    overlap = resume_set & job_set
    union = resume_set | job_set

    resume_len = max(len(resume_tokens), 1)
    job_len = max(len(job_tokens), 1)
    union_len = max(len(union), 1)

    return np.asarray(
        [
            len(overlap) / union_len,
            len(overlap) / max(len(resume_set), 1),
            len(overlap) / max(len(job_set), 1),
            len(resume_tokens) / max(job_len, 1),
            len(job_tokens) / max(resume_len, 1),
        ],
        dtype=np.float32,
    )


def load_pretrained_embeddings(embedding_path, vocab, embedding_dim: int):
    embedding_path = Path(embedding_path)
    scale = 0.05
    matrix = np.random.normal(0.0, scale, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab[PAD_TOKEN]] = np.zeros(embedding_dim, dtype=np.float32)

    found = 0
    with embedding_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split()
            if len(parts) != embedding_dim + 1:
                continue
            token = parts[0]
            idx = vocab.get(token)
            if idx is None:
                continue
            matrix[idx] = np.asarray(parts[1:], dtype=np.float32)
            found += 1

    coverage = found / max(len(vocab) - 2, 1)
    return matrix, coverage


class ResumeJobDataset(Dataset):
    def __init__(
        self,
        df,
        vocab,
        max_resume_len: int = 256,
        max_job_len: int = 128,
        struct_cols=None,
    ):
        self.resume_tokens, self.job_tokens = tokenize_frame(df)
        self.targets = df["matched_score"].astype(np.float32).to_numpy()
        self.vocab = vocab
        self.max_resume_len = max_resume_len
        self.max_job_len = max_job_len
        self.struct_cols = struct_cols or []
        self.lexical_features = np.asarray(
            [
                lexical_overlap_features(resume_tokens, job_tokens)
                for resume_tokens, job_tokens in zip(self.resume_tokens, self.job_tokens)
            ],
            dtype=np.float32,
        )
        if self.struct_cols:
            self.struct_features = (
                df[self.struct_cols].fillna(df[self.struct_cols].median()).astype(np.float32).to_numpy()
            )
        else:
            self.struct_features = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        resume_ids, resume_len = encode_tokens(self.resume_tokens[idx], self.vocab, self.max_resume_len)
        job_ids, job_len = encode_tokens(self.job_tokens[idx], self.vocab, self.max_job_len)

        item = {
            "resume_ids": torch.tensor(resume_ids, dtype=torch.long),
            "resume_len": torch.tensor(max(resume_len, 1), dtype=torch.long),
            "job_ids": torch.tensor(job_ids, dtype=torch.long),
            "job_len": torch.tensor(max(job_len, 1), dtype=torch.long),
            "lexical": torch.tensor(self.lexical_features[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }
        if self.struct_features is not None:
            item["struct"] = torch.tensor(self.struct_features[idx], dtype=torch.float32)
        return item


def collate_batch(batch):
    output = {
        "resume_ids": torch.stack([item["resume_ids"] for item in batch]),
        "resume_len": torch.stack([item["resume_len"] for item in batch]),
        "job_ids": torch.stack([item["job_ids"] for item in batch]),
        "job_len": torch.stack([item["job_len"] for item in batch]),
        "lexical": torch.stack([item["lexical"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }
    if "struct" in batch[0]:
        output["struct"] = torch.stack([item["struct"] for item in batch])
    return output


class SequenceEncoder(nn.Module):
    def __init__(self, embedding_layer, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, lengths):
        embedded = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.dropout(hidden[-1])


class PooledBiLSTMEncoder(nn.Module):
    def __init__(self, embedding_layer, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, lengths):
        embedded = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=token_ids.size(1))
        output = self.dropout(output)

        max_len = token_ids.size(1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        masked_output = output.masked_fill(~mask, 0.0)
        lengths_float = lengths.unsqueeze(1).clamp(min=1).to(output.dtype)
        mean_pool = masked_output.sum(dim=1) / lengths_float

        neg_inf = torch.full_like(output, float("-inf"))
        max_source = torch.where(mask, output, neg_inf)
        max_pool = max_source.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        return torch.cat([mean_pool, max_pool], dim=1)


class SiameseLSTMRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        dropout: float,
        embedding_matrix=None,
        freeze_embeddings: bool = False,
        struct_dim: int = 0,
    ):
        super().__init__()
        if embedding_matrix is not None:
            weight = torch.tensor(embedding_matrix, dtype=torch.float32)
            embedding_layer = nn.Embedding.from_pretrained(
                weight,
                freeze=freeze_embeddings,
                padding_idx=0,
            )
        else:
            embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if freeze_embeddings:
                embedding_layer.weight.requires_grad = False

        self.encoder = SequenceEncoder(embedding_layer, hidden_size, dropout)
        self.struct_dim = struct_dim
        pair_dim = hidden_size * 4 + struct_dim
        self.regressor = nn.Sequential(
            nn.Linear(pair_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, resume_ids, resume_len, job_ids, job_len, struct=None):
        resume_vec = self.encoder(resume_ids, resume_len)
        job_vec = self.encoder(job_ids, job_len)
        pair_features = [
            resume_vec,
            job_vec,
            torch.abs(resume_vec - job_vec),
            resume_vec * job_vec,
        ]
        if struct is not None:
            pair_features.append(struct)
        features = torch.cat(pair_features, dim=1)
        return self.regressor(features).squeeze(1)


class HybridSiameseBiLSTMRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        dropout: float,
        embedding_matrix=None,
        freeze_embeddings: bool = False,
        struct_dim: int = 0,
        lexical_dim: int = 5,
    ):
        super().__init__()
        if embedding_matrix is not None:
            weight = torch.tensor(embedding_matrix, dtype=torch.float32)
            embedding_layer = nn.Embedding.from_pretrained(
                weight,
                freeze=freeze_embeddings,
                padding_idx=0,
            )
        else:
            embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if freeze_embeddings:
                embedding_layer.weight.requires_grad = False

        self.encoder = PooledBiLSTMEncoder(embedding_layer, hidden_size, dropout)
        encoded_dim = hidden_size * 4
        pair_dim = encoded_dim * 4 + struct_dim + lexical_dim
        self.regressor = nn.Sequential(
            nn.Linear(pair_dim, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, resume_ids, resume_len, job_ids, job_len, lexical=None, struct=None):
        resume_vec = self.encoder(resume_ids, resume_len)
        job_vec = self.encoder(job_ids, job_len)
        pair_features = [
            resume_vec,
            job_vec,
            torch.abs(resume_vec - job_vec),
            resume_vec * job_vec,
        ]
        if lexical is not None:
            pair_features.append(lexical)
        if struct is not None:
            pair_features.append(struct)
        features = torch.cat(pair_features, dim=1)
        return self.regressor(features).squeeze(1)


@dataclass
class EpochResult:
    loss: float
    mse: float
    mae: float
    r2: float


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    target_mean = float(np.mean(y_true))
    total = float(np.sum((y_true - target_mean) ** 2))
    residual = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - residual / total if total > 0 else math.nan
    return {"mse": mse, "mae": mae, "r2": r2}


def run_epoch(model, dataloader, optimizer, criterion, device):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses = []
    y_true = []
    y_pred = []

    for batch in dataloader:
        resume_ids = batch["resume_ids"].to(device)
        resume_len = batch["resume_len"].to(device)
        job_ids = batch["job_ids"].to(device)
        job_len = batch["job_len"].to(device)
        lexical = batch.get("lexical")
        if lexical is not None:
            lexical = lexical.to(device)
        target = batch["target"].to(device)
        struct = batch.get("struct")
        if struct is not None:
            struct = struct.to(device)

        with torch.set_grad_enabled(train_mode):
            pred = model(
                resume_ids,
                resume_len,
                job_ids,
                job_len,
                lexical=lexical,
                struct=struct,
            )
            loss = criterion(pred, target)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        losses.append(float(loss.item()))
        y_true.extend(target.detach().cpu().numpy())
        y_pred.extend(pred.detach().cpu().numpy())

    metrics = regression_metrics(y_true, y_pred)
    return EpochResult(
        loss=float(np.mean(losses)),
        mse=metrics["mse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
    )


def predict(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            resume_ids = batch["resume_ids"].to(device)
            resume_len = batch["resume_len"].to(device)
            job_ids = batch["job_ids"].to(device)
            job_len = batch["job_len"].to(device)
            lexical = batch.get("lexical")
            if lexical is not None:
                lexical = lexical.to(device)
            target = batch["target"].to(device)
            struct = batch.get("struct")
            if struct is not None:
                struct = struct.to(device)

            pred = model(
                resume_ids,
                resume_len,
                job_ids,
                job_len,
                lexical=lexical,
                struct=struct,
            )
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs: int,
    patience: int,
    checkpoint_path,
    scheduler=None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, optimizer, criterion, device)
        val_result = run_epoch(model, val_loader, None, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_mse": train_result.mse,
                "train_mae": train_result.mae,
                "train_r2": train_result.r2,
                "val_loss": val_result.loss,
                "val_mse": val_result.mse,
                "val_mae": val_result.mae,
                "val_r2": val_result.r2,
            }
        )

        if scheduler is not None:
            scheduler.step(val_result.loss)

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history, best_epoch, checkpoint_path
