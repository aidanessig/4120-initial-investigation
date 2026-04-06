from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


PRIMARY_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 1146 / 7635
STRUCT_COLS = [
    "num_skills",
    "num_degrees",
    "num_positions",
    "experience_years",
    "age_min",
    "age_max",
    "skill_jaccard",
]
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_+.#/&-]*")
NGRAM_SIZES = (2, 3)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def _ngram_set(tokens: list[str], size: int) -> set[str]:
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[idx : idx + size]) for idx in range(len(tokens) - size + 1)}


def lexical_overlap_features(resume_text: str, job_text: str) -> np.ndarray:
    resume_tokens = tokenize(resume_text)
    job_tokens = tokenize(job_text)
    resume_set = set(resume_tokens)
    job_set = set(job_tokens)
    overlap = resume_set & job_set
    union = resume_set | job_set
    features = [
        len(overlap) / max(len(union), 1),
        len(overlap) / max(len(resume_set), 1),
        len(overlap) / max(len(job_set), 1),
        len(resume_tokens) / max(len(job_tokens), 1),
        len(job_tokens) / max(len(resume_tokens), 1),
    ]

    # Phrase-aware overlap helps preserve multi-token skill signals that are
    # blurred by unigram-only matching.
    for size in NGRAM_SIZES:
        resume_phrases = _ngram_set(resume_tokens, size)
        job_phrases = _ngram_set(job_tokens, size)
        phrase_overlap = resume_phrases & job_phrases
        phrase_union = resume_phrases | job_phrases
        features.extend(
            [
                len(phrase_overlap) / max(len(phrase_union), 1),
                len(phrase_overlap) / max(len(resume_phrases), 1),
                len(phrase_overlap) / max(len(job_phrases), 1),
            ]
        )

    return np.asarray(features, dtype=np.float32)


def _trim_text(text: str, max_words: int) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words])


def load_dataframe(data_path: str | Path, struct_cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.copy()
    df["source_row"] = np.arange(len(df), dtype=np.int32)
    for col in ["resume_text", "job_text", "combined_text"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    struct_cols = struct_cols or STRUCT_COLS
    for col in struct_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def prepare_splits(
    df: pd.DataFrame,
    seed: int = PRIMARY_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=seed)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class ResumeJobTransformerDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 256,
        resume_word_limit: int = 220,
        job_word_limit: int = 96,
        struct_cols: list[str] | None = None,
        struct_mean: np.ndarray | None = None,
        struct_std: np.ndarray | None = None,
        use_structured: bool = True,
        use_lexical: bool = True,
        truncation: str = "only_first",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.resume_word_limit = resume_word_limit
        self.job_word_limit = job_word_limit
        self.struct_cols = struct_cols or STRUCT_COLS
        self.struct_mean = struct_mean
        self.struct_std = struct_std
        self.use_structured = use_structured
        self.use_lexical = use_lexical
        self.truncation = truncation

        self.resume_text = self.df["resume_text"].fillna("").astype(str).tolist()
        self.job_text = self.df["job_text"].fillna("").astype(str).tolist()
        self.targets = self.df["matched_score"].astype(np.float32).to_numpy()

        if self.use_structured:
            struct_values = self.df[self.struct_cols].astype(np.float32).to_numpy()
            if self.struct_mean is not None and self.struct_std is not None:
                struct_values = (struct_values - self.struct_mean) / self.struct_std
            self.struct_features = struct_values
        else:
            self.struct_features = None

        if self.use_lexical:
            self.lexical_features = np.asarray(
                [
                    lexical_overlap_features(resume_text, job_text)
                    for resume_text, job_text in zip(self.resume_text, self.job_text)
                ],
                dtype=np.float32,
            )
        else:
            self.lexical_features = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        resume_text = _trim_text(self.resume_text[idx], self.resume_word_limit)
        job_text = _trim_text(self.job_text[idx], self.job_word_limit)
        encoded = self.tokenizer(
            resume_text,
            job_text,
            truncation=self.truncation,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        if self.struct_features is not None:
            item["struct"] = torch.tensor(self.struct_features[idx], dtype=torch.float32)
        if self.lexical_features is not None:
            item["lexical"] = torch.tensor(self.lexical_features[idx], dtype=torch.float32)
        return item


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    encoder_name: str,
    batch_size: int = 16,
    max_length: int = 256,
    resume_word_limit: int = 220,
    job_word_limit: int = 96,
    struct_cols: list[str] | None = None,
    use_structured: bool = True,
    use_lexical: bool = True,
    truncation: str = "only_first",
) -> tuple[DataLoader, DataLoader, DataLoader, object]:
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    struct_cols = struct_cols or STRUCT_COLS
    struct_mean = None
    struct_std = None
    if use_structured:
        train_struct = train_df[struct_cols].astype(np.float32).to_numpy()
        struct_mean = train_struct.mean(axis=0)
        struct_std = train_struct.std(axis=0)
        struct_std = np.where(struct_std < 1e-6, 1.0, struct_std)

    dataset_kwargs = {
        "tokenizer": tokenizer,
        "max_length": max_length,
        "resume_word_limit": resume_word_limit,
        "job_word_limit": job_word_limit,
        "struct_cols": struct_cols,
        "struct_mean": struct_mean,
        "struct_std": struct_std,
        "use_structured": use_structured,
        "use_lexical": use_lexical,
        "truncation": truncation,
    }
    train_dataset = ResumeJobTransformerDataset(train_df, **dataset_kwargs)
    val_dataset = ResumeJobTransformerDataset(val_df, **dataset_kwargs)
    test_dataset = ResumeJobTransformerDataset(test_df, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, tokenizer


class HybridTransformerRegressor(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        dropout: float = 0.2,
        hidden_dim: int = 256,
        struct_dim: int = 0,
        lexical_dim: int = 0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        input_dim = encoder_dim + struct_dim + lexical_dim
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        lexical: torch.Tensor | None = None,
        struct: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        token_embeddings = outputs.last_hidden_state
        pair_vec = mean_pool(token_embeddings, attention_mask)

        features = [pair_vec]
        if lexical is not None:
            features.append(lexical)
        if struct is not None:
            features.append(struct)
        merged = torch.cat(features, dim=1)
        return self.regressor(self.dropout(merged)).squeeze(1)


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@dataclass
class EpochResult:
    loss: float
    mse: float
    mae: float
    r2: float


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    target_mean = float(np.mean(y_true))
    total = float(np.sum((y_true - target_mean) ** 2))
    residual = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - residual / total if total > 0 else math.nan
    return {"mse": mse, "mae": mae, "r2": r2}


def run_epoch(model, dataloader, optimizer, criterion, device, scheduler=None) -> EpochResult:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses = []
    y_true = []
    y_pred = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        lexical = batch.get("lexical")
        if lexical is not None:
            lexical = lexical.to(device)
        struct = batch.get("struct")
        if struct is not None:
            struct = struct.to(device)
        target = batch["target"].to(device)

        with torch.set_grad_enabled(train_mode):
            pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                lexical=lexical,
                struct=struct,
            )
            loss = criterion(pred, target)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

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


@torch.no_grad()
def predict(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        lexical = batch.get("lexical")
        if lexical is not None:
            lexical = lexical.to(device)
        struct = batch.get("struct")
        if struct is not None:
            struct = struct.to(device)

        pred = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            lexical=lexical,
            struct=struct,
        )
        y_true.extend(batch["target"].cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 6,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 2,
    warmup_ratio: float = 0.1,
):
    model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.regressor.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    total_steps = max(len(train_loader) * epochs, 1)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_state = None
    best_val_loss = float("inf")
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, optimizer, criterion, device, scheduler=scheduler)
        val_result = run_epoch(model, val_loader, optimizer=None, criterion=criterion, device=device)
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
                "encoder_lr": float(optimizer.param_groups[0]["lr"]),
                "head_lr": float(optimizer.param_groups[1]["lr"]),
            }
        )

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    return model, history_df


def build_analysis_frame(test_df: pd.DataFrame, y_true, y_pred, encoder: str, variant: str) -> pd.DataFrame:
    analysis_df = test_df[["source_row", "resume_text", "job_text", "matched_score"]].copy()
    analysis_df["predicted_score"] = np.asarray(y_pred, dtype=np.float32)
    analysis_df["residual"] = np.asarray(y_true, dtype=np.float32) - analysis_df["predicted_score"]
    analysis_df["abs_error"] = analysis_df["residual"].abs()
    analysis_df["encoder"] = encoder
    analysis_df["variant"] = variant
    return analysis_df


def summarize_token_lengths(df: pd.DataFrame) -> pd.DataFrame:
    resume_lengths = df["resume_text"].fillna("").map(lambda x: len(tokenize(x)))
    job_lengths = df["job_text"].fillna("").map(lambda x: len(tokenize(x)))
    return pd.DataFrame(
        {
            "resume_tokens": resume_lengths,
            "job_tokens": job_lengths,
        }
    )


def run_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    encoder_name: str,
    use_structured: bool,
    use_lexical: bool,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 256,
    resume_word_limit: int = 220,
    job_word_limit: int = 96,
    epochs: int = 6,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 2,
    warmup_ratio: float = 0.1,
    seed: int = PRIMARY_SEED,
    truncation: str = "only_first",
) -> dict[str, object]:
    set_seed(seed)
    train_loader, val_loader, test_loader, _ = make_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        encoder_name=encoder_name,
        batch_size=batch_size,
        max_length=max_length,
        resume_word_limit=resume_word_limit,
        job_word_limit=job_word_limit,
        use_structured=use_structured,
        use_lexical=use_lexical,
        truncation=truncation,
    )
    sample_batch = next(iter(train_loader))
    lexical_dim = sample_batch["lexical"].shape[1] if use_lexical else 0
    struct_dim = sample_batch["struct"].shape[1] if use_structured else 0

    model = HybridTransformerRegressor(
        encoder_name=encoder_name,
        dropout=0.2,
        hidden_dim=256,
        lexical_dim=lexical_dim,
        struct_dim=struct_dim,
    )
    model, history_df = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        patience=patience,
        warmup_ratio=warmup_ratio,
    )
    y_true, y_pred = predict(model, test_loader, device=device)
    test_metrics = regression_metrics(y_true, y_pred)
    variant = "hybrid" if (use_structured or use_lexical) else "text-only"
    analysis_df = build_analysis_frame(test_df, y_true, y_pred, encoder=encoder_name, variant=variant)

    return {
        "model": model,
        "history": history_df,
        "test_metrics": test_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "analysis_df": analysis_df,
        "variant": variant,
        "encoder": encoder_name,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "seed": seed,
    }
