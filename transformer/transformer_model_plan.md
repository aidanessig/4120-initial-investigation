# Transformer Model Plan

This document outlines our plan for building and evaluating a **Transformer Model** for resume-job compatibility prediction. The goal of this final model is to improve on the current Hybrid BiLSTM by directly addressing the limitations identified in the earlier recurrent experiments while preserving the structured and lexical signals that have already proven useful in this project.

## 1. Motivation

Our earlier experiments showed a clear progression:

- simple structured-only regression performed poorly
- TF-IDF text baselines provided a strong improvement
- the plain LSTM underperformed badly relative to TF-IDF
- the Hybrid BiLSTM became the strongest model by combining better sequence encoding with lexical and structured features

The LSTM model report suggests that the plain LSTM failed for three main reasons:

1. It summarized long resume and job texts too aggressively using only a final hidden state.
2. It encoded the resume and job description independently, which weakened direct matching.
3. It relied on word-level embeddings with limited vocabulary coverage.

Although the Hybrid BiLSTM improved substantially by adding bidirectionality, pooling, overlap features, and structured features, it still retains one major structural limitation: the resume and job description are encoded separately and are only compared after both representations are already compressed. For a compatibility task, this is not ideal. Resume-job matching is fundamentally about whether requirements in one text align with evidence in the other text. A transformer is a natural next step because it allows the two texts to interact throughout the encoding process rather than only at the end.

## 2. Proposed Final Architecture

We propose a **Transformer Model** that treats the resume and job description as a single paired input and predicts a continuous compatibility score.

### Core Input Format

The main transformer input will be:

`[CLS] resume_text [SEP] job_text [SEP]`

This design allows the model to process both texts jointly. Instead of generating two separate vectors and comparing them afterward, the model can use self-attention to learn direct token-level and phrase-level relationships between the two sides.

This is theoretically well-motivated for the task because compatibility depends on alignment patterns such as:

- whether job-required skills appear in the resume
- whether education requirements match degree information
- whether experience phrases in the resume correspond to the stated experience requirement
- whether semantically related terms align even when phrased differently

This directly addresses the weakness in the plain LSTM design, which emphasized independent semantic encoding more than explicit semantic matching.

### Text Encoder

The text backbone will be a pretrained transformer encoder. The exact pretrained checkpoint can be selected based on computational feasibility, but the model should satisfy the following requirements:

- bidirectional contextual encoding
- subword tokenization
- support for paired-text classification or regression
- straightforward fine-tuning in PyTorch or Hugging Face

A pretrained encoder is strongly justified here. Earlier work showed that pretrained word-vector coverage from the relativized DOLMA embeddings was only about 64%, which indicates that many domain-specific terms were either missing or poorly represented in the word-level pipeline. Transformer tokenizers use subword units, which substantially reduce the out-of-vocabulary problem. Even if a specialized skill or framework is not present as a single token, it can still be decomposed and represented compositionally.

This is especially important in a resume screening setting, where rare technologies, abbreviations, and domain terms often carry significant signal.

### Encoder Comparison Plan

We also want to explicitly test how the choice of text encoder affects downstream performance. This is worth studying because one of the clearest lessons from the earlier experiments was that representation quality matters: TF-IDF beat the plain LSTM, and the stronger Hybrid BiLSTM improved only after the text representation and matching pipeline became more expressive.

Our encoder study should proceed in stages:

- first, use the original relativized DOLMA-based embedding setup as the historical reference point
- next, test `sentence-transformers/all-MiniLM-L6-v2` as a stronger compact pretrained text encoder

The reason to include this comparison is that it helps answer a more focused research question than "did the transformer work?" It lets us ask whether gains come mainly from the transformer architecture itself, from better pretrained representations, or from both.

The DOLMA setup remains important because it reflects the original project direction and provides continuity with the earlier neural models. However, the earlier report also showed only about 64% embedding coverage, which gives strong reason to believe that a modern subword-based pretrained encoder such as `all-MiniLM-L6-v2` could represent domain vocabulary more effectively.

In practice, this means we should treat encoder choice as an experimental variable rather than a fixed assumption.

### Pair Representation

After joint encoding, the model will extract a fixed-size representation of the resume-job pair. The most direct approach is to use the contextualized representation associated with the `[CLS]` token. This vector will serve as the global text-pair summary.

The reason for using a learned pair-level summary is that the transformer has already allowed information from the resume and job description to interact during encoding. The resulting representation is therefore not merely a generic summary of one document, but a compatibility-aware representation shaped by the entire pair.

### Structured and Lexical Features

The transformer output alone should not be the only signal used for prediction. Earlier experiments showed that structured features were weak in isolation but helpful in combination with text. The Hybrid Siamese BiLSTM improved precisely because it did not ignore these side signals.

We therefore plan to append engineered features to the transformer pair representation before the final regression head. These features should include:

- `num_skills`
- `num_degrees`
- `num_positions`
- `num_institutions`
- `experience_years`
- `age_min`
- `age_max`
- lexical overlap features already used in the LSTM pipeline

We should also consider adding improved phrase-aware overlap features, since the data cleaning report noted that exact skill Jaccard was very low partly because resume skills were often multi-word phrases while job skills were represented at a different granularity. A phrase-aware overlap representation would be more faithful to the task than naive token overlap alone.

The justification for retaining these features is not only empirical but also conceptual. Resume-job compatibility is not a pure free-text semantic task. It also depends on interpretable side signals such as experience thresholds, approximate credential counts, and concrete overlap in technical vocabulary.

### Regression Head

The final model will concatenate:

- transformer pair representation
- lexical features
- structured numerical features

This combined vector will be passed into a feedforward regression head with dropout and nonlinearities, and the output will be a single scalar prediction for `matched_score`.

This retains the successful hybrid idea from the BiLSTM model while replacing the recurrent text encoder with a stronger matching-oriented backbone.

## 3. Handling Long Resume Text

One practical challenge is input length. Resumes are often much longer than job descriptions, and a standard transformer has a limited maximum token window.

This is not a minor engineering detail; it is directly related to one of the main weaknesses identified in the LSTM experiments. The recurrent baseline lost information by compressing long inputs too aggressively, and a poorly designed transformer pipeline could recreate the same problem if it simply truncates away too much of the resume.

Our implementation plan should therefore explicitly account for length.

### Preferred Strategy: Targeted Truncation

The first implementation should use a disciplined truncation policy rather than arbitrary cutting. For example:

- allocate a larger token budget to the resume than to the job description
- preserve the beginning of the resume, where summary and top-level skills often appear
- preserve sections likely to contain strong evidence, such as skills, positions, and responsibilities

This is justified because not all resume text is equally informative. If we must truncate for computational reasons, we should prioritize the sections most relevant to compatibility rather than treating all tokens as equally valuable.

### Extended Strategy: Chunked Resume Encoding

If targeted truncation still appears too lossy, a stronger version of the architecture would split the resume into chunks and pair each chunk with the full job description. Each chunk-job pair would be encoded separately, and the chunk-level outputs would then be aggregated using attention or mean pooling before the regression head.

This chunked design is theoretically attractive because it allows the model to inspect more of the resume without forcing the entire document into one short window. It also mirrors the linguistic reality of the task: evidence for a match may be spread across different parts of the resume rather than concentrated near the beginning.

The chunked approach is more complex, so it should be treated as an upgrade path if the simpler implementation underperforms.

## 4. Training Plan

### Compute Environment

We should plan to run the transformer experiments in **Google Colab** so that training can take advantage of GPU acceleration.

This is an important practical choice rather than a convenience detail. Transformer fine-tuning is substantially more expensive than the earlier regression baselines and more computationally demanding than static embedding experiments. Running in Colab gives us access to GPU-backed training without requiring specialized local hardware, and it makes it more realistic to run the main transformer model, ablations, and limited multi-seed experiments within project time constraints.

Using Colab is especially justified if we test `all-MiniLM-L6-v2` or a chunked resume setup, since both tokenization and forward passes become more expensive than the earlier DOLMA-based pipeline.

### Data Split

To ensure fair comparison with earlier models, we should preserve the same train, validation, and test partitioning used in the Siamese model experiments:

- train: 6,489
- validation: 1,146
- test: 1,909

This is important because the entire point of the final model is to compare against the established baselines under the same evaluation conditions. Changing the split would make the comparison less credible.

### Objective Function

The target is continuous `matched_score`, so the model will be trained as a regression system. A robust regression loss such as `SmoothL1Loss` is a strong default because it already worked well in the Hybrid Siamese BiLSTM and is less sensitive than plain MSE to a small number of larger residuals.

This choice is justified by continuity with previous successful training behavior. Unless there is a specific reason to change it, the final model should inherit stable optimization choices that already produced strong results.

### Optimization

We should fine-tune the transformer using:

- `AdamW`
- a small learning rate for pretrained transformer parameters
- optionally a slightly larger learning rate for the new regression head
- learning-rate scheduling based on validation performance
- early stopping based on validation loss

This training strategy follows standard transformer fine-tuning practice and is also consistent with prior project experience. The earlier hybrid recurrent model already benefited from `AdamW`, regularization, and validation-aware scheduling. Since transformers are even more sensitive to optimization settings than shallow regression baselines, stable fine-tuning is especially important.

### Regularization

The model should include:

- dropout on the final regression head
- optional weight decay through `AdamW`
- early stopping

This is important because the dataset has only 9,544 examples, which is moderate but not large for transformer fine-tuning. A final model with too much capacity and too little regularization could overfit the training set while appearing promising early in development.

## 5. Implementation Steps

The implementation plan should proceed in the following order.

### Step 1: Reuse the Existing Cleaned Data Pipeline

Use `resume_data_cleaned.csv` and preserve the established text fields:

- `resume_text`
- `job_text`
- `matched_score`

Reuse existing structured features when possible rather than rebuilding them from scratch. This keeps the final model directly connected to the cleaned data pipeline already documented in the project.

This is justified because the project has already invested substantial effort in feature cleaning and extraction, and the strongest existing model explicitly benefited from those features.

### Step 2: Build a Transformer Dataset Class

Create a new dataset class that:

- tokenizes paired inputs as resume-job text pairs
- returns input IDs, attention masks, and optional token type IDs if supported by the backbone
- returns lexical and structured feature vectors
- returns the scalar regression target

This step mirrors the existing LSTM data pipeline but updates it for transformer-style input formatting.

### Step 3: Implement the Hybrid Transformer Regressor

Create a model module with:

- pretrained transformer encoder
- pair-level text representation extraction
- concatenation with lexical and structured features
- MLP regression head

This design preserves the strongest lesson from the BiLSTM experiments: text-only modeling was not enough, but text plus explicit side information was strong.

### Step 3a: Implement the Encoder Comparison Setup

Before locking in a final backbone, we should structure the code so that the text encoder can be swapped cleanly.

The initial comparison should include:

- a DOLMA-based reference setup connected to the original embedding pipeline
- an `all-MiniLM-L6-v2` transformer-based setup

This modularity is important because it allows us to compare encoders under the same preprocessing, same split, and same regression framework. Without that control, it would be difficult to tell whether performance differences came from encoder quality or from unrelated pipeline changes.

Even if the final model is transformer-based, keeping the DOLMA comparison in the plan is valuable because it ties the final conclusions back to the original project hypothesis about pretrained semantic representations.

### Step 4: Train a Text-Only Transformer Baseline

Before training the full hybrid version, train a text-only transformer regressor using only the paired text input.

This is important for interpretability. If the hybrid model improves, we want to know whether the gain comes from the transformer backbone itself, the appended side features, or both. Running a text-only transformer first gives a cleaner ablation.

### Step 5: Train the Full Hybrid Transformer

After confirming the text-only setup works, train the full model with lexical and structured features appended.

This gives the final architecture the best chance to outperform the current BiLSTM while still allowing us to explain where the gains come from.

### Step 6: Analyze Error Cases

After training, inspect examples where:

- the transformer is much better than the BiLSTM
- the transformer is much worse than the BiLSTM
- predictions are consistently too high or too low

This analysis is useful both scientifically and for final reporting. If the transformer succeeds, we can explain what kinds of matching behavior it captured better. If it fails, we can explain whether the issue was truncation, data size, noisy targets, or overfitting.

## 6. Benchmarking Plan

The transformer model should be benchmarked under the same framework as the earlier models so that comparisons remain meaningful.

### Primary Comparison Set

The main benchmarks should be:

- structured-only regression
- TF-IDF separate resume/job regression
- TF-IDF combined-text regression
- hybrid TF-IDF + structured regression
- plain LSTM
- Hybrid BiLSTM
- Transformer Model

This comparison is necessary because the project narrative is not simply about whether one neural model beats another. It is about tracing the progression from simple baselines to more expressive semantic matching systems.

### Evaluation Metrics

We should report at minimum:

- `R²`
- `MSE`
- `MAE`

These are already established in the project, so continuing with them ensures direct comparability. `R²` should remain the headline metric because it has been the main measure used throughout the project.

### Ablation Benchmarks

To understand why the transformer performs as it does, we should run a small ablation study:

- text-only transformer
- transformer + lexical features
- transformer + structured features
- full transformer hybrid

### Encoder Benchmarks

In addition to architectural ablations, we should also benchmark the effect of encoder choice. At minimum, the plan should compare:

- original DOLMA-based text representation
- `all-MiniLM-L6-v2`

Ideally, these comparisons should be run under the same train/validation/test split and with as much of the surrounding pipeline held fixed as possible.

This benchmark is important because it isolates a core project question: whether better pretrained text representations alone can improve compatibility prediction, or whether the main gains require the full joint transformer matching architecture.

If `all-MiniLM-L6-v2` clearly outperforms the DOLMA-based setup, that would support the argument that subword-based pretrained encoders are better suited to resume-job matching than static word vectors with limited vocabulary coverage.

This is especially important because the BiLSTM results already showed that hybridization matters. If the final model improves, the ablation study will let us explain whether the gain came primarily from better joint text encoding or from the combination of that encoding with side information.

### Robustness Across Seeds

If computationally feasible, train the final transformer with multiple random seeds.

This is justified by the earlier hybrid BiLSTM workflow, which already evaluated several seeds. Since fine-tuned neural models can vary noticeably with initialization and batch order, a multi-seed evaluation makes the final result more credible and less likely to be a lucky run.

### Fairness of Comparison

To make the benchmark defensible, we should keep the following fixed wherever possible:

- same train/validation/test split
- same cleaned dataset
- same target variable
- same reported metrics

This matters because the strength of the final argument depends not only on the transformer score itself, but on whether that score can be interpreted as a genuine architectural improvement.

## 7. What Success Would Look Like

The current best model is the Hybrid BiLSTM with test performance around:

- `R² = 0.6679`
- `MSE = 0.0092`
- `MAE = 0.0745`

The Transformer Model should be considered successful if it:

- clearly exceeds the BiLSTM on `R²`
- does so consistently across validation and test evaluation
- shows qualitatively better handling of skill alignment, qualification matching, and domain-specific vocabulary

A result in the `0.80-0.90` range would be an excellent outcome, but it should be treated as an aspirational target rather than an assumption. Given the dataset size and possible noise in `matched_score`, even a smaller but credible improvement over the BiLSTM would still be a strong final result.

## 8. Final Rationale

The Transformer Model is the most justified final architecture because it is not merely a larger version of the previous recurrent models. It is a model whose inductive bias is better aligned with the structure of the task.

The earlier experiments already taught us several important lessons:

- long resume-job texts should not be compressed too aggressively
- exact and near-exact matching matters, not just broad semantic similarity
- structured side information helps when combined with text
- domain vocabulary coverage is a real issue

The proposed transformer architecture responds directly to all of these points. It enables joint encoding and token-level interaction, reduces vocabulary brittleness through subword tokenization, and can still incorporate the lexical and numeric side features that have already proven useful. For these reasons, it represents the most theoretically and empirically grounded final model direction for the project.
