# Siamese Model Report

This report summarizes the saved outputs from:

- `lstm_experiment.ipynb`
- `bilstm_hybrid_experiment.ipynb`

The goal of both notebooks was to model compatibility between a resume and a job description as a continuous regression problem. Both experiments used the same cleaned dataset of 9,544 resume-job pairs, with:

- Train rows: 6,489
- Validation rows: 1,146
- Test rows: 1,909

The shared training vocabulary contained 3,904 tokens, and pretrained embedding coverage from the relativized DOLMA file was 64.02%.

## 1. Baseline LSTM

### Architecture

The first notebook implemented a basic LSTM regressor:

- Shared embedding layer
- Shared one-direction LSTM encoder
- Final hidden state used as the representation for each text
- Pairwise comparison features:
  - `resume_vec`
  - `job_vec`
  - `abs(resume_vec - job_vec)`
  - `resume_vec * job_vec`
- MLP regression head

This model keeps the core Siamese idea: both texts are passed through the same encoder so that resume and job description are embedded into the same learned semantic space.

### Result

The basic LSTM selected its best checkpoint at epoch 13 and achieved:

- R²: `0.2910`
- MSE: `0.0196`
- MAE: `0.1110`

Relative to earlier baselines, this was:

- better than structured-only regression (`R² = 0.0939`)
- much worse than all of the TF-IDF based text baselines
- much worse than the earlier hybrid TF-IDF + structured model (`R² = 0.5256`)

### High-Level Interpretation

At a conceptual level, the plain Siamese LSTM likely underperformed because it was too simple for this task.

First, it used only the **final hidden state** of a one-direction LSTM as the summary of an entire resume or job description. In NLP terms, this means the model compresses a long sequence into one vector by relying heavily on the last part of the recurrent state. That can work for short sequences, but resumes are long, noisy, and contain many scattered signals such as skills, positions, technologies, education, and domain terms. Important information may appear anywhere in the sequence, not just near the end. A single final state is therefore a weak bottleneck.

Second, the baseline LSTM treated the problem mostly as **semantic encoding**, but not strongly enough as **semantic matching**. It encoded each side independently, then compared the two vectors only at the end. That design may miss finer-grained compatibility cues such as direct skill overlap, partial terminology matches, or alignment between specific qualifications and job requirements.

Third, the model depended mostly on word embeddings and recurrent composition, but your dataset already showed that **structured features matter**. The plain LSTM ignored useful side information such as counts of skills, degrees, positions, experience years, and overlap-based features. In other words, it threw away signals that earlier baselines had already shown to be predictive.

So the main lesson from the first notebook is not that Siamese models are bad, but that a minimal LSTM encoder was not expressive enough for this resume-job matching problem.

## 2. Hybrid Siamese BiLSTM

### Architecture Upgrades

The second notebook upgraded the model in several important ways:

- Shared pretrained embedding layer
- Shared **bidirectional** LSTM encoder
- **Mean pooling** and **max pooling** over sequence outputs
- Additional **lexical overlap features**
- Structured numeric features appended to the regression head
- `AdamW` optimizer
- `SmoothL1Loss`
- `ReduceLROnPlateau`

It also evaluated several seeds, but the more important point is that the architecture itself was stronger.

### Final Result

The best selected run achieved:

- R²: `0.6679`
- MSE: `0.009184`
- MAE: `0.074484`

This made it the strongest model in the project so far, outperforming:

- combined text TF-IDF (`R² = 0.4938`)
- separate resume/job TF-IDF (`R² = 0.4792`)
- hybrid TF-IDF + structured regression (`R² = 0.5256`)
- plain Siamese LSTM (`R² = 0.2910`)

## 3. Why The Upgraded Model Worked Better

The improvement is large enough that it is worth explaining in NLP terms.

### Bidirectionality Captures Context More Completely

The upgraded model replaced a one-direction LSTM with a **bidirectional LSTM**. In a normal LSTM, each token is interpreted mainly using left-to-right context. In a BiLSTM, each token representation is informed by both the preceding and following tokens.

That matters for resumes and job descriptions because many important phrases are multiword expressions or depend on surrounding context:

- `machine learning engineer`
- `bachelor of science`
- `project management`
- `data pipeline design`

A bidirectional encoder is better at representing these phrases because each token is seen in context from both directions. At a high level, this gives the model richer local and sentence-level semantics.

### Pooling Is Better Than a Single Final Hidden State

The upgraded model no longer depended on one final hidden state. Instead, it pooled over the whole sequence using:

- **mean pooling**
- **max pooling**

This is important conceptually.

Mean pooling gives the model a kind of global semantic average over the text. It helps preserve broad information about what topics and skills appear throughout the sequence.

Max pooling acts more like a detector for especially strong features. If an important technical term or phrase appears anywhere in the sequence, max pooling helps preserve that strong activation.

Together, these pooling methods reduce the compression problem of the basic LSTM. Instead of asking the model to summarize an entire resume in one recurrent endpoint, the upgraded model can preserve information distributed across the full text.

### The Model Became More About Matching, Not Just Encoding

The improved model still uses the Siamese framework, but it strengthens the **matching** part of the architecture.

The plain LSTM mainly produced two embeddings and compared them. The upgraded model adds:

- lexical overlap features
- structured compatibility signals
- richer pairwise comparison on top of stronger text embeddings

This better reflects the actual NLP task. Resume-job compatibility is not only about semantic similarity in a dense space. It also depends on whether concrete requirements line up:

- overlapping skill words
- overlap in technical vocabulary
- comparable experience or credentials

So the upgraded model mixes **distributed semantics** with **explicit matching features**. That is a more faithful representation of the problem.

### Lexical Overlap Features Help With Precision

In many NLP tasks, word embeddings are useful because they capture semantic similarity. But for hiring-style matching, exact or near-exact overlap also matters.

For example, if a job asks for `python`, `react`, or `data engineer`, seeing those exact or related terms in the resume should matter directly. A pure recurrent encoder may smooth this information too much. Lexical overlap features reintroduce a more symbolic signal.

This is a classic NLP tradeoff:

- dense representations are good for semantic generalization
- explicit overlap features are good for precision and alignment

The upgraded model benefits from using both.

### Structured Features Inject Useful Non-Text Information

Your earlier regression experiments already showed that structured features alone were weak, but useful when combined with text. The upgraded model follows that lesson instead of ignoring it.

Features such as:

- number of skills
- number of degrees
- number of positions
- years of experience
- skill overlap

do not replace the text encoder, but they provide extra clues about candidate-job fit. In NLP terms, this is a hybrid model that combines sequence representations with engineered side information. That tends to work well when text is informative but not the only source of signal.

### The Optimization Setup Was More Stable

The second notebook also improved the training setup:

- `AdamW` for more stable parameter updates
- `SmoothL1Loss` to reduce sensitivity to large regression errors
- `ReduceLROnPlateau` to lower the learning rate when validation loss stops improving

These are not the main conceptual reason for the gain, but they likely helped the larger model train more reliably. In a sequence model, architecture and optimization interact closely: a better model still needs a stable training process to reach its potential.

## 4. Why These Upgrades Were Chosen

These changes were not arbitrary. They were chosen because they directly address the weaknesses of the plain Siamese LSTM.

The plain model had three main limitations:

1. It summarized long text too aggressively with a single final hidden state.
2. It relied too much on implicit semantic encoding and not enough on explicit matching signals.
3. It ignored structured side information that earlier experiments had already shown to be useful.

The upgraded model was designed as a direct response:

- **BiLSTM** addresses weak contextual encoding.
- **Mean/max pooling** addresses over-compression of long sequences.
- **Lexical overlap features** address missing exact-match information.
- **Structured features** address useful non-text signals.

In other words, the upgraded model is better because it is better aligned with what the task actually requires.

## 5. Comparison Summary

| Model | R² | MSE |
|---|---:|---:|
| Structured only | 0.0939 | 0.0251 |
| Plain Siamese LSTM | 0.2910 | 0.0196 |
| Separate TF-IDF | 0.4792 | 0.0144 |
| Combined text TF-IDF | 0.4938 | 0.0140 |
| Hybrid TF-IDF + structured | 0.5256 | 0.0131 |
| Hybrid Siamese BiLSTM | 0.6679 | 0.0092 |

The upgraded model improved over the plain Siamese LSTM by about `0.3769` R² points and over the best earlier regression baseline by about `0.1423` R² points.

## 6. Conclusion

The experiments show a clear NLP lesson.

A basic Siamese LSTM is an understandable starting point for text-pair regression, but for long, noisy, domain-specific text like resumes and job descriptions, it is too limited when it relies on only a final hidden state.

The upgraded hybrid Siamese BiLSTM worked much better because it:

- encoded words with richer bidirectional context
- summarized the whole sequence instead of only its endpoint
- combined semantic representations with exact overlap signals
- integrated useful structured side information

So the main conclusion is not simply that a “bigger model” did better. The stronger result came from using a model whose architecture better matched the linguistic structure of the task.
