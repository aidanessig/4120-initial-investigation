# Transformer Model

This report summarizes the final MiniLM-based transformer iteration for resume-job compatibility prediction. It explains what we changed after the first pass, why those changes improved performance, what final configuration we went with, and why we believe there is limited room left before further tuning starts to look more like overfitting than real progress.

## 1. Motivation

The transformer direction was originally chosen to address the main weaknesses of the earlier recurrent models:

1. the plain LSTM compressed long texts too aggressively
2. the recurrent models encoded resume and job text separately before comparing them
3. the older DOLMA-based word embedding pipeline had limited vocabulary coverage for specialized resume and job terminology

The first transformer pass already showed that a jointly encoded text-pair model was strong, but it also showed that the initial MiniLM setup was likely undertrained. The original MiniLM results were:

- MiniLM hybrid: `R² = 0.6603`, `MSE = 0.00902`, `MAE = 0.06973`
- MiniLM text-only: `R² = 0.6441`, `MSE = 0.00945`, `MAE = 0.07372`

Those results were competitive with the Hybrid BiLSTM, but not clearly better. So instead of changing encoders, we kept `sentence-transformers/all-MiniLM-L6-v2` fixed and improved the surrounding training and feature pipeline.

## 2. What We Changed In The Final MiniLM Iteration

We kept MiniLM as the encoder and changed the parts around it that were most likely holding the model back.

### Training Improvements

- increased the training budget from `5` epochs to `10`
- relaxed early stopping from patience `2` to patience `3`
- added a linear warmup/decay learning-rate scheduler
- preserved the two-learning-rate setup:
  - small learning rate for the pretrained MiniLM encoder
  - larger learning rate for the regression head

These changes matter because the first pass looked undertrained. The later loss curves support that interpretation: both train and validation loss continued to improve smoothly through the longer run.

### Cleaner Evaluation Setup

- reran everything on the shared neural split:
  - train: `6489`
  - validation: `1146`
  - test: `1909`
- saved split metadata with the artifacts
- added source-row tracking so the final analysis is auditable against the cleaned dataset

This fixed the earlier comparison problem where the saved transformer analysis outputs did not line up cleanly with the documented split size.

### Stronger Side Features

- kept the original lexical overlap features
- added phrase-aware overlap features using multi-word matches
- standardized structured numeric features using train-split statistics
- ran side-feature ablations instead of only comparing text-only vs hybrid

This was important because the first pass suggested the transformer backbone was already strong, so we needed to check whether lexical or structured side information still contributed meaningful signal.

### Input Handling

- kept the paired-input transformer setup:
  - `[CLS] resume_text [SEP] job_text [SEP]`
- changed truncation behavior to preserve the full job text and truncate the resume first when needed

This was a better fit for the task, since job text is usually shorter and should remain fully visible while the longer resume absorbs the truncation budget.

## 3. Final Results

The final MiniLM runs on the corrected shared split were:

| Model | R² | MSE | MAE |
|---|---:|---:|---:|
| MiniLM text-only | `0.7075` | `0.008090` | `0.06655` |
| Best feature-augmented MiniLM run | `0.7086` | `0.008058` | `0.06683` |

The artifact summary also contains two additional feature-augmented ablation runs at:

- `R² = 0.6953`, `MSE = 0.008426`, `MAE = 0.06735`
- `R² = 0.7001`, `MSE = 0.008292`, `MAE = 0.06764`

So the overall story is clear:

- every final MiniLM run beat the old MiniLM hybrid baseline
- the final MiniLM runs also beat the earlier Hybrid BiLSTM baseline
- the best overall MiniLM result was `R² = 0.7086`

### Improvement Over The Earlier MiniLM Baseline

Compared with the original MiniLM hybrid result:

- `R²` improved from `0.6603` to `0.7086`
- `MSE` improved from `0.00902` to `0.00806`
- `MAE` improved from `0.06973` to `0.06683`

That is a substantial gain without changing the encoder model at all.

### Historical Comparison

| Model | R² | MSE | MAE |
|---|---:|---:|---:|
| Best feature-augmented MiniLM | `0.7086` | `0.008058` | `0.06683` |
| MiniLM text-only | `0.7075` | `0.008090` | `0.06655` |
| Hybrid BiLSTM (DOLMA-backed) | `0.6679` | `0.009184` | `0.07448` |
| Hybrid TF-IDF + structured | `0.5256` | `0.0131` | — |
| Combined TF-IDF | `0.4938` | `0.0140` | — |
| Separate TF-IDF | `0.4792` | `0.0144` | — |
| Plain LSTM (DOLMA-backed) | `0.2910` | `0.0196` | `0.1110` |
| Structured only | `0.0939` | `0.0251` | — |

This means the final MiniLM setup outperformed the strongest earlier DOLMA-backed BiLSTM by:

- about `0.0407` `R²`
- about `0.00113` MSE
- about `0.00765` MAE

## 4. What We Went With

Our final direction was to keep `all-MiniLM-L6-v2` fixed and improve optimization and feature handling rather than changing the transformer backbone.

The final story is:

- the encoder itself was already good enough
- the first pass was mainly undertrained
- a better schedule, longer training budget, cleaner split handling, and better side-feature construction produced the improvement

Although the best single score came from a feature-augmented MiniLM run, the text-only result was almost identical. That is an important conclusion. It means the paired MiniLM encoder is now carrying most of the signal directly from the text, and the side features only provide a small extra lift at best.

So our final transformer story is not that the hybrid add-ons dramatically changed the model. The bigger change was getting the MiniLM training setup and evaluation pipeline into a stable, fair, better-optimized state.

## 5. Why The New Setup Worked Better

The results suggest that the improvement came from three main sources.

### A. MiniLM Was Undertrained In The First Pass

The longer budget and scheduler helped the model continue improving instead of stopping too early. In the saved plots, both train and validation loss decrease smoothly across the full run rather than plateauing immediately.

### B. The Shared Split Is Now Credible

The final results are all reported on the same documented `6489 / 1146 / 1909` split used for the neural comparisons. That makes the gain over the Hybrid BiLSTM much more defensible than the earlier draft comparison.

### C. Phrase-Aware And Structured Signals Help At The Margin

The side features do not dominate the model the way they did in older baselines, but they still help fine-tune prediction quality. The best overall run came from a feature-augmented configuration, even though the gain over text-only was small.

This is exactly what we would expect once the text encoder becomes strong: explicit overlap and structured features stop being the main driver and become secondary refinements.

## 6. Why We Do Not Think There Is Much Room Left Before Overfitting

We do **not** think the final MiniLM run is severely overfit already. The train and validation curves stay close, and the residual plots remain centered near zero. But we do think we are approaching the point where additional tuning is more likely to fit noise than produce meaningful generalization gains.

There are several reasons for that.

### A. The Remaining Headroom Is Small

The final test `MSE` is about `0.00806` on a target scaled from `0` to `1`, and the test `MAE` is about `0.0668`. In practical terms, the model is now missing the target by only about `6.7` percentage points on average.

At this point, the remaining error is small enough that some of it is likely coming from:

- label noise in `matched_score`
- borderline resume-job pairs that are genuinely ambiguous
- imperfectly captured human judgment in the original data

### B. Train And Validation Loss Have Flattened

By the later epochs, the curves keep converging but with diminishing returns. The model is not opening a large train-validation gap, which is good, but it also means there is no sign of a large untapped improvement still sitting in the current setup.

### C. Text-Only And Feature-Augmented Results Are Nearly Tied

The best feature-augmented run (`R² = 0.7086`) and the text-only run (`R² = 0.7075`) are extremely close. That suggests the model has already extracted most of the available signal from the paired text, and the remaining gains from extra feature engineering are likely to be marginal.

### D. More Complexity Now Carries More Risk Than Reward

The next possible upgrades would be things like chunked resumes, more aggressive hybrid architectures, or deeper hyperparameter tuning. Those are all much more complex than the changes that already worked. Given how low the error already is and how narrow the remaining margin is, those changes are more likely to overfit to this dataset or to produce gains too small to be trustworthy.

So our conclusion is not that improvement is mathematically impossible. It is that we have reached a reasonable stopping point where:

- the model is already outperforming every earlier baseline
- the error is already low on a `0-1` target
- the remaining headroom is small
- additional optimization would carry a much higher risk of fitting dataset-specific noise

## 7. Conclusion

The final transformer story is stronger than the first draft.

We kept `all-MiniLM-L6-v2` fixed, improved the training setup, strengthened the overlap features, corrected the evaluation split, and reran the model under a cleaner and more credible pipeline. Those changes pushed MiniLM from a merely competitive transformer result to the strongest model in the project.

The best final MiniLM run achieved:

- `R² = 0.7086`
- `MSE = 0.008058`
- `MAE = 0.06683`

This clearly beats the earlier Hybrid BiLSTM and does so without needing a different encoder. Just as importantly, the final results suggest that most of the meaningful performance gain has already been captured. From here, further work would likely be trading simplicity and robustness for increasingly marginal gains, with a growing risk of overfitting to a relatively small and noisy compatibility dataset.
