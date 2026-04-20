# Resume-Job Match Prediction

This repository experiments with resume-job compatibility prediction on `9,544` resume-job pairs with a continuous target `matched_score` in `[0, 1]`. The project starts off with a raw-data TF-IDF regression baseline and progresses through a cleaned-data regression pipeline, then to a Deep Averaging Network (DAN), an LSTM, a hybrid BiLSTM, and finally a transformer using `sentence-transformers/all-MiniLM-L6-v2`. The cleaned dataset used by the later models is `resume_data_cleaned.csv`, which is produced from `resume_data.csv` by normalizing placeholders, fixing schema issues, feature engineering, and building `resume_text`, `job_text`, and `combined_text`. The strongest final result in the repo is the MiniLM transformer with `R² = 0.7086`, `MSE = 0.008058`, and `MAE = 0.06683`.

## Repository Layout

- `resume_data.csv`: raw dataset.
- `resume_data_cleaned.csv`: cleaned dataset used by the post-cleaning models.
- `data_cleaning.ipynb`: generates the cleaned dataset and engineered features.
- `trying_regression.ipynb`: raw-data TF-IDF + linear regression baseline.
- `regression_cleaned.ipynb`: cleaned-data regression baselines and hybrid structured model.
- `dan + predefined embeddings/DAN.ipynb`: Deep Averaging Network experiments using relativized DOLMA embeddings.
- `lstm/lstm_experiment.ipynb`: plain shared-encoder LSTM regression model.
- `lstm/bilstm_hybrid_experiment.ipynb`: hybrid BiLSTM with lexical and structured features.
- `lstm/lstm.py`: shared dataset/model/training utilities for the LSTM notebooks.
- `transformer/colab_transformer_experiment.ipynb`: MiniLM transformer experiments, intended for Colab.
- `transformer/transformer.py`: shared dataset/model/training utilities for the transformer notebook.
- `transformer/requirements.txt`: transformer dependencies.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn scipy torch jupyter
```

## Recommended Run Order

If you want to reproduce the full project from earliest baseline to strongest model, run the notebooks in this order:

1. `data_cleaning.ipynb`
2. `trying_regression.ipynb`
3. `regression_cleaned.ipynb`
4. `dan + predefined embeddings/DAN.ipynb`
5. `lstm/lstm_experiment.ipynb`
6. `lstm/bilstm_hybrid_experiment.ipynb`
7. `transformer/colab_transformer_experiment.ipynb`

This order mirrors the modeling progression and ensures that `resume_data_cleaned.csv` exists before any post-cleaning model is run.

## How To Run And Test Each Model

### 1. Data Cleaning

Open `data_cleaning.ipynb` and run all cells from top to bottom. This notebook loads `resume_data.csv`, removes BOM and duplicate columns, fixes naming issues, converts placeholder strings to nulls, parses list-like text fields, engineers numeric features such as `num_skills`, `experience_years`, `age_min`, `age_max`, and `skill_jaccard`, and builds `resume_text`, `job_text`, and `combined_text`. The final cell writes `resume_data_cleaned.csv`.

### 2. Raw TF-IDF Regression Baseline

Open `trying_regression.ipynb` and run all cells. This notebook works directly from the raw CSV and uses a `TfidfVectorizer(max_features=500)` over a simple concatenation of `skills`, `job_position_name`, `educationaL_requirements`, and `responsibilities`, followed by `LinearRegression`. Testing is built into the notebook: it creates an `80/20` train-test split and prints `MSE` and `R²` on the held-out set. The saved output in the notebook reports `9,488` usable rows after dropping missing values, `MSE = 0.0147`, and `R² = 0.4770`.

### 3. Cleaned Regression Baselines

Open `regression_cleaned.ipynb` after the cleaned CSV exists and run it top to bottom. This notebook compares four cleaned-data baselines on a shared split: combined-text TF-IDF, separate resume/job TF-IDF, structured-only regression, and a hybrid TF-IDF + structured model. It uses the engineered text fields plus structured features `num_skills`, `num_degrees`, `num_positions`, `experience_years`, `age_min`, `age_max`, and `skill_jaccard`. Each experiment trains on the training partition and prints held-out `MSE` and `R²`, followed by a comparison table and plots. The best regression result here is the hybrid model with `MSE = 0.0131` and `R² = 0.5256`.

### 4. DAN With Predefined Embeddings

Open `dan + predefined embeddings/DAN.ipynb` and run it from inside that directory context so relative paths resolve correctly. The notebook loads `../resume_data_cleaned.csv` and `embeddings/dolma_300_relativised.txt`, trains a Deep Averaging Network over `combined_text`, and also includes a variant with a small set of numerical features. The simplest test is the notebook's own evaluation block, which predicts on the held-out split and prints `MSE`, `R²`, and `MAE`. The saved output currently reports `MSE = 0.0255`, `R² = 0.0778`, and `MAE = 0.1261` for the text-only DAN, so if you rerun it you should expect a weak baseline rather than a competitive final model. Trained weights can be saved and reloaded from `dan + predefined embeddings/models/`.

### 5. Plain LSTM

Open `lstm/lstm_experiment.ipynb` and run all cells after confirming the DOLMA embedding file still exists at `dan + predefined embeddings/embeddings/dolma_300_relativised.txt`. This notebook uses utilities from `lstm/lstm.py`, builds a train-only vocabulary, initializes embeddings from the DOLMA file, and trains a shared one-direction LSTM encoder followed by an MLP regression head. It evaluates on the documented split of `6489 / 1146 / 1909` train/validation/test rows and prints checkpoint, best epoch, and held-out metrics. The saved result is `R² = 0.2910`, `MSE = 0.0196`, and `MAE = 0.1110`, with the best checkpoint selected at epoch `13`.

### 6. Hybrid BiLSTM

Open `lstm/bilstm_hybrid_experiment.ipynb` and run it after the plain LSTM if you want the same split logic and embedding source. This notebook reuses `lstm/lstm.py` but upgrades the encoder to a pooled bidirectional LSTM and appends lexical overlap plus structured features before the regression head. It can run a small seed sweep over `42`, `52`, and `62`, then keeps the best validation run. Testing is again built in: it prints validation/test metrics per seed, selects the best checkpoint, and compares the chosen run to all earlier baselines. The strongest saved run is seed `42` with `R² = 0.6679`, `MSE = 0.009184`, and `MAE = 0.074484`.

### 7. Transformer

Open `transformer/colab_transformer_experiment.ipynb`, preferably in Google Colab, and run all cells in order. The notebook installs `transformer/requirements.txt`, loads `resume_data_cleaned.csv`, prepares the shared neural split, and calls utilities from `transformer/transformer.py` to train paired-input transformer regressors. The core model feeds `[resume_text, job_text]` into `sentence-transformers/all-MiniLM-L6-v2`, optionally augments the pooled representation with lexical and structured features, and writes artifacts under `transformer/artifacts/`. Testing is handled by the notebook itself: each run logs validation history, test `R²/MSE/MAE`, comparison plots, and per-example analysis files. The best saved final result is the feature-augmented MiniLM run with `R² = 0.7086`, `MSE = 0.008058`, and `MAE = 0.06683`; the text-only MiniLM variant is nearly tied at `R² = 0.7075`.

## Model Progression And Findings

### Regression

The regression stage established a strong classical baseline before moving to neural models. The raw-data TF-IDF baseline already performed reasonably well, reaching `R² = 0.4770`, which meant simple lexical text representations were carrying real signal. After cleaning the data and engineering features, the combined-text TF-IDF model improved slightly to `R² = 0.4938`, while separate resume/job TF-IDF stayed roughly comparable at `R² = 0.4792`. Structured features alone were weak at `R² = 0.0939`, showing that counts and extracted numeric fields could not explain match quality by themselves. The best regression system was the hybrid TF-IDF + structured model at `R² = 0.5256` and `MSE = 0.0131`, which showed that structured features were useful only when paired with strong text features. The main lesson from this stage was that text dominates this task, but structured signals are still worth keeping as auxiliary inputs. This stage also set the benchmark that later neural models needed to beat to justify their extra complexity.

### DAN

The DAN stage tested whether averaging pretrained word embeddings over the text could outperform the regression baselines. In practice, it did not: the saved text-only DAN result was `R² = 0.0778`, `MSE = 0.0255`, and `MAE = 0.1261`, which was worse than even the simple raw TF-IDF regression. That result suggests that compressing a long, noisy resume-job pair into a single mean embedding discards too much of the alignment structure that matters for matching. Exact skill overlap, phrase context, and localized evidence inside the text were all effectively blurred away. The notebook includes a version with extra numeric features, but the project trajectory still moved on because the core averaging approach was too weak. The DAN was still useful as a negative result: it showed that pretrained embeddings alone are not enough if the model cannot preserve sequence structure or explicit interaction patterns. The main takeaway was that better neural models needed to do more than average word vectors.

### LSTM

The plain LSTM was the first neural model that tried to encode resume and job text as sequences instead of bags of words or mean embeddings. It improved clearly over the DAN, ending at `R² = 0.2910`, `MSE = 0.0196`, and `MAE = 0.1110`, but it still fell far short of the TF-IDF baselines. The problem was that a one-direction shared LSTM summarized each text with only its final hidden state, which is too narrow a bottleneck for long resumes with scattered evidence. The model also encoded the two texts independently and only compared them after both had already been compressed. The takeaway was that a minimal sequence encoder was not expressive enough for this matching problem.

### BiLSTM

The hybrid BiLSTM was the first model in the repo that clearly surpassed all earlier baselines. It combined bidirectional sequence encoding, mean and max pooling, lexical overlap features, and structured numeric features, and the best saved run reached `R² = 0.6679`, `MSE = 0.009184`, and `MAE = 0.074484`. That jump over the plain LSTM showed that the problem was not "neural vs. non-neural" but whether the architecture matched the task. Bidirectionality gave better contextual representations, pooling preserved evidence from across the full sequence, and the hybrid feature path reintroduced exact-match and numeric compatibility signals that mattered for hiring-style text pairs. This stage established the main pattern that carried into the transformer: strong text representations work best when they are paired with explicit matching signals. The BiLSTM therefore became the neural benchmark the transformer had to beat.

### Transformer

The transformer stage produced the strongest final model in the project. The key idea was to encode the resume and job description jointly with MiniLM so the model could learn alignment during encoding instead of only after separate compression. The final MiniLM runs improved substantially over both the first transformer pass and the hybrid BiLSTM, with the best feature-augmented model reaching `R² = 0.7086`, `MSE = 0.008058`, and `MAE = 0.06683`, while the text-only run reached `R² = 0.7075`. That near-tie between text-only and feature-augmented MiniLM is itself an important result: once the paired text encoder became strong enough, lexical and structured features provided only a marginal lift rather than driving performance. The main takeaway is that jointly encoded transformer representations are the best fit for this task in the current repo, and the remaining room for improvement is likely limited without risking overfitting.

## Final Historical Results

| Model | R² | MSE | MAE |
|---|---:|---:|---:|
| Raw TF-IDF regression | 0.4770 | 0.0147 | — |
| Combined TF-IDF (cleaned) | 0.4938 | 0.0140 | — |
| Separate TF-IDF (cleaned) | 0.4792 | 0.0144 | — |
| Structured only | 0.0939 | 0.0251 | — |
| Hybrid TF-IDF + structured | 0.5256 | 0.0131 | — |
| DAN | 0.0778 | 0.0255 | 0.1261 |
| Plain LSTM | 0.2910 | 0.0196 | 0.1110 |
| Hybrid BiLSTM | 0.6679 | 0.009184 | 0.074484 |
| MiniLM text-only | 0.7075 | 0.008090 | 0.06655 |
| Best MiniLM feature-augmented | 0.7086 | 0.008058 | 0.06683 |