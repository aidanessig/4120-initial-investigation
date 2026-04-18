# CS4120: Final Project Check In

**Aidan Essig, Joseph Trisnandi, Alex Miale**

## Introduction

For our project, we are investigating models that predict compatibility between one’s resume and a job description. The goal is to give job seekers a clearer sense of how well their qualifications align with a potential role. The models will output a single compatibility score (from 0 to 1) that allows for an easy-to-understand potential fit (0 being poor, 1 being best).

We are focusing on whether word embedding techniques combined with a neural network can effectively predict these scores, and which embedding approach performs the best. This is done by experimenting with a few different embedding strategies and benchmarking them against a simple linear regression baseline. We are also intentionally leaving out sensitive features, such as name and location, to keep the model focused on skills, experience, and qualifications.

## Data

Our initial dataset consisted of 9,544 raw resume-job pairs structured across 34 columns. To prepare the continuous `matched_score` target variable for prediction, we developed a data cleaning pipeline. First, we normalized the unstructured text by converting strings to lowercase, properly formatting column names, and sweeping for placeholder values (for example, empty strings, `n/a`, `[]`, or filler like `['company name']`) to replace them with accurate `NaN` nulls. This normalization process revealed that 14 columns were overwhelmingly empty (over 80% missing), which we deemed unusable, so they were entirely dropped to reduce sparsity.

With a cleaner foundation, we moved to feature extraction. A crucial step was parsing poorly formatted categorical lists (for example, some columns stored `['python', 'mysql']` as strings) back into true Python lists. We engineered structural numerical features representing:

- `num_skills`
- `num_degrees`
- `num_positions`
- `experience_years`

Finally, we combined key descriptive fields from the resume (career objective, skills, degrees, responsibilities) and the corresponding job listing (position name, required skills, and experience) into three richer NLP text features:

- `resume_text`
- `job_text`
- `combined_text`

Overall, our final cleaned dataset currently totals 9,544 rows and 17 polished columns with exactly 0 missing target values, primed to support our numerical and textual baseline representations.

## Models

Our analytical strategy is structured around establishing conventional NLP benchmarks before advancing into more complex continuous representations.

### Linear Regression (Baseline Framework)

To measure performance organically, we established a sequence of standard baselines utilizing `LinearRegression` from the `scikit-learn` suite. The models ingest either engineered structured variables (numeric skill counts and experience years) or textual datasets normalized via Term Frequency-Inverse Document Frequency (TF-IDF). To accurately investigate improvements between models, all permutations utilize a standard 80/20 train/test split.

### Word Embedding Neural Networks

Recognizing the limitations of Bag-of-Words and TF-IDF in capturing rich semantic context between highly specialized skills, we built Deep Averaging Networks (DAN) utilizing PyTorch, hoping that this architecture will bypass TF-IDF sparsity altogether by processing continuous word vector representations to parse similarities.

Currently, we are investigating predefined semantic spaces using large pre-trained vectors from the GloVe/DOLMA dataset (trained over 220B tokens and a 1.2M vocab). To handle immense computational load effectively, we have successfully built a preprocessing pipeline strictly relativizing the multi-gigabyte embeddings model down to the relevant vocabulary in our training data.

## Preliminary Results

So far, we have run four linear regression experiments to establish a baseline before moving to embeddings and neural networks. All four use the same 80/20 train/test split for an equal comparison.

- For our pure baseline, we used TF-IDF on a small set of raw columns, achieving an R² of **0.477**.
- Using the richer `combined_text` from our cleaned dataset resulted in an R² of **0.494**.
- Splitting resume and job text into separate TF-IDF vectors came in just above the original baseline at **0.479**.
- Structured features alone (skill counts, experience years, etc.) performed poorly at **0.094**, confirming that numeric features by themselves do not carry enough signal.
- The best result so far came from combining TF-IDF with the structured features, reaching an R² of **0.526**.

While these are encouraging early results, the weak performance of structured features alone and the ceiling on TF-IDF both point to the same conclusion: we need richer semantic representations, which is exactly what the word embedding models are meant to address.

Additionally, our early exploration using the massive GloVe/DOLMA embeddings on our training corpus has yielded a very interesting finding. A significant number of specialized vocabulary words naturally prevalent in job resumes (for example, complex software frameworks or industry-specific terms like `webscraping`) do not appear in the pre-trained embedding vocabulary. This insight suggests that off-the-shelf general language embeddings might not be entirely adequate for domain-specific resume screening applications without specialized continuous representations or fine-tuning, which will heavily inform our upcoming iterations.

## Additional Results and Findings

### LSTM Findings

After the preliminary regression stage, we trained two recurrent neural models on the cleaned dataset using the shared neural split of:

- train: `6489`
- validation: `1146`
- test: `1909`

The first was a baseline LSTM regressor built with a shared embedding layer, a shared one-direction LSTM encoder, pairwise comparison features, and an MLP regression head. That model selected its best checkpoint at epoch 13 and achieved:

- plain LSTM: `R² = 0.2910`, `MSE = 0.0196`, `MAE = 0.1110`

Although this outperformed the structured-only regression model, it performed substantially worse than the TF-IDF text baselines. This suggests that a simple shared LSTM relying on the final hidden state was too limited for long, noisy resume and job texts. In practice, too much information was likely compressed into a single endpoint representation, and the model did not incorporate enough direct matching signals between the two texts.

The second recurrent model was a stronger Hybrid BiLSTM that incorporated:

- a shared pretrained embedding layer
- a shared bidirectional LSTM encoder
- mean pooling and max pooling
- lexical overlap features
- structured numeric features
- more stable optimization through `AdamW`, `SmoothL1Loss`, and `ReduceLROnPlateau`

The best Hybrid BiLSTM run achieved:

- Hybrid BiLSTM: `R² = 0.6679`, `MSE = 0.009184`, `MAE = 0.074484`

This made it the strongest model in the project at that stage, outperforming every regression baseline, including the earlier hybrid TF-IDF + structured model. The improvement suggests that the stronger recurrent architecture was better aligned with the actual task: bidirectionality captured context more completely, mean/max pooling preserved information distributed across the sequence, and the lexical plus structured features reintroduced explicit compatibility signals that pure encoding alone could miss.

These experiments also reinforced an important finding from our embedding pipeline. The shared training vocabulary contained 3,904 tokens, but pretrained embedding coverage from the relativized DOLMA file was only `64.02%`. This supported our earlier suspicion that many specialized resume and job terms were poorly covered by off-the-shelf word embeddings, which helped motivate our final transformer direction.

### Transformer Findings

The transformer experiments used MiniLM as a jointly encoded paired-text model and produced the strongest results in the project. Early MiniLM runs were already competitive, but the final iteration improved substantially after:

- extending the training budget
- using a warmup/decay scheduler
- preserving a fair shared neural split
- improving phrase-aware overlap features
- standardizing structured side features
- preserving full job text while truncating the resume first when needed

The final transformer results were:

- MiniLM text-only: `R² = 0.7075`, `MSE = 0.008090`, `MAE = 0.06655`
- best feature-augmented MiniLM: `R² = 0.7086`, `MSE = 0.008058`, `MAE = 0.06683`

This means the final transformer outperformed the strongest earlier Hybrid BiLSTM by about:

- `0.0407` in `R²`
- `0.00113` in MSE
- `0.00765` in MAE

One of the most interesting final findings is that the text-only MiniLM result was almost identical to the best feature-augmented MiniLM result. This suggests that once the paired transformer encoder became strong enough, it was able to capture most of the useful compatibility signal directly from the text, with engineered lexical and structured features providing only a small additional lift.

## Final Comparison Summary

| Model | R² | MSE | MAE |
|---|---:|---:|---:|
| Best feature-augmented MiniLM | `0.7086` | `0.008058` | `0.06683` |
| MiniLM text-only | `0.7075` | `0.008090` | `0.06655` |
| Hybrid BiLSTM | `0.6679` | `0.009184` | `0.074484` |
| Hybrid TF-IDF + structured | `0.5256` | `0.0131` | — |
| Combined TF-IDF | `0.4938` | `0.0140` | — |
| Separate TF-IDF | `0.4792` | `0.0144` | — |
| Plain LSTM | `0.2910` | `0.0196` | `0.1110` |
| Structured only | `0.0939` | `0.0251` | — |

## Conclusion

The final outcome of the project is a clear progression in model quality as the representations became better aligned with the structure of the task. Classical TF-IDF baselines showed that resume-job compatibility is strongly text-driven, while the weak structured-only baseline showed that counts and metadata alone are not sufficient. The plain LSTM confirmed that a simple recurrent encoder is not enough for long, noisy, domain-specific documents, but the Hybrid BiLSTM demonstrated that recurrent models can become much stronger when they combine richer sequence encoding with explicit overlap and structured signals.

The strongest overall result came from the MiniLM transformer, which achieved `R² = 0.7086`. Our final conclusion is that joint paired-text transformer encoding is the most effective approach we tested for resume-job compatibility prediction because it supports direct interaction between the resume and job description, handles specialized vocabulary more robustly through subword tokenization, and captures most of the meaningful predictive signal directly from the text.

At the same time, the final transformer report suggests that we are approaching a point of diminishing returns. The remaining error is already relatively small on a `0-1` target, the train and validation curves flatten without opening a large gap, and the text-only and feature-augmented MiniLM runs are nearly tied. Taken together, these results suggest that there may not be much room left for substantial improvement before further tuning begins to risk overfitting to dataset-specific noise. For that reason, the final transformer model represents both the strongest result in the project and a reasonable stopping point for the current scope of the work.
