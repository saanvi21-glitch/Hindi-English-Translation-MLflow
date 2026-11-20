# Hindi to English Translation – MLflow Champion–Challenger

## 1. Project Overview

**Task:** Hindi to English Machine Translation  
**IDE:** Visual Studio Code  
**Framework:** Hugging Face Transformers  
**Experiment Tracking:** MLflow  
**Evaluation Dataset:** `data/eval_inputs.csv`  
**Evaluation Samples:** 100  
**New Prediction Samples:** 10  

The objective of this project is to compare two pretrained Hindi-to-English translation models using the same evaluation dataset, track their performance using MLflow, select a Champion model using a predefined decision rule, and use the registered Champion to translate new sentences.

## 2. Models Used

**Model A – Initial Champion**

- Model ID: `Helsinki-NLP/opus-mt-hi-en`
- Role: Initial Champion

**Model B – Challenger**

- Model ID: `facebook/nllb-200-distilled-600M`
- Role: Challenger

Both models were evaluated using the same 100 Hindi sentences and corresponding English reference translations.

## 3. Evaluation Metrics

- **BLEU:** Primary metric. Measures similarity between the generated translation and reference translation. Higher is better.
- **chrF:** Tie-breaker metric based on character n-gram similarity. Higher is better.
- **Average latency:** Average time taken to translate one sentence, in milliseconds.
- **Total inference time:** Total time taken to translate all evaluation sentences, in seconds.

## 4. Champion–Challenger Decision Rule

Model A was treated as the initial Champion and Model B as the Challenger.

The decision rule was defined before the final evaluation:

1. Primary metric: BLEU (higher is better).
2. Minimum improvement margin: 1 BLEU point.
3. If Model B improves BLEU by at least 1 point, it replaces Model A.
4. If the BLEU difference is less than 1 point in either direction, compare chrF (higher is better).
5. If both metrics are tied, Model A remains Champion.

The 1-point margin was selected as a practical minimum improvement threshold for this experiment.

## 5. Experimental Results

The following results were obtained from the evaluation of 100 sentences.

| Metric | OPUS-MT (Model A) | NLLB-200 (Model B) |
|---|---:|---:|
| BLEU | 44.570979 | 69.028249 |
| chrF | 64.889648 | 80.804653 |
| Average latency (ms) | 163.246394 | 522.296713 |
| Total inference time (s) | 16.324639 | 52.229671 |

### Champion Selection

NLLB-200 achieved a BLEU improvement of approximately **24.46 points** over OPUS-MT.

Its chrF score was also approximately **15.92 points higher**.

Since the BLEU improvement exceeded the predefined 1-point margin, NLLB-200 was selected as the Champion.

**Final Champion:** `facebook/nllb-200-distilled-600M`

**Challenger:** `Helsinki-NLP/opus-mt-hi-en`

### Result Interpretation

NLLB-200 performed better than OPUS-MT on both translation quality metrics. However, it was slower: its average inference latency was approximately 522 ms per sentence, compared with 163 ms for OPUS-MT. This means NLLB-200 took roughly 3.2 times as long in this experiment.

The improvement in BLEU is substantial on this 100-example evaluation set. However, the dataset is relatively small and its reference translations are limited, so the result should not be treated as proof of general superiority across all Hindi-to-English text. Translation quality may vary with sentence type and complexity.

## 6. MLflow Experiment Tracking and Model Registry

Experiment name:

`hf-hindi-english-champion-challenger`

MLflow was used to record:

- Model parameters and configuration.
- BLEU and chrF scores.
- Average latency and total inference time.
- Model predictions and evaluation artifacts.
- Model comparison results.

Both models were registered under the same registered model name:

`hindi-english-translation`

The aliases were assigned as follows:

| Model version | Model | Alias |
|---|---|---|
| Version 1 | NLLB-200 | champion |
| Version 2 | OPUS-MT | challenger |

## 7. Prediction on New Data

A separate script, `predict_from_mlflow.py`, was created to load the Champion using:

`models:/hindi-english-translation@champion`

The script loads the model through MLflow rather than directly calling a Hugging Face pipeline.

It was tested on 10 new Hindi sentences that were not part of the evaluation dataset.

The script successfully:

- Loaded NLLB-200 from MLflow.
- Identified the Champion model version as Version 1.
- Generated English translations for all 10 sentences.
- Saved the results in `outputs/new_data_predictions.csv`.

## 8. Project Structure

```text
Hindi_English_Translation_MLflow/
│
├── data/
│   ├── eval_inputs.csv
│   └── new_data.csv
│
├── outputs/
│   ├── A_translation_opus_mt_metrics.json
│   ├── A_translation_opus_mt_predictions.csv
│   ├── B_translation_nllb_200_metrics.json
│   ├── B_translation_nllb_200_predictions.csv
│   ├── metrics_summary.csv
│   ├── metrics_summary.json
│   ├── translation_metrics_comparison.png
│   ├── inference_time_comparison.png
│   ├── qualitative_examples.csv
│   └── new_data_predictions.csv
│
├── screenshots/
│
├── evaluate.py
├── register_champion
##12. AI declaration:ChatGPT was used as an AI-assisted learning tool to provide guidance during the development of this project, particularly in writing, understanding, and debugging the Python code and terminal commands. The implementation and results were reviewed and tested by the student to ensure understanding of the project.
## 13. Project Completion

The project successfully completed model evaluation, MLflow experiment tracking, Champion selection, model registration, and prediction on 10 new Hindi sentences.

The final Champion model is NLLB-200 (facebook/nllb-200-distilled-600M).
## 13. Project Artifacts

The outputs folder contains:
- Evaluation metrics for both models.
- Predictions generated by both models.
- BLEU and chrF comparison chart.
- Inference time comparison chart.
- Qualitative translation examples.
- Predictions on 10 new Hindi sentences.

The screenshots folder contains evidence of MLflow experiment tracking, model registration, evaluation results, and new-data predictions.
