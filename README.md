# Advanced NLP Exercise 1: Fine Tuning

This is the code base for the ANLP HUJI course exercise 1. The exercise focuses on fine-tuning the bert-base-uncased model to perform paraphrase detection on the MRPC (Microsoft Research Paraphrase Corpus) dataset from the GLUE benchmark.

## Installation

To install the required dependencies, run:

``` bash
  pip install -r requirements.txt
```

## Project Structure

- `ex1.py`: Main script for fine-tuning and prediction
- `wx1_answers.pdf`: Answers for assignment's Questions
- `qualitative_analysis.py`: Script for analyzing differences between model configurations
- `res.txt`: Results file containing validation accuracies for different configurations
- `predictions.txt`: Test predictions for the best model (Config d: epoch_num=5, lr=3e-05, batch_size=8).
- `requirements.txt`: Required Python packages
- `train_loss.png`: Plot of training loss from Weights & Biases
- `differing_examples.txt`: Examples where the best model succeeded but the worst failed, used for qualitative analysis.`
- `val_predictions_*.txt`: Validation predictions for each configuration, used for qualitative analysis.

## Usage

### Fine-Tuning

To fine-tune the model, use the following command:

```bash
  python ex1.py --max_train_samples NUM_TRAIN --max_eval_samples NUM_EVAL --num_train_epochs EPOCHS --lr LEARNING_RATE --batch_size BATCH_SIZE --do_train
```

Example:
```bash
  python ex1.py --max_train_samples -1 --max_eval_samples -1 --num_train_epochs 5 --lr 3e-05 --batch_size 8 --do_train
```

### Prediction

To run prediction using a trained model:

```bash
  python ex1.py --max_predict_samples NUM_PREDICT --model_path MODEL_PATH --do_predict
```

Example:
```bash
  python ex1.py --max_predict_samples -1 --model_path model_epochs_5.0_lr_3e-05_batch_8 --do_predict
```

## Parameters

- `--max_train_samples`: Number of samples to use for training. Use -1 to include all samples.
- `--max_eval_samples`: Number of samples to use for validation. Use -1 to include all samples.
- `--max_predict_samples`: Number of samples to use for prediction. Use -1 to include all samples.
- `--num_train_epochs`: Number of training epochs.
- `--lr`: Learning rate.
- `--batch_size`: Batch size for training.
- `--do_train`: Flag to enable training.
- `--do_predict`: Flag to enable prediction.
- `--model_path`: Path to the model for prediction.

## Qualitative Analysis

To run the qualitative analysis comparing the best and worst model configurations:

``` bash
  python qualitative_analysis.py --best_predictions_file <best_model_predictions> --worst_predictions_file <worst_model_predictions>
```

Example:
```bash
      python qualitative_analysis.py --best_predictions_file val_predictions_epochs_5.0_lr_3e-05_batch_8.txt --worst_predictions_file val_predictions_epochs_2.0_lr_1e-05_batch_32.txt
```

This will generate a file `differing_examples.txt` with examples where the best model succeeded but the worst model failed.

### Available Parameters

- `--best_predictions_file`: Path to the validation predictions file from the best model
- `--worst_predictions_file`: Path to the validation predictions file from the worst model
- `--output_file`: Path to save the analysis results (default: differing_examples.txt)
- `--best_config_name`: Name identifier for the best configuration (default: Config d)
- `--worst_config_name`: Name identifier for the worst configuration (default: Config c)

## Dataset

The code uses the MRPC dataset from the GLUE benchmark, loading it from the Hugging Face Datasets library (`nyu-mll/glue`). The dataset contains pairs of sentences, with labels indicating whether they are paraphrases (equivalent in meaning) or not.

## Implementation Details

- Model: bert-base-uncased
- Truncation: Inputs are truncated to the maximum sequence length (512 tokens)
- Padding: Dynamic padding is used during training
- Evaluation: Accuracy on the validation set is logged during training
- Logging: Weights & Biases is used for experiment tracking

## Author
Yuval Cohen (208305052)
