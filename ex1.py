from dataclasses import dataclass, field
from transformers import (HfArgumentParser, TrainingArguments, AutoTokenizer, AutoConfig,
                          AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer)
from datasets import load_dataset, DatasetDict
from typing import Optional, Tuple, Dict, Any
from sklearn.metrics import accuracy_score
import torch


MODEL_NAME = "bert-base-uncased"

@dataclass
class DataArguments:
    """
    Arguments for dataset loading and sample selection.
    """

    max_train_samples: int = field(
        default = -1,
        metadata={
            "help": "Number of samples to use for training. Use -1 to include all samples."
        }
    )
    max_eval_samples: int = field(
        default = -1,
        metadata={
            "help": "Number of samples to use for validation. Use -1 to include all samples."
        }
    )
    max_predict_samples: int = field(
        default = -1,
        metadata={
            "help": "Number of samples to use for prediction. Use -1 to include all samples."
        }
    )

@dataclass
class ModelArguments:
    """
    Arguments for model loading.
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model path to use when running prediction."
        }
    )

def parse_and_validate_args() -> Tuple[DataArguments, ModelArguments, TrainingArguments]:
    """
    Parse and validate command line arguments.
    """
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    # Add custom --lr argument to map to learning_rate
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for training.",
        dest="learning_rate"
    )
    # Add custom --batch_size argument to map to per_device_train_batch_size
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training and evaluation.",
        dest="per_device_train_batch_size"
    )
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # check arguments
    if data_args.max_train_samples < -1:
        raise ValueError("max_train_samples must be -1 or a non-negative integer")
    if data_args.max_eval_samples < -1:
        raise ValueError("max_eval_samples must be -1 or a non-negative integer")
    if data_args.max_predict_samples < -1:
        raise ValueError("max_predict_samples must be -1 or a non-negative integer")
    if training_args.do_predict and model_args.model_path is None:
        raise ValueError("Must provide --model_path when --do_predict is set")

    return data_args, model_args, training_args


def load_and_select_dataset(data_args: DataArguments) -> DatasetDict:
    """
    Load the MRPC dataset and apply sample selection based on DataArguments.

    Args:
        data_args: Parsed DataArguments containing sample selection parameters.

    Returns:
        DatasetDict containing the train, validation, and test splits after sample selection.
    """

    print("Loading MRPC dataset from nyu-mll/glue...")
    dataset = load_dataset("nyu-mll/glue", "mrpc")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Print original dataset sizes
    print(f"Original dataset sizes: train={len(train_dataset)}, "
          f"validation={len(val_dataset)}, test={len(test_dataset)}")

    # Apply sample selection
    if data_args.max_train_samples != -1:
        num_samples = min(data_args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(num_samples))
        print(f"Selected {num_samples} training samples")
    if data_args.max_eval_samples != -1:
        num_samples = min(data_args.max_eval_samples, len(val_dataset))
        val_dataset = val_dataset.select(range(num_samples))
        print(f"Selected {num_samples} validation samples")
    if data_args.max_predict_samples != -1:
        num_samples = min(data_args.max_predict_samples, len(test_dataset))
        test_dataset = test_dataset.select(range(num_samples))
        print(f"Selected {num_samples} test samples")

    # Return the dataset as a DatasetDict
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })


def setup_tokenizer() -> AutoTokenizer:
    """
    Load the tokenizer for bert-base-uncased.

    Returns:
        AutoTokenizer instance for bert-base-uncased.
    """
    print("Loading tokenizer for bert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer

def preprocess_function(examples: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Tokenize the sentence pairs in the MRPC dataset.

    Args:
        examples: Batch of examples from the dataset (contains sentence1, sentence2, and label).
        tokenizer: AutoTokenizer instance for tokenization.

    Returns:
        Dictionary with tokenized inputs (input_ids, attention_mask, token_type_ids) and labels.
    """

    # Tokenize the sentence pairs
    # applying dynamic padding - will be handled dynamically by the Trainer for training/validation
    result = tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        max_length=512,
        truncation=True,
        padding=False,
        return_tensors=None, # DataCollatorWithPadding will convert the lists to tensors dynamically
    )

    # Ensure labels are included in the output
    result["labels"] = examples["label"]
    return result


def preprocess_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Preprocess the dataset by tokenizing and formatting for PyTorch.

    Args:
        dataset: DatasetDict containing train, validation, and test splits.
        tokenizer: AutoTokenizer instance for tokenization.

    Returns:
        Preprocessed DatasetDict ready for training.
    """
    print("Preprocessing dataset...")

    # Apply tokenization to all splits using map
    # batched=True processes the dataset in batches for efficiency
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        desc="Tokenizing dataset"
    )

    # Define the columns to keep (no need for sentence1, sentence2, idx, etc.)
    columns_to_keep = ["input_ids", "attention_mask", "token_type_ids", "labels"]

    # Set format to PyTorch and keep only the necessary columns
    tokenized_dataset.set_format(
        type="torch",
        columns=columns_to_keep,
    )

    print("Dataset preprocessing complete.")
    return tokenized_dataset

def setup_model(model_args: ModelArguments, training_args: TrainingArguments) -> AutoModelForSequenceClassification:
    """
    Load the bert-base-uncased model for sequence classification.

    Args:
        model_args: Parsed ModelArguments containing the model path.
        training_args: Parsed TrainingArguments to check if prediction mode is enabled.

    Returns:
        AutoModelForSequenceClassification instance.
    """

    print("Loading model for bert-base-uncased...")
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    # Load the model
    if training_args.do_predict and model_args.model_path:
        print(f"Loading model from {model_args.model_path} for prediction...")
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_path, config=config)
    else:
        print(f"Loading pretrained {MODEL_NAME} for training...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    return model

def get_device() -> torch.device:
    """
    Dynamically detect the appropriate device (MPS, CUDA, or CPU).

    Returns:
        torch.device object representing the selected device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    """
    Compute accuracy for validation during training.
    Args:
        eval_pred: Tuple containing predictions and labels from the evaluation.

    Returns:
        Dictionary with the accuracy metric.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)  # Get the predicted class (0 or 1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def setup_trainer(
    model: AutoModelForSequenceClassification,
    training_args: TrainingArguments,
    tokenized_dataset: DatasetDict,
    tokenizer: AutoTokenizer
) -> Trainer:
    """
    Set up the Hugging Face Trainer for training the model.

    Args:
        model: The model to train.
        training_args: Parsed TrainingArguments with training hyperparameters.
        tokenized_dataset: Preprocessed DatasetDict containing train and validation splits.
        tokenizer: AutoTokenizer for dynamic padding.

    Returns:
        Trainer instance configured for training.
    """

    # initialize data collator for dynamic padding during training/validation
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Configure Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer

def configure_training_args(training_args: TrainingArguments) -> TrainingArguments:
    """
    Configure TrainingArguments for W&B logging, evaluation, and saving.

    Args:
        training_args: Parsed TrainingArguments to configure.

    Returns:
        Configured TrainingArguments instance.
    """
    training_args.report_to = ["wandb"]  # Enable W&B logging
    training_args.wandb_project = "anlp-ex1-mrpc"
    training_args.logging_strategy = "steps"  # Log at each step
    training_args.logging_steps = 1  # Log every step
    training_args.evaluation_strategy = "epoch"  # Evaluate at the end of each epoch
    training_args.save_strategy = "no"  # assignment says not to save during training
    training_args.load_best_model_at_end = False  # Load the best model based on validation metric
    training_args.metric_for_best_model = "accuracy"  # Use accuracy to determine the best model
    training_args.greater_is_better = True  # Higher accuracy is better
    return training_args

def train_model(trainer: Trainer, training_args: TrainingArguments) -> str:
    """
    Train the model if --do_train is set and save the trained model.

    Args:
        trainer: Configured Trainer instance.
        training_args: Configured TrainingArguments instance.

    Returns:
        Path to the saved model directory.
    """
    if not training_args.do_train:
        print("Skipping training as --do_train is not set.")
        return None

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # Save the final model
    # Use hyperparameters to create a unique directory name
    model_dir = (
        f"model_epochs_{training_args.num_train_epochs}_"
        f"lr_{training_args.learning_rate}_"
        f"batch_{training_args.per_device_train_batch_size}"
    )
    print(f"Saving model to {model_dir}...")
    trainer.save_model(model_dir)
    print(f"Model saved to {model_dir}.")

    return model_dir

def predict_and_save(
    model: AutoModelForSequenceClassification,
    tokenized_dataset: Dict[str, Any],
    raw_dataset: Dict[str, Any],
    device: torch.device,
    output_file: str,
    split: str = "test"
) -> None:
    """
    Perform prediction on the specified split and save results to the given output file.

    Args:
        model: Loaded model for prediction.
        tokenized_dataset: Preprocessed dataset containing the split.
        raw_dataset: Raw dataset containing the original sentences.
        device: Device to run prediction on (MPS, CUDA, or CPU).
        output_file: File path to save predictions.
        split: Dataset split to predict on ("validation" or "test").
    """
    print(f"Starting prediction on {split} split...")
    model.eval()  # Set model to evaluation mode
    model.to(device)  # Move model to the appropriate device

    # Get the datasets for the specified split
    tokenized_split = tokenized_dataset[split]
    raw_split = raw_dataset[split]
    predictions = []

    # Process each sample individually to avoid padding
    with torch.no_grad():  # Disable gradient computation for inference
        for i in range(len(tokenized_split)):
            # Get a single sample from the tokenized dataset
            sample = {key: tokenized_split[i][key].unsqueeze(0).to(device) for key in ["input_ids", "attention_mask", "token_type_ids"]}
            # Perform prediction
            outputs = model(**sample)
            logits = outputs.logits
            predicted_class = logits.argmax(dim=-1).item()  # Get the predicted class (0 or 1)
            # Get the original sentences from the raw dataset
            sentence1 = raw_split[i]["sentence1"]
            sentence2 = raw_split[i]["sentence2"]
            predictions.append((sentence1, sentence2, predicted_class))

    # Save predictions to the specified output file
    with open(output_file, "w") as f:
        for sentence1, sentence2, predicted_class in predictions:
            f.write(f"{sentence1}###{sentence2}###{predicted_class}\n")

    print(f"Predictions saved to {output_file} with {len(predictions)} entries.")

def save_results(training_args: TrainingArguments, trainer: Trainer) -> float:
    """
    Save validation accuracy to res.txt after training.

    Args:
        training_args: TrainingArguments containing hyperparameters.
        trainer: Trainer instance with evaluation metrics.

    Returns:
        Validation accuracy for the run.
    """
    # Evaluate the model on the validation set to get the accuracy
    eval_results = trainer.evaluate()
    eval_acc = eval_results["eval_accuracy"]

    # Append results to res.txt in the required format
    with open("res.txt", "a") as f:
        f.write(
            f"epoch_num: {training_args.num_train_epochs}, "
            f"lr: {training_args.learning_rate}, "
            f"batch_size: {training_args.per_device_train_batch_size}, "
            f"eval_acc: {eval_acc:.4f}\n"
        )

    print(f"Validation accuracy {eval_acc:.4f} appended to res.txt.")
    return eval_acc

def compute_test_accuracy(predictions_file: str, raw_dataset: DatasetDict) -> float:
    """
    Compute test accuracy by comparing predictions to true labels.

    Args:
        predictions_file: Path to the file containing test predictions.
        raw_dataset: Raw DatasetDict containing the test split with true labels.

    Returns:
        Test accuracy as a float.
    """
    from sklearn.metrics import accuracy_score

    # Load predictions
    predictions = []
    with open(predictions_file, "r") as f:
        for line in f:
            predicted_label = int(line.strip().split("###")[-1])  # Extract the predicted label
            predictions.append(predicted_label)

    # Get true labels from the test set
    test_dataset = raw_dataset["test"]
    true_labels = [example["label"] for example in test_dataset]

    # Ensure the number of predictions matches the test set size
    if len(predictions) != len(true_labels):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match test set size ({len(true_labels)})"
        )

    # Compute accuracy
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


def main():
    data_args, model_args, training_args = parse_and_validate_args()
    raw_dataset = load_and_select_dataset(data_args)
    tokenizer = setup_tokenizer()
    tokenized_dataset = preprocess_dataset(raw_dataset, tokenizer)
    model = setup_model(model_args, training_args)

    # Set up device (mostly for prediction. hopefully Trainer handles device for training)
    device = get_device()

    training_args = configure_training_args(training_args)
    trainer = setup_trainer(model, training_args, tokenized_dataset, tokenizer)

    # Train the model if --do_train is set
    model_dir = train_model(trainer, training_args)

    if training_args.do_train and model_dir:
        # Save validation accuracy to res.txt
        eval_acc = save_results(training_args, trainer)

        # Generate a config string for unique filenames
        config_str = (
            f"epochs_{training_args.num_train_epochs}_"
            f"lr_{training_args.learning_rate}_"
            f"batch_{training_args.per_device_train_batch_size}"
        )

        # Predict on validation set for qualitative analysis
        val_output_file = f"val_predictions_{config_str}.txt"
        predict_and_save(model, tokenized_dataset, raw_dataset, device, val_output_file, split="validation")

    # Perform prediction on the entire test set if --do_predict is set
    if training_args.do_predict:
        output_file = "new_predictions.txt"
        predict_and_save(model, tokenized_dataset, raw_dataset, device, output_file, split="test")

        # Compute test accuracy
        test_acc = compute_test_accuracy(output_file, raw_dataset)
        print(f"Test accuracy for {model_args.model_path}: {test_acc:.4f}")



if __name__ == "__main__":
    main()
