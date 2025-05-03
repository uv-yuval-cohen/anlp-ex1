from datasets import load_dataset, DatasetDict
from typing import Dict, Any
import argparse


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description="Qualitative analysis of model predictions.")
    parser.add_argument(
        "--best_predictions_file",
        type=str,
        default="val_predictions_epochs_5.0_lr_3e-05_batch_8.txt",
        help="Path to the predictions file of the best model."
    )
    parser.add_argument(
        "--worst_predictions_file",
        type=str,
        default="val_predictions_epochs_2.0_lr_1e-05_batch_32.txt",
        help="Path to the predictions file of the worst model."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="differing_examples.txt",
        help="Path to save the analysis results."
    )
    parser.add_argument(
        "--best_config_name",
        type=str,
        default="Config d",
        help="Name identifier for the best configuration."
    )
    parser.add_argument(
        "--worst_config_name",
        type=str,
        default="Config c",
        help="Name identifier for the worst configuration."
    )

    return parser.parse_args()


def load_and_select_dataset() -> DatasetDict:
    """
    Load the MRPC dataset without sample selection for qualitative analysis.

    Returns:
        DatasetDict containing the train, validation, and test splits.
    """
    print("Loading MRPC dataset from nyu-mll/glue...")
    dataset = load_dataset("nyu-mll/glue", "mrpc")
    return dataset


def load_predictions(file_path: str) -> list:
    """
    Load prediction results from a file.

    Args:
        file_path: Path to the predictions file.

    Returns:
        List of dictionaries containing predictions.
    """
    predictions = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split("###")
                if len(parts) == 3:  # Ensure the line has the expected format
                    predictions.append({
                        "sentence1": parts[0],
                        "sentence2": parts[1],
                        "predicted_label": int(parts[2])
                    })
        print(f"Successfully loaded {len(predictions)} predictions from {file_path}")
        return predictions
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading predictions from {file_path}: {e}")
        exit(1)


def perform_qualitative_analysis(
        raw_dataset: DatasetDict,
        best_predictions_file: str,
        worst_predictions_file: str,
        output_file: str,
        best_config_name: str,
        worst_config_name: str
) -> None:
    """
    Perform qualitative analysis by comparing the best and worst configurations' validation predictions.
    Also counts how many times the worst model overpredicts (predicts 1 when true label is 0) and underpredicts
    (predicts 0 when true label is 1) equivalence among differing examples.

    Args:
        raw_dataset: Raw DatasetDict containing the validation split with true labels.
        best_predictions_file: Path to the predictions file of the best model.
        worst_predictions_file: Path to the predictions file of the worst model.
        output_file: Path to save the analysis results.
        best_config_name: Name identifier for the best configuration.
        worst_config_name: Name identifier for the worst configuration.
    """
    # Load predictions
    best_predictions = load_predictions(best_predictions_file)
    worst_predictions = load_predictions(worst_predictions_file)

    # Get true labels from the validation set
    val_dataset = raw_dataset["validation"]
    true_labels = [example["label"] for example in val_dataset]

    # Validate that predictions match dataset size
    if len(best_predictions) != len(true_labels) or len(worst_predictions) != len(true_labels):
        print(f"Warning: Prediction counts don't match validation set size.")
        print(
            f"Validation set: {len(true_labels)}, Best predictions: {len(best_predictions)}, Worst predictions: {len(worst_predictions)}")
        print("Will analyze only the overlapping part.")
        min_len = min(len(true_labels), len(best_predictions), len(worst_predictions))
        true_labels = true_labels[:min_len]
        best_predictions = best_predictions[:min_len]
        worst_predictions = worst_predictions[:min_len]

    # Find examples where the best model succeeded but the worst failed
    differing_examples = []
    overpredict_equivalence = 0  # Worst predicts 1 when true label is 0
    underpredict_equivalence = 0  # Worst predicts 0 when true label is 1

    for i in range(len(true_labels)):
        true_label = true_labels[i]
        best_pred = best_predictions[i]["predicted_label"]
        worst_pred = worst_predictions[i]["predicted_label"]

        if best_pred == true_label and worst_pred != true_label:
            # Add to differing examples
            differing_examples.append({
                "index": i,
                "sentence1": best_predictions[i]["sentence1"],
                "sentence2": best_predictions[i]["sentence2"],
                "true_label": true_label,
                "best_pred": best_pred,
                "worst_pred": worst_pred
            })

            # Count overprediction (worst predicts 1, true label is 0)
            if worst_pred == 1 and true_label == 0:
                overpredict_equivalence += 1
            # Count underprediction (worst predicts 0, true label is 1)
            elif worst_pred == 0 and true_label == 1:
                underpredict_equivalence += 1

    # Print the counts
    print(f"Found {len(differing_examples)} examples where the best model succeeded but the worst failed:")
    print(f"Overpredictions by worst model (predicted 1, true label 0): {overpredict_equivalence}")
    print(f"Underpredictions by worst model (predicted 0, true label 1): {underpredict_equivalence}")

    # Print the differing examples
    for example in differing_examples[:15]:  # Print up to 15 examples for analysis
        print(f"\nExample {example['index']}:")
        print(f"Sentence 1: {example['sentence1']}")
        print(f"Sentence 2: {example['sentence2']}")
        print(f"True Label: {example['true_label']} (1=equivalent, 0=not equivalent)")
        print(f"Best Model ({best_config_name}) Prediction: {example['best_pred']}")
        print(f"Worst Model ({worst_config_name}) Prediction: {example['worst_pred']}")

    # Save all differing examples to a file for further analysis
    with open(output_file, "w") as f:
        f.write(f"Analysis comparing {best_predictions_file} (best) and {worst_predictions_file} (worst)\n\n")
        f.write(f"Found {len(differing_examples)} examples where the best model succeeded but the worst failed:\n")
        f.write(f"Overpredictions by worst model (predicted 1, true label 0): {overpredict_equivalence}\n")
        f.write(f"Underpredictions by worst model (predicted 0, true label 1): {underpredict_equivalence}\n\n")

        for example in differing_examples:
            f.write(f"Example {example['index']}:\n")
            f.write(f"Sentence 1: {example['sentence1']}\n")
            f.write(f"Sentence 2: {example['sentence2']}\n")
            f.write(f"True Label: {example['true_label']} (1=equivalent, 0=not equivalent)\n")
            f.write(f"Best Model ({best_config_name}) Prediction: {example['best_pred']}\n")
            f.write(f"Worst Model ({worst_config_name}) Prediction: {example['worst_pred']}\n\n")

    print(f"Analysis results saved to {output_file}")


def main():
    """
    Main function to run qualitative analysis comparing the best and worst models.
    """
    args = parse_arguments()
    raw_dataset = load_and_select_dataset()

    perform_qualitative_analysis(
        raw_dataset,
        args.best_predictions_file,
        args.worst_predictions_file,
        args.output_file,
        args.best_config_name,
        args.worst_config_name
    )


if __name__ == "__main__":
    main()