import json
from pathlib import Path
import pandas as pd

def main():
    # Initialize empty list for metrics
    all_metrics = []

    # Load all metrics files from the metrics directory
    metrics_dir = Path("metrics")
    for metrics_path in metrics_dir.glob("**/metrics_*.json"):
        with open(metrics_path) as f:
            metrics = json.load(f)
            all_metrics.append(metrics)

    if not all_metrics:
        print("No metrics files found!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Sort by accuracy
    df_sorted = df.sort_values('accuracy', ascending=False)

    # Print all results
    print("\nAll Models Performance:")
    print(df_sorted.to_string(index=False))

    # Get best model
    best_model = df_sorted.iloc[0]

    print("\nBest Model Configuration:")
    print(f"Learning Rate: {best_model['learning_rate']}")
    print(f"Batch Size: {best_model['batch_size']}")
    print(f"Test Accuracy: {best_model['accuracy']:.4f}")
    print(f"Test Loss: {best_model['test_loss']:.4f}")

    # Save comparison results
    df_sorted.to_csv("model_comparison.csv", index=False)
    print("\nDetailed results saved to 'model_comparison.csv'")

if __name__ == "__main__":
    main()
