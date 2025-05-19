import pandas as pd
from collections import Counter
from tabulate import tabulate
import yaml
import os


def main():
    # Config
    def load_config(path="config/config.yaml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    config = load_config()
    time_stride = config["time_stride"]
    noise_std = config["noise_std"]

    # CSV
    df = pd.read_csv("outputs/comparison_results.csv")

    # Accuracy
    total = len(df)
    acc_sup = (df["true_word"] == df["supervised_word"]).sum() / total
    acc_ass = (df["true_word"] == df["associative_word"]).sum() / total

    print("\nACCURACY")
    print(f"Supervised:  {acc_sup * 100:.2f}%")
    print(f"Associative: {acc_ass * 100:.2f}%")
    print(f"Time Stride: {time_stride}")
    print(f"Noise Std:   {noise_std}")

    # Errori comuni
    df_errors = df[df["true_word"] != df["associative_word"]]
    error_pairs = list(zip(df_errors["true_word"], df_errors["associative_word"]))
    error_counts = Counter(error_pairs)

    print("\nERRORI PIÙ COMUNI (Associative Model):")
    for (true, pred), count in error_counts.most_common(5):
        print(f"{true} → {pred} : {count} volte")

    # Confusion Matrix
    labels = sorted(list(set(df["true_word"]).union(set(df["associative_word"]))))
    conf_mat = pd.DataFrame(0, index=labels, columns=labels)
    for _, row in df.iterrows():
        true = row["true_word"]
        pred = row["associative_word"]
        conf_mat.loc[true, pred] += 1

    print("\nCONFUSION MATRIX (Associative Model):")
    print(tabulate(conf_mat, headers='keys', tablefmt='grid'))

    os.makedirs("outputs", exist_ok=True)
    conf_mat.to_csv("outputs/confusion_matrix_associative.csv")
    print("\nConfusion matrix salvata in 'outputs/confusion_matrix_associative.csv'")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()