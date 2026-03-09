import argparse
import os
import pandas as pd


def balance_csv(
    input_csv,
    output_csv=None,
    label_column="label",
    text_column="code",
    random_state=42,
):
    df = pd.read_csv(input_csv)

    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in input CSV.")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in input CSV.")

    working_df = df[[text_column, label_column]].copy()

    working_df[text_column] = (
        working_df[text_column]
        .astype(str)
        .str.replace(r'^"(.*)"$', r'\1', regex=True)
    )

    label_counts = working_df[label_column].value_counts()
    if len(label_counts) < 2:
        raise ValueError("Input CSV must contain at least 2 classes to balance.")

    min_count = int(label_counts.min())
    balanced_parts = []
    for class_value in label_counts.index:
        class_df = working_df[working_df[label_column] == class_value]
        balanced_parts.append(class_df.sample(n=min_count, random_state=random_state))

    balanced_df = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    if output_csv is None:
        base_dir = os.path.dirname(input_csv) or "."
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(base_dir, f"{base_name}_balanced.csv")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    balanced_df.to_csv(output_csv, index=False)

    print("\n=== CSV Balancing ===")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Original rows: {len(working_df)} | class counts: {label_counts.to_dict()}")
    print(f"Balanced rows: {len(balanced_df)} | class counts: {balanced_df[label_column].value_counts().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance a CSV by downsampling each class to minority count.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV")
    parser.add_argument("--output-csv", default="", help="Optional output CSV path")
    parser.add_argument("--text-column", default="code", help="Name of text column")
    parser.add_argument("--label-column", default="label", help="Name of label column")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    balance_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv or None,
        label_column=args.label_column,
        text_column=args.text_column,
        random_state=args.random_state,
    )
