"""
Utility for dataset consistency checks on raw text + label pairs.

Purpose:
- Load samples (e.g., MegaVul) and build a pandas table with two columns: text and label.
- Group by raw text to detect duplicates and label conflicts (same text mapped to both labels).
- Export lightweight CSV reports for duplicate/conflicting text groups.

Typical use:
- Quick consistency audit before/after feature engineering for classical models (ID3/LogReg).
"""

import os
import argparse
import pandas as pd
from dataset_loader import MegaVulDataset


class TextLabelConflictAnalyzer:
    """Analyze duplicate and conflicting labels by grouping rows on raw text."""

    def __init__(self, text_column="code", label_column="label"):
        self.text_column = text_column
        self.label_column = label_column

    def build_dataframe(self, samples):
        rows = [
            {
                self.text_column: sample[self.text_column],
                self.label_column: sample[self.label_column],
            }
            for sample in samples
        ]
        return pd.DataFrame(rows)

    def analyze(self, df):
        grouped = df.groupby(self.text_column)[self.label_column].agg(
            occurrences="size",
            distinct_labels="nunique",
            label_0_count=lambda s: int((s == 0).sum()),
            label_1_count=lambda s: int((s == 1).sum()),
        )

        duplicates_df = grouped[grouped["occurrences"] > 1].sort_values(
            by="occurrences", ascending=False
        )
        conflicts_df = grouped[grouped["distinct_labels"] > 1].sort_values(
            by="occurrences", ascending=False
        )

        summary = {
            "total_rows": int(len(df)),
            "unique_texts": int(df[self.text_column].nunique()),
            "duplicate_text_groups": int(len(duplicates_df)),
            "conflicting_text_groups": int(len(conflicts_df)),
        }

        return summary, duplicates_df, conflicts_df

    def build_source_of_truth(self, grouped_df):
        non_conflicting = grouped_df[grouped_df["distinct_labels"] == 1].copy()
        non_conflicting["label"] = (non_conflicting["label_1_count"] > 0).astype(int)

        source_of_truth_df = (
            non_conflicting.reset_index()[[self.text_column, "label", "occurrences"]]
            .sort_values(by="occurrences", ascending=False)
            .reset_index(drop=True)
        )
        return source_of_truth_df

    def build_balanced_source_of_truth(self, source_of_truth_df, random_state=42):
        if source_of_truth_df.empty:
            return source_of_truth_df.copy()

        label_counts = source_of_truth_df["label"].value_counts()
        if len(label_counts) < 2:
            print("Warning: only one label present in source-of-truth; balancing skipped.")
            return source_of_truth_df.copy()

        min_count = int(label_counts.min())
        balanced_df = (
            source_of_truth_df
            .groupby("label", group_keys=False)
            .apply(lambda group: group.sample(n=min_count, random_state=random_state))
            .sample(frac=1.0, random_state=random_state)
            .reset_index(drop=True)
        )
        return balanced_df

    def run(self, samples, output_dir="artifacts/classical/id3/text_conflicts", save_all_rows=False):
        os.makedirs(output_dir, exist_ok=True)

        df = self.build_dataframe(samples)
        summary, duplicates_df, conflicts_df = self.analyze(df)

        grouped = df.groupby(self.text_column)[self.label_column].agg(
            occurrences="size",
            distinct_labels="nunique",
            label_0_count=lambda s: int((s == 0).sum()),
            label_1_count=lambda s: int((s == 1).sum()),
        )
        source_of_truth_df = self.build_source_of_truth(grouped)

        all_rows_path = os.path.join(output_dir, "all_text_label_rows.csv")
        duplicates_path = os.path.join(output_dir, "duplicate_text_groups.csv")
        conflicts_path = os.path.join(output_dir, "conflicting_text_groups.csv")
        source_of_truth_path = os.path.join(output_dir, "source_of_truth_non_conflicting_texts.csv")
        balanced_source_of_truth_path = os.path.join(
            output_dir,
            "source_of_truth_non_conflicting_texts_balanced.csv"
        )

        if save_all_rows:
            df.to_csv(all_rows_path, index=False)
            duplicates_df.to_csv(duplicates_path)
            conflicts_df.to_csv(conflicts_path)
            source_of_truth_df.to_csv(source_of_truth_path, index=False)
        balanced_source_of_truth_df = self.build_balanced_source_of_truth(source_of_truth_df)
        balanced_source_of_truth_df.to_csv(balanced_source_of_truth_path, index=False)

        print("\n=== Text/Label Conflict Analysis ===")
        print(f"Total rows: {summary['total_rows']}")
        print(f"Unique texts: {summary['unique_texts']}")
        print(f"Duplicate text groups: {summary['duplicate_text_groups']}")
        print(f"Conflicting text groups: {summary['conflicting_text_groups']}")
        print(f"Source-of-truth rows (1 row per non-conflicting unique text): {len(source_of_truth_df)}")
        if not balanced_source_of_truth_df.empty:
            balanced_counts = balanced_source_of_truth_df["label"].value_counts().to_dict()
            print(f"Balanced source-of-truth rows: {len(balanced_source_of_truth_df)} | label counts: {balanced_counts}")
        if save_all_rows:
            print(f"Saved rows to: {all_rows_path}")
        print(f"Saved duplicates to: {duplicates_path}")
        print(f"Saved conflicts to: {conflicts_path}")
        print(f"Saved source-of-truth to: {source_of_truth_path}")
        print(f"Saved balanced source-of-truth to: {balanced_source_of_truth_path}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze duplicate/conflicting labels by grouping raw text with pandas."
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/classical/id3/text_conflicts",
        help="Directory where CSV reports are saved",
    )
    parser.add_argument(
        "--save-all-rows",
        action="store_true",
        help="Also save all raw text-label rows to CSV (can be large)",
    )
    args = parser.parse_args()

    dataset = MegaVulDataset()
    samples = dataset.load(limit=args.limit)

    analyzer = TextLabelConflictAnalyzer(text_column="code", label_column="label")
    analyzer.run(samples, output_dir=args.output_dir, save_all_rows=args.save_all_rows)


if __name__ == "__main__":
    main()
