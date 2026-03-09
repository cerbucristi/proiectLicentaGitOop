from dataset_loader import MegaVulDataset
from logistic_regression_model import LogisticRegressionModel
from decision_tree_model import DecisionTreeModel
from codebert_model import CodeBertModel
from code_processor import CodePreprocessor
from data_pipeline import DataPipeline
import os


max_per_class_raw = os.getenv("MAX_PER_CLASS", "5000").strip().lower()
MAX_PER_CLASS = None if max_per_class_raw in {"", "0", "none", "all"} else int(max_per_class_raw)
SOURCE_DATASET_CSV = os.getenv("SOURCE_DATASET_CSV", "").strip()
ID3_OUTPUT_DIR = os.getenv(
	"ID3_OUTPUT_DIR",
	"artifacts/classical/id3FilteredInput" if SOURCE_DATASET_CSV else "artifacts/classical/id3"
)

# ============================================================================
# 1. LOAD AND BALANCE DATASET
# ============================================================================
print("Loading dataset...")
dataset = MegaVulDataset()
if SOURCE_DATASET_CSV:
	print(f"Loading source-of-truth dataset from CSV: {SOURCE_DATASET_CSV}")
	samples = dataset.load_from_csv(SOURCE_DATASET_CSV)
else:
	samples = dataset.load()

print("\nBalancing dataset...")
if SOURCE_DATASET_CSV:
	print("Skipping balancing because SOURCE_DATASET_CSV is provided.")
	balanced_samples = samples
else:
	if MAX_PER_CLASS is None:
		print("Using full balanced dataset (MAX_PER_CLASS disabled)")
	else:
		print(f"Using MAX_PER_CLASS={MAX_PER_CLASS} (total after balancing ≈ {MAX_PER_CLASS * 2})")
	balanced_samples = DataPipeline.balance_dataset(samples, max_per_class=MAX_PER_CLASS)

# ============================================================================
# 2. SPLIT INTO TRAIN AND VALIDATION (FIXED FOR BOTH MODELS)
# ============================================================================
print("\nSplitting into train and validation sets...")
train_samples, val_samples = DataPipeline.split_dataset(balanced_samples, test_size=0.2)

# Extract labels for metrics calculation
train_labels = [s["label"] for s in train_samples]
val_labels = [s["label"] for s in val_samples]

codeProcessor = CodePreprocessor()

# ============================================================================
# 3. TRAIN AND EVALUATE LOGISTIC REGRESSION
# ============================================================================
"""print("\n" + "="*80)
print("LOGISTIC REGRESSION MODEL")
print("="*80)

logreg = LogisticRegressionModel()
print("\nTraining LogisticRegression on train set...")
logreg.train(train_samples)

print("\nEvaluating LogisticRegression...")
# Convert probabilities to class labels for metrics.
logreg_train_preds = [1 if logreg.predict(s["code"]) > 0.5 else 0 for s in train_samples]
logreg_val_preds = [1 if logreg.predict(s["code"]) > 0.5 else 0 for s in val_samples]

logreg_train_metrics = DataPipeline.calculate_metrics(train_labels, logreg_train_preds, "LogReg - TRAIN SET")
logreg_val_metrics = DataPipeline.calculate_metrics(val_labels, logreg_val_preds, "LogReg - VALIDATION SET")"""

# ============================================================================
# 3.1 TRAIN AND EVALUATE ID3 (DECISION TREE) WITH TWO FEATURE SETS
# ============================================================================
print("\n" + "="*80)
print("ID3 (DECISION TREE) MODEL")
print("="*80)
print(f"ID3 output directory: {ID3_OUTPUT_DIR}")

id3_word = DecisionTreeModel(feature_mode="word_tfidf", output_dir=ID3_OUTPUT_DIR)
print("\nTraining ID3 (word_tfidf) on train set...")
id3_word.train(train_samples)

print("\nEvaluating ID3 (word_tfidf)...")
id3_word_train_preds = [1 if id3_word.predict(s["code"]) > 0.5 else 0 for s in train_samples]
id3_word_val_preds = [1 if id3_word.predict(s["code"]) > 0.5 else 0 for s in val_samples]

id3_word_train_metrics = DataPipeline.calculate_metrics(train_labels, id3_word_train_preds, "ID3(word_tfidf) - TRAIN SET")
id3_word_val_metrics = DataPipeline.calculate_metrics(val_labels, id3_word_val_preds, "ID3(word_tfidf) - VALIDATION SET")

id3_char = DecisionTreeModel(feature_mode="char_tfidf", output_dir=ID3_OUTPUT_DIR)
print("\nTraining ID3 (char_tfidf) on train set...")
id3_char.train(train_samples)

print("\nEvaluating ID3 (char_tfidf)...")
id3_char_train_preds = [1 if id3_char.predict(s["code"]) > 0.5 else 0 for s in train_samples]
id3_char_val_preds = [1 if id3_char.predict(s["code"]) > 0.5 else 0 for s in val_samples]

id3_char_train_metrics = DataPipeline.calculate_metrics(train_labels, id3_char_train_preds, "ID3(char_tfidf) - TRAIN SET")
id3_char_val_metrics = DataPipeline.calculate_metrics(val_labels, id3_char_val_preds, "ID3(char_tfidf) - VALIDATION SET")

# ============================================================================
# 4. TRAIN AND EVALUATE CODEBERT
# ============================================================================
"""print("\n" + "="*80)
print("CODEBERT MODEL")
print("="*80)

codebert = CodeBertModel()
print("\nTraining CodeBERT on train set...")
codebert.train(train_samples)

print("\nEvaluating CodeBERT...")
codebert_train_preds = [1 if codebert.predict(s["code"]) > 0.5 else 0 for s in train_samples]
codebert_val_preds = [1 if codebert.predict(s["code"]) > 0.5 else 0 for s in val_samples]

codebert_train_metrics = DataPipeline.calculate_metrics(train_labels, codebert_train_preds, "CodeBERT - TRAIN SET")
codebert_val_metrics = DataPipeline.calculate_metrics(val_labels, codebert_val_preds, "CodeBERT - VALIDATION SET")
"""
# ============================================================================
# 5. COMPARE MODELS ON VALIDATION SET
# ============================================================================
# DataPipeline.print_comparison(logreg_val_metrics, codebert_val_metrics)

# Save models
print("Saving models...")
# logreg.save("artifacts/classical/model.pkl")
id3_word.save(os.path.join(ID3_OUTPUT_DIR, "id3_word_model.pkl"))
id3_char.save(os.path.join(ID3_OUTPUT_DIR, "id3_char_model.pkl"))
# codebert.save("artifacts/codebert/model")
print("Models saved successfully!")




# model = CodeBertModel()
# model.train(samples)      #  samples folosite
# prob = model.predict(code)
# model.save("artifacts/codebert")