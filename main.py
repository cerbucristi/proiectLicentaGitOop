from dataset_loader import MegaVulDataset
from logistic_regression_model import LogisticRegressionModel
from codebert_model import CodeBertModel
from code_processor import CodePreprocessor
from data_pipeline import DataPipeline

# ============================================================================
# 1. LOAD AND BALANCE DATASET
# ============================================================================
print("Loading dataset...")
dataset = MegaVulDataset()
samples = dataset.load()

print("\nBalancing dataset...")
balanced_samples = DataPipeline.balance_dataset(samples)

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
print("\n" + "="*80)
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
logreg_val_metrics = DataPipeline.calculate_metrics(val_labels, logreg_val_preds, "LogReg - VALIDATION SET")

# ============================================================================
# 4. TRAIN AND EVALUATE CODEBERT
# ============================================================================
print("\n" + "="*80)
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

# ============================================================================
# 5. COMPARE MODELS ON VALIDATION SET
# ============================================================================
DataPipeline.print_comparison(logreg_val_metrics, codebert_val_metrics)

# Save models
print("Saving models...")
logreg.save("artifacts/classical/model.pkl")
codebert.save("artifacts/codebert/model")
print("Models saved successfully!")




# model = CodeBertModel()
# model.train(samples)      #  samples folosite
# prob = model.predict(code)
# model.save("artifacts/codebert")