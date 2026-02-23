from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np

class DataPipeline:
    """Handle dataset balancing, splitting, and metrics calculation."""
    
    @staticmethod
    def balance_dataset(samples, random_state=42):
        """Balance dataset by downsampling the majority class."""
        vulnerable = [s for s in samples if s["label"] == 1]
        safe = [s for s in samples if s["label"] == 0]
        
        print(f"Before balancing: Vulnerable={len(vulnerable)}, Safe={len(safe)}")
        
        # Downsample majority class to match minority
        min_count = min(len(vulnerable), len(safe))
        
        np.random.seed(random_state)
        vulnerable = np.random.choice(vulnerable, size=min_count, replace=False).tolist()
        safe = np.random.choice(safe, size=min_count, replace=False).tolist()
        
        balanced = vulnerable + safe
        np.random.shuffle(balanced)
        
        print(f"After balancing: Total={len(balanced)}, Vulnerable={len(vulnerable)}, Safe={len(safe)}")
        return balanced
    
    @staticmethod
    def split_dataset(samples, test_size=0.2, random_state=42):
        """Split dataset into train and validation sets."""
        train_samples, val_samples = train_test_split(
            samples,
            test_size=test_size,
            random_state=random_state,
            stratify=[s["label"] for s in samples]  # maintain label distribution
        )
        
        print(f"Train set: {len(train_samples)} samples")
        print(f"Validation set: {len(val_samples)} samples")
        
        return train_samples, val_samples
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, set_name=""):
        """Calculate classification metrics."""
        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        }
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["tn"] = tn
        metrics["fp"] = fp
        metrics["fn"] = fn
        metrics["tp"] = tp
        
        print(f"\n{'='*60}")
        print(f"Metrics for {set_name}")
        print(f"{'='*60}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TN: {tn:4d}")
        print(f"{'='*60}\n")
        
        return metrics
    
    @staticmethod
    def print_comparison(logreg_metrics, codebert_metrics):
        """Compare metrics between two models."""
        print("\n" + "="*80)
        print("MODEL COMPARISON - VALIDATION SET")
        print("="*80)
        print(f"{'Metric':<15} {'LogisticRegression':<25} {'CodeBERT':<25} {'Winner':<15}")
        print("-"*80)
        
        for metric in ["precision", "recall", "f1", "accuracy"]:
            logreg_val = logreg_metrics[metric]
            codebert_val = codebert_metrics[metric]
            winner = "CodeBERT ✓" if codebert_val > logreg_val else "LogReg ✓" if logreg_val > codebert_val else "TIE"
            
            print(f"{metric:<15} {logreg_val:<25.4f} {codebert_val:<25.4f} {winner:<15}")
        
        print("="*80 + "\n")
