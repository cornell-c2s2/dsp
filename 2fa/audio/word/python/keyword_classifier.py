import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns


class AudioClassifier:
    def __init__(self, sample_rate=16000, n_mfcc=13, max_length=500):
        """
        Initialize the audio classifier.

        Args:
            sample_rate: Sample rate for audio processing
            n_mfcc: Number of MFCC coefficients to extract
            max_length: Maximum length to pad/truncate MFCC sequences
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_length = max_length
        self.model = None
        self.scaler = StandardScaler()

    def extract_mfcc(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            n_fft = 512
            hop_length = 160
            win_length = 400
            n_mels = 40

            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.n_mfcc,  # 13
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window="hann",
                center=False,
                n_mels=n_mels,
                htk=True,
            )

            if mfccs.shape[1] < self.max_length:
                pad_width = self.max_length - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode="constant")
            else:
                mfccs = mfccs[:, : self.max_length]

            return mfccs.flatten().astype(np.float32)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None



    def load_data(self, pos_dir, neg_dir):
        """
        Load audio files from positive and negative directories.

        Args:
            pos_dir: Directory containing positive samples (with "stop")
            neg_dir: Directory containing negative samples (without "stop")

        Returns:
            features, labels
        """
        features = []
        labels = []

        # Check if directories exist
        if not os.path.exists(pos_dir):
            raise ValueError(f"Positive directory '{pos_dir}' does not exist!")
        if not os.path.exists(neg_dir):
            raise ValueError(f"Negative directory '{neg_dir}' does not exist!")

        # Load positive samples
        print(f"Loading positive samples from: {pos_dir}")
        all_pos_files = os.listdir(pos_dir)
        pos_files = [
            f
            for f in all_pos_files
            if f.endswith((".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC", ".opus"))
        ]

        print(f"  Found {len(all_pos_files)} total files, {len(pos_files)} audio files")
        if len(pos_files) == 0 and len(all_pos_files) > 0:
            print(f"  Sample files in directory: {all_pos_files[:5]}")
            print("  Note: Looking for .wav, .mp3, or .flac extensions")

        for filename in pos_files:
            filepath = os.path.join(pos_dir, filename)
            mfcc = self.extract_mfcc(filepath)
            if mfcc is not None:
                features.append(mfcc)
                labels.append(1)  # Positive class

        # Load negative samples
        print(f"Loading negative samples from: {neg_dir}")
        all_neg_files = os.listdir(neg_dir)
        neg_files = [
            f
            for f in all_neg_files
            if f.endswith((".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC", ".opus"))
        ]

        print(f"  Found {len(all_neg_files)} total files, {len(neg_files)} audio files")
        if len(neg_files) == 0 and len(all_neg_files) > 0:
            print(f"  Sample files in directory: {all_neg_files[:5]}")
            print("  Note: Looking for .wav, .mp3, or .flac extensions")

        for filename in neg_files:
            filepath = os.path.join(neg_dir, filename)
            mfcc = self.extract_mfcc(filepath)
            if mfcc is not None:
                features.append(mfcc)
                labels.append(0)  # Negative class

        pos_count = len([l for l in labels if l == 1])
        neg_count = len([l for l in labels if l == 0])
        print(
            f"\nSuccessfully loaded {pos_count} positive and {neg_count} negative samples"
        )

        if pos_count == 0 and neg_count == 0:
            raise ValueError(
                "No audio files loaded! Please check your data directories and file formats."
            )
        if pos_count == 0:
            raise ValueError(
                "No positive samples loaded! Please add audio files to the positive directory."
            )
        if neg_count == 0:
            raise ValueError(
                "No negative samples loaded! Please add audio files to the negative directory."
            )

        return np.array(features), np.array(labels)

    def build_model(self, input_shape):
        """
        Build a simple neural network for classification.

        Args:
            input_shape: Shape of input features

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(
            [
                layers.Input(shape=(input_shape,)),
                layers.Dense(4, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(2, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(2, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the neural network.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history
        """
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Build model
        self.model = self.build_model(X_train.shape[1])

        print("\nModel architecture:")
        self.model.summary()

        # Early stopping callback
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Train model
        print("\nTraining model...")
        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1,
        )

        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Test metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        results = self.model.evaluate(X_test_scaled, y_test, verbose=0)

        print("\n=== Test Results ===")
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")

        return results

    def predict(self, audio_path):
        """
        Predict whether an audio file contains "stop".

        Args:
            audio_path: Path to audio file

        Returns:
            Prediction (0 or 1) and confidence score
        """
        mfcc = self.extract_mfcc(audio_path)
        if mfcc is None:
            return None, None

        mfcc_scaled = self.scaler.transform(mfcc.reshape(1, -1))
        prediction = self.model.predict(mfcc_scaled, verbose=0)[0][0]

        return int(prediction > 0.5), prediction

    def get_ground_truth_from_filename(self, filename):
        """
        Extract ground truth label from filename.
        Files with 'stop' in the name are positive (1), others are negative (0).

        Args:
            filename: Audio filename

        Returns:
            Ground truth label (0 or 1)
        """
        return 1 if "stop" in filename.lower() else 0

    def predict_directory(self, test_dir):
        """
        Predict on all audio files in a directory.

        Args:
            test_dir: Directory containing test audio files
        """
        if not os.path.exists(test_dir):
            print(f"Error: Directory '{test_dir}' does not exist!")
            return

        print(f"\n=== Predictions for {test_dir} ===")
        all_files = os.listdir(test_dir)
        test_files = [
            f
            for f in all_files
            if f.endswith((".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC", ".opus"))
        ]

        if len(test_files) == 0:
            print(f"No audio files found in {test_dir}")
            if len(all_files) > 0:
                print(f"Files found: {all_files[:5]}")
                print("Note: Looking for .wav, .mp3, or .flac extensions")
            return

        print(f"Found {len(test_files)} audio files to classify\n")

        for filename in test_files:
            filepath = os.path.join(test_dir, filename)
            label, confidence = self.predict(filepath)

            if label is not None:
                class_name = (
                    "POSITIVE (contains 'stop')"
                    if label == 1
                    else "NEGATIVE (no 'stop')"
                )
                print(f"{filename}: {class_name} (confidence: {confidence:.4f})")

    def evaluate_directory(self, test_dir, save_report=True):
        """
        Evaluate model on a directory with automatic ground truth labeling.
        Files with 'stop' in the filename are considered positive, others negative.

        Args:
            test_dir: Directory containing test audio files
            save_report: Whether to save a detailed report

        Returns:
            Dictionary with metrics
        """
        if not os.path.exists(test_dir):
            print(f"Error: Directory '{test_dir}' does not exist!")
            return None

        print(f"\n{'=' * 70}")
        print(f"EVALUATING MODEL ON: {test_dir}")
        print(f"{'=' * 70}")
        print("Ground truth: Files with 'stop' in name = POSITIVE, others = NEGATIVE\n")

        # Get all audio files
        all_files = os.listdir(test_dir)
        test_files = [
            f
            for f in all_files
            if f.endswith((".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC", ".opus"))
        ]

        if len(test_files) == 0:
            print(f"No audio files found in {test_dir}")
            return None

        print(f"Processing {len(test_files)} audio files...\n")

        # Collect predictions and ground truth
        y_true = []
        y_pred = []
        y_scores = []
        results = []

        for filename in test_files:
            filepath = os.path.join(test_dir, filename)

            # Get ground truth from filename
            true_label = self.get_ground_truth_from_filename(filename)

            # Get prediction
            pred_label, confidence = self.predict(filepath)

            if pred_label is not None:
                y_true.append(true_label)
                y_pred.append(pred_label)
                y_scores.append(confidence)

                results.append(
                    {
                        "filename": filename,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "confidence": confidence,
                        "correct": true_label == pred_label,
                    }
                )

        if len(y_true) == 0:
            print("No valid predictions made!")
            return None

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Print results
        print(f"{'=' * 70}")
        print("OVERALL METRICS")
        print(f"{'=' * 70}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(
            f"Precision: {precision:.4f} (of predicted positives, how many are correct)"
        )
        print(f"Recall:    {recall:.4f} (of actual positives, how many detected)")
        print(f"F1 Score:  {f1:.4f} (harmonic mean of precision and recall)")

        # Confusion matrix breakdown
        tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)
        print(f"\n{'=' * 70}")
        print("CONFUSION MATRIX")
        print(f"{'=' * 70}")
        print(f"True Negatives:  {tn} (correctly predicted as negative)")
        print(f"False Positives: {fp} (incorrectly predicted as positive)")
        print(f"False Negatives: {fn} (incorrectly predicted as negative)")
        print(f"True Positives:  {tp} (correctly predicted as positive)")

        # Per-class accuracy
        print(f"\n{'=' * 70}")
        print("PER-CLASS BREAKDOWN")
        print(f"{'=' * 70}")
        total_positive = np.sum(y_true == 1)
        total_negative = np.sum(y_true == 0)
        correct_positive = np.sum((y_true == 1) & (y_pred == 1))
        correct_negative = np.sum((y_true == 0) & (y_pred == 0))

        print(f"Positive class (contains 'stop'):")
        print(f"  Total: {total_positive}")
        print(f"  Correct: {correct_positive}")
        print(
            f"  Accuracy: {correct_positive / total_positive:.4f}"
            if total_positive > 0
            else "  Accuracy: N/A"
        )

        print(f"\nNegative class (no 'stop'):")
        print(f"  Total: {total_negative}")
        print(f"  Correct: {correct_negative}")
        print(
            f"  Accuracy: {correct_negative / total_negative:.4f}"
            if total_negative > 0
            else "  Accuracy: N/A"
        )

        # Show misclassified examples
        print(f"\n{'=' * 70}")
        print("MISCLASSIFIED FILES")
        print(f"{'=' * 70}")
        misclassified = [r for r in results if not r["correct"]]
        if len(misclassified) > 0:
            print(f"Found {len(misclassified)} misclassified files:\n")
            for r in misclassified:
                true_class = "POSITIVE" if r["true_label"] == 1 else "NEGATIVE"
                pred_class = "POSITIVE" if r["pred_label"] == 1 else "NEGATIVE"
                print(f"  {r['filename']}")
                print(
                    f"    True: {true_class}, Predicted: {pred_class}, Confidence: {r['confidence']:.4f}"
                )
        else:
            print("No misclassified files - perfect accuracy!")

        # Create visualizations
        self._plot_evaluation_results(conf_matrix, y_scores, y_true, test_dir)

        # Save detailed report if requested
        if save_report:
            self._save_evaluation_report(
                results, accuracy, precision, recall, f1, conf_matrix, test_dir
            )

        # Package metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "total_samples": len(y_true),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "results": results,
        }

        return metrics

    def _plot_evaluation_results(self, conf_matrix, y_scores, y_true, test_dir):
        """
        Create visualization plots for evaluation results.

        Args:
            conf_matrix: Confusion matrix
            y_scores: Prediction confidence scores
            y_true: Ground truth labels
            test_dir: Test directory name (for plot title)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion Matrix Heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=axes[0],
            cbar_kws={"label": "Count"},
        )
        axes[0].set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("True Label", fontsize=12)
        axes[0].set_xlabel("Predicted Label", fontsize=12)

        # Confidence Distribution
        positive_scores = y_scores[y_true == 1]
        negative_scores = y_scores[y_true == 0]

        if len(positive_scores) > 0:
            axes[1].hist(
                positive_scores,
                bins=20,
                alpha=0.6,
                label="Positive (with stop)",
                color="green",
                edgecolor="black",
            )
        if len(negative_scores) > 0:
            axes[1].hist(
                negative_scores,
                bins=20,
                alpha=0.6,
                label="Negative (no stop)",
                color="red",
                edgecolor="black",
            )

        axes[1].axvline(
            x=0.5, color="black", linestyle="--", linewidth=2, label="Threshold (0.5)"
        )
        axes[1].set_title(
            "Prediction Confidence Distribution", fontsize=14, fontweight="bold"
        )
        axes[1].set_xlabel("Confidence Score", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/evaluation_results.png", dpi=150, bbox_inches="tight")
        print(f"\n✓ Evaluation plots saved as 'results/evaluation_results.png'")
        plt.close()

    def _save_evaluation_report(
        self, results, accuracy, precision, recall, f1, conf_matrix, test_dir
    ):
        """
        Save a detailed evaluation report to a text file.

        Args:
            results: List of prediction results
            accuracy, precision, recall, f1: Metric values
            conf_matrix: Confusion matrix
            test_dir: Test directory name
        """
        report_path = "results/evaluation_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("AUDIO CLASSIFIER EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Test Directory: {test_dir}\n")
            f.write(f"Total Files: {len(results)}\n")
            f.write(f"Labeling Rule: Files with 'stop' in name = POSITIVE\n\n")

            f.write("=" * 70 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n\n")

            tn, fp, fn, tp = (
                conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)
            )
            f.write("=" * 70 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("=" * 70 + "\n")
            f.write(f"True Negatives:  {tn}\n")
            f.write(f"False Positives: {fp}\n")
            f.write(f"False Negatives: {fn}\n")
            f.write(f"True Positives:  {tp}\n\n")

            f.write("=" * 70 + "\n")
            f.write("DETAILED RESULTS PER FILE\n")
            f.write("=" * 70 + "\n\n")

            # Sort by correctness (incorrect first) and confidence
            sorted_results = sorted(
                results, key=lambda x: (x["correct"], -x["confidence"])
            )

            for r in sorted_results:
                status = "✓" if r["correct"] else "✗"
                true_class = "POSITIVE" if r["true_label"] == 1 else "NEGATIVE"
                pred_class = "POSITIVE" if r["pred_label"] == 1 else "NEGATIVE"

                f.write(f"{status} {r['filename']}\n")
                f.write(
                    f"   True: {true_class:8s} | Predicted: {pred_class:8s} | Confidence: {r['confidence']:.4f}\n\n"
                )

        print(f"✓ Detailed report saved as '{report_path}'")

    def plot_training_history(self, history):
        """
        Plot training history.

        Args:
            history: Training history from model.fit()
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Accuracy
        axes[0, 0].plot(history.history["accuracy"], label="Train")
        axes[0, 0].plot(history.history["val_accuracy"], label="Validation")
        axes[0, 0].set_title("Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(history.history["loss"], label="Train")
        axes[0, 1].plot(history.history["val_loss"], label="Validation")
        axes[0, 1].set_title("Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(history.history["precision"], label="Train")
        axes[1, 0].plot(history.history["val_precision"], label="Validation")
        axes[1, 0].set_title("Precision")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(history.history["recall"], label="Train")
        axes[1, 1].plot(history.history["val_recall"], label="Validation")
        axes[1, 1].set_title("Recall")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("results/training_history.png")
        print("\nTraining history plot saved as 'training_history.png'")

    def save_model(self, filepath="models/audio_classifier_model.keras"):
        """Save the trained model."""
        self.model.save(filepath)
        np.save("models/scaler_mean.npy", self.scaler.mean_)
        np.save("models/scaler_scale.npy", self.scaler.scale_)
        print(f"\nModel saved to {filepath}")

    def load_model(self, filepath="models/audio_classifier_model.keras"):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        self.scaler.mean_ = np.load("models/scaler_mean.npy")
        self.scaler.scale_ = np.load("models/scaler_scale.npy")
        print(f"\nModel loaded from {filepath}")


def main():
    # Initialize classifier
    classifier = AudioClassifier(n_mfcc=13, max_length=500)

    # Load data
    X, y = classifier.load_data("../../data/pos", "../../data/neg")

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nDataset split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Train model
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=50)

    # Evaluate on test set
    classifier.evaluate(X_test, y_test)

    # Plot training history
    classifier.plot_training_history(history)

    # Save model
    classifier.save_model()

    # Evaluate on testing directory with metrics
    if os.path.exists("../../data/testing"):
        print("\n" + "=" * 70)
        print("TESTING ON UNLABELED DATA")
        print("=" * 70)
        metrics = classifier.evaluate_directory("../../data/testing", save_report=True)

        if metrics:
            print(f"\n{'=' * 70}")
            print("EVALUATION COMPLETE")
            print(f"{'=' * 70}")
            print(f"✓ Results saved to 'results/evaluation_results.png'")
            print(f"✓ Detailed report saved to 'results/evaluation_report.txt'")
    else:
        print("\nNo '../../data/testing' directory found. Skipping evaluation.")


if __name__ == "__main__":
    main()
