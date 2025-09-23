#!/usr/bin/env python3
"""
Risk Model for CancerScope - Tabular Data Analysis with SHAP Explanations
Uses scikit-learn's breast cancer dataset to train a Logistic Regression model
CLI Commands:
  python risk_model.py predict <csv_file> [shap]
  python risk_model.py metrics
  python risk_model.py train
"""

import sys
import json
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP integration
try:
    import shap

    SHAP_AVAILABLE = True
    print("SHAP library loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP library not available. Install with: pip install shap")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class BreastCancerRiskModel:
    def __init__(self, model_type="logistic"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.explainer = None
        self.X_background = None
        self.model_path = "models/risk_model.pkl"
        self.scaler_path = "models/risk_scaler.pkl"
        self.metrics_path = "models/risk_metrics.json"
        self.explainer_path = "models/risk_explainer.pkl"
        self.background_path = "models/risk_background.pkl"

        # Ensure models and results directories exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("results/shap_plots", exist_ok=True)

    def load_data(self):
        """Load and prepare the breast cancer dataset"""
        data = load_breast_cancer()
        X, y = data.data, data.target
        self.feature_names = data.feature_names.tolist()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train_model(self):
        """Train the risk prediction model with SHAP explainer"""
        print("Loading breast cancer dataset...")
        X_train, X_test, y_train, y_test = self.load_data()

        # Scale the features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Choose model
        if self.model_type == "logistic":
            print("Training Logistic Regression model...")
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            print("Training Random Forest model...")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Create SHAP explainer
        if SHAP_AVAILABLE:
            print("Creating SHAP explainer...")
            try:
                # Use a subset of training data as background for efficiency
                background_size = min(100, len(X_train_scaled))
                self.X_background = X_train_scaled[:background_size]

                if hasattr(self.model, "predict_proba"):
                    self.explainer = shap.Explainer(
                        self.model.predict_proba, self.X_background
                    )
                else:
                    self.explainer = shap.Explainer(
                        self.model.predict, self.X_background
                    )

                # Save explainer and background data
                joblib.dump(self.explainer, self.explainer_path)
                joblib.dump(self.X_background, self.background_path)
                print("SHAP explainer created and saved successfully")
            except Exception as e:
                print(f"Warning: Could not create SHAP explainer: {e}")
                self.explainer = None

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Save model, scaler, and metrics
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Model trained and saved!")
        print(f"Test Accuracy: {metrics['accuracy']:.3f}")
        print(f"ROC-AUC: {metrics['rocAuc']:.3f}")

        return metrics

    def load_model(self):
        """Load the trained model, scaler, and SHAP explainer"""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            print("Model not found. Training new model...")
            self.train_model()

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        # Load SHAP explainer if available
        if SHAP_AVAILABLE and os.path.exists(self.explainer_path):
            try:
                self.explainer = joblib.load(self.explainer_path)
                if os.path.exists(self.background_path):
                    self.X_background = joblib.load(self.background_path)
                print("SHAP explainer loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load SHAP explainer: {e}")
                self.explainer = None

        # Load feature names
        data = load_breast_cancer()
        self.feature_names = data.feature_names.tolist()

    def generate_shap_explanation(self, sample_data, patient_id):
        """Generate SHAP explanation for a single sample with enhanced visualization"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return None, None, None

        try:
            # Calculate SHAP values
            shap_values = self.explainer(sample_data)

            # Handle different SHAP value formats
            if hasattr(shap_values, "values"):
                if len(shap_values.values.shape) > 2:
                    # Binary classification - use positive class (malignant)
                    shap_vals = shap_values.values[0, :, 1]
                    base_value = shap_values.base_values[0][1]
                else:
                    shap_vals = shap_values.values[0]
                    base_value = (
                        shap_values.base_values[0]
                        if hasattr(shap_values, "base_values")
                        else 0
                    )
            else:
                shap_vals = shap_values[0]
                base_value = 0

            # Create enhanced waterfall plot
            plt.figure(figsize=(14, 10))

            # Get feature importance and sort by absolute value
            feature_importance = dict(zip(self.feature_names, shap_vals))
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )[:15]  # Top 15 features

            features, values = zip(*sorted_features)

            # Create colors based on positive/negative contribution
            colors = ["#ff4444" if v > 0 else "#4444ff" for v in values]

            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            bars = plt.barh(
                y_pos, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5
            )

            # Customize the plot
            plt.yticks(y_pos, [f.replace("_", " ").title() for f in features])
            plt.xlabel(
                "SHAP Value (Impact on Prediction)", fontsize=12, fontweight="bold"
            )
            plt.title(
                f"SHAP Feature Importance - {patient_id}",
                fontsize=14,
                fontweight="bold",
            )
            plt.grid(True, alpha=0.3, axis="x")

            # Add value labels on bars
            for bar, value in zip(bars, values):
                width = bar.get_width()
                plt.text(
                    width + (0.01 if width >= 0 else -0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    ha="left" if width >= 0 else "right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

            # Add legend
            positive_patch = plt.Rectangle((0, 0), 1, 1, color="#ff4444", alpha=0.8)
            negative_patch = plt.Rectangle((0, 0), 1, 1, color="#4444ff", alpha=0.8)
            plt.legend(
                [positive_patch, negative_patch],
                ["Increases Cancer Risk", "Decreases Cancer Risk"],
                loc="lower right",
                fontsize=10,
            )

            # Add baseline information
            prediction = self.model.predict_proba(sample_data)[0][1]
            plt.figtext(
                0.02,
                0.02,
                f"Base Value: {base_value:.3f} | Final Prediction: {prediction:.3f}",
                fontsize=10,
                style="italic",
            )

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = (
                f"{patient_id.lower().replace(' ', '_')}_shap_{timestamp}.png"
            )
            plot_path = os.path.join("results", "shap_plots", plot_filename)

            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close()

            # Get top contributing features
            top_features = [f[0] for f in sorted_features[:5]]

            # Create detailed explanation text
            explanation = self.create_explanation_text(
                feature_importance, patient_id, prediction
            )

            return (
                f"/results/shap_plots/{plot_filename}",
                top_features,
                explanation,
            )

        except Exception as e:
            print(f"Error generating SHAP explanation for {patient_id}: {e}")
            import traceback

            traceback.print_exc()
            return None, None, None

    def create_explanation_text(self, feature_importance, patient_id, prediction):
        """Create human-readable explanation text with more detail"""
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )

        explanation = f"SHAP Analysis for {patient_id}:\n"
        explanation += f"Predicted Probability of Malignancy: {prediction:.1%}\n"
        explanation += (
            f"Predicted Class: {'Malignant' if prediction > 0.5 else 'Benign'}\n\n"
        )

        positive_features = [(f, v) for f, v in sorted_features if v > 0][:5]
        negative_features = [(f, v) for f, v in sorted_features if v < 0][:5]

        if positive_features:
            explanation += "Top factors INCREASING cancer risk:\n"
            for i, (feature, value) in enumerate(positive_features, 1):
                explanation += (
                    f"{i}. {feature.replace('_', ' ').title()}: +{value:.4f}\n"
                )
            explanation += "\n"

        if negative_features:
            explanation += "Top factors DECREASING cancer risk:\n"
            for i, (feature, value) in enumerate(negative_features, 1):
                explanation += (
                    f"{i}. {feature.replace('_', ' ').title()}: {value:.4f}\n"
                )
            explanation += "\n"

        explanation += "Note: SHAP values show how each feature contributes to the final prediction.\n"
        explanation += (
            "Positive values push toward malignant, negative values push toward benign."
        )

        return explanation

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive model metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)  # This is sensitivity
        f1 = f1_score(y_true, y_pred)

        # ROC-AUC and PR-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        # Confusion matrix for specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "sensitivity": float(recall),  # Same as recall
            "specificity": float(specificity),
            "f1Score": float(f1),
            "rocAuc": float(roc_auc),
            "prAuc": float(pr_auc),
            "confusionMatrix": {
                "trueNegatives": int(tn),
                "falsePositives": int(fp),
                "falseNegatives": int(fn),
                "truePositives": int(tp),
            },
        }

    def predict_csv(self, csv_path, use_shap=True):
        """Make predictions on uploaded CSV file with SHAP explanations by default"""
        try:
            # Load the model
            self.load_model()

            # Load and validate CSV
            df = pd.read_csv(csv_path)

            if df.empty:
                raise ValueError("CSV file is empty")

            # Store original patient IDs if available
            if "id" in df.columns:
                patient_ids = df["id"].astype(str).tolist()
                df = df.drop("id", axis=1)
            elif "ID" in df.columns:
                patient_ids = df["ID"].astype(str).tolist()
                df = df.drop("ID", axis=1)
            else:
                patient_ids = [f"Patient_{i + 1:03d}" for i in range(len(df))]

            # Remove diagnosis column if present
            if "diagnosis" in df.columns:
                df = df.drop("diagnosis", axis=1)
            if "target" in df.columns:
                df = df.drop("target", axis=1)

            # Ensure we have the right number of features
            if len(df.columns) != len(self.feature_names):
                # If column names don't match, assume they're in the correct order
                if len(df.columns) == len(self.feature_names):
                    df.columns = self.feature_names
                else:
                    raise ValueError(
                        f"Expected {len(self.feature_names)} features, got {len(df.columns)}"
                    )

            # Handle missing values
            df = df.fillna(df.mean())

            # Scale the features
            X_scaled = self.scaler.transform(df.values)

            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)

            # Format results
            results = []
            for i, (patient_id, pred, prob) in enumerate(
                zip(patient_ids, predictions, probabilities)
            ):
                # prob[1] is probability of malignant (class 1)
                malignant_prob = float(prob[1])
                predicted_class = "Malignant" if pred == 1 else "Benign"

                result = {
                    "patientId": patient_id,
                    "predictedClass": predicted_class,
                    "probability": malignant_prob,
                }

                # Always try to add SHAP explanations
                if use_shap and SHAP_AVAILABLE and self.explainer is not None:
                    sample_data = X_scaled[i : i + 1]  # Single sample for SHAP
                    shap_url, top_features, explanation = (
                        self.generate_shap_explanation(sample_data, patient_id)
                    )

                    result.update(
                        {
                            "shapPlotUrl": shap_url,
                            "topFeatures": top_features or [],
                            "explanationText": explanation
                            or f"SHAP explanation for {patient_id}",
                            "featureImportances": dict(
                                zip(self.feature_names, sample_data[0])
                            )
                            if top_features
                            else {},
                        }
                    )
                else:
                    result.update(
                        {
                            "shapPlotUrl": None,
                            "topFeatures": [],
                            "explanationText": "SHAP explanations not available - SHAP library may not be installed",
                            "featureImportances": {},
                        }
                    )

                results.append(result)

            return {"predictions": results, "status": "success"}

        except Exception as e:
            print(f"Error in predict_csv: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e), "predictions": []}

    def get_metrics(self):
        """Get model performance metrics"""
        try:
            # Load metrics if available
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r") as f:
                    metrics = json.load(f)
            else:
                # Train model and get metrics
                print("Metrics not found. Training model...")
                metrics = self.train_model()

            # Return in the format expected by frontend
            return {
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "rocAuc": metrics["rocAuc"],
                "prAuc": metrics["prAuc"],
                "accuracy": metrics["accuracy"],
                "status": "success",
            }

        except Exception as e:
            return {
                "error": str(e),
                "sensitivity": 0.0,
                "specificity": 0.0,
                "rocAuc": 0.0,
                "prAuc": 0.0,
            }


def main():
    try:
        if len(sys.argv) < 2:
            print(
                json.dumps(
                    {
                        "error": "Usage: python risk_model.py <command> [args]",
                        "commands": ["predict <csv_file> [shap]", "metrics", "train"],
                    }
                )
            )
            sys.exit(1)

        command = sys.argv[1].lower()
        model = BreastCancerRiskModel(model_type="logistic")

        if command == "predict":
            if len(sys.argv) < 3:
                print(
                    json.dumps(
                        {
                            "error": "Usage: python risk_model.py predict <csv_file> [shap]"
                        }
                    )
                )
                sys.exit(1)

            csv_file = sys.argv[2]
            use_shap = len(sys.argv) > 3 and sys.argv[3].lower() == "shap"

            if not os.path.exists(csv_file):
                print(json.dumps({"error": f"CSV file not found: {csv_file}"}))
                sys.exit(1)

            result = model.predict_csv(csv_file, use_shap=use_shap)
            print(json.dumps(result))

        elif command == "metrics":
            result = model.get_metrics()
            print(json.dumps(result))

        elif command == "train":
            print("Training model...")
            result = model.train_model()
            print(
                json.dumps({"status": "Model trained successfully", "metrics": result})
            )

        else:
            print(
                json.dumps(
                    {
                        "error": f"Unknown command: {command}",
                        "available_commands": ["predict", "metrics", "train"],
                    }
                )
            )
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
