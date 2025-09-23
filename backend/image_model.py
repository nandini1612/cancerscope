#!/usr/bin/env python3
"""
Modified Image Model for Real Medical Data
Example integration with medical imaging datasets
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pydicom  # For DICOM medical images
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path


class MedicalBreastCancerImageModel:
    def __init__(self):
        self.model = None
        # Updated for medical images - higher resolution
        self.input_shape = (224, 224, 3)  # Standard medical image size
        self.num_classes = 2
        self.model_path = "models/medical_image_model.h5"
        self.metrics_path = "models/metrics.json"
        self.class_names = ["Benign", "Malignant"]

    def load_ddsm_dataset(self, data_path):
        """
        Load DDSM or CBIS-DDSM dataset
        Expected structure:
        data_path/
        ├── benign/
        │   ├── image1.png
        │   └── image2.png
        └── malignant/
            ├── image1.png
            └── image2.png
        """
        images = []
        labels = []

        for class_idx, class_name in enumerate(["benign", "malignant"]):
            class_path = os.path.join(data_path, class_name)
            if not os.path.exists(class_path):
                continue

            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".dcm")
                ):
                    image_path = os.path.join(class_path, image_file)

                    # Load and preprocess image
                    if image_file.lower().endswith(".dcm"):
                        # Handle DICOM files
                        img = self.load_dicom_image(image_path)
                    else:
                        img = self.load_standard_image(image_path)

                    if img is not None:
                        images.append(img)
                        labels.append(class_idx)

        return np.array(images), np.array(labels)

    def load_dicom_image(self, dicom_path):
        """Load and preprocess DICOM medical image"""
        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array

            # Normalize pixel values
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # Convert to 3-channel if needed
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

            # Resize to target shape
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))

            return image

        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            return None

    def load_kaggle_histopathology(self, data_path):
        """
        Load Kaggle Breast Histopathology Images dataset
        Expected structure after download:
        data_path/
        └── IDC_regular_ps50_idx5/
            ├── 0/  (benign - IDC negative)
            └── 1/  (malignant - IDC positive)
        """
        images = []
        labels = []

        # Find the actual data directory
        base_path = Path(data_path)
        idc_dir = None

        # Look for IDC directory
        for item in base_path.iterdir():
            if item.is_dir() and "IDC" in item.name:
                idc_dir = item
                break

        if idc_dir is None:
            # Try direct structure
            if (base_path / "0").exists() and (base_path / "1").exists():
                idc_dir = base_path
            else:
                raise ValueError(
                    f"Could not find expected directory structure in {data_path}"
                )

        print(f"Loading from: {idc_dir}")

        # Load benign images (class 0)
        benign_dir = idc_dir / "0"
        malignant_dir = idc_dir / "1"

        for class_idx, (class_dir, class_name) in enumerate(
            [(benign_dir, "benign"), (malignant_dir, "malignant")]
        ):
            if not class_dir.exists():
                print(f"Warning: {class_name} directory not found: {class_dir}")
                continue

            print(f"Loading {class_name} images from {class_dir}")
            count = 0

            for image_file in class_dir.glob("*.png"):
                try:
                    img = self.load_standard_image(str(image_file))
                    if img is not None:
                        images.append(img)
                        labels.append(class_idx)
                        count += 1

                        # Limit for testing (remove this in production)
                        if count >= 5000:  # Limit per class for faster testing
                            break

                except Exception as e:
                    print(f"Error loading {image_file}: {e}")
                    continue

            print(f"Loaded {count} {class_name} images")

        return np.array(images), np.array(labels)

    def load_breakhis_dataset(self, data_path, magnification="40X"):
        """
        Load BreakHis histopathological dataset
        Expected structure:
        data_path/
        ├── benign/
        │   └── 40X/
        │       ├── adenosis/
        │       ├── fibroadenoma/
        │       └── ...
        └── malignant/
            └── 40X/
                ├── ductal_carcinoma/
                ├── lobular_carcinoma/
                └── ...
        """
        images = []
        labels = []

        for class_idx, class_name in enumerate(["benign", "malignant"]):
            class_path = os.path.join(data_path, class_name, magnification)
            if not os.path.exists(class_path):
                continue

            # Walk through subdirectories
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(root, file)
                        img = self.load_standard_image(image_path)

                        if img is not None:
                            images.append(img)
                            labels.append(class_idx)

        return np.array(images), np.array(labels)

    def load_busi_dataset(self, data_path):
        """Load BUSI (Breast Ultrasound Images) dataset"""
        images = []
        labels = []

        # BUSI has 3 classes: normal, benign, malignant
        # Combine normal + benign = 0 (benign), malignant = 1 (malignant)
        class_mapping = {"normal": 0, "benign": 0, "malignant": 1}

        base_path = Path(data_path)
        print(f"Loading BUSI dataset from: {base_path}")

        for class_name, class_label in class_mapping.items():
            class_path = base_path / class_name
            if class_path.exists():
                print(f"Loading {class_name} images...")
                count = 0
                for img_file in class_path.glob("*.png"):
                    try:
                        img = self.load_standard_image(str(img_file))
                        if img is not None:
                            images.append(img)
                            labels.append(class_label)
                            count += 1

                            # Limit for testing (remove in production)
                            if count >= 300:  # Limit per class for faster testing
                                break

                    except Exception as e:
                        print(f"Error loading {img_file}: {e}")
                        continue

                print(f"Loaded {count} {class_name} images")
            else:
                print(f"Warning: {class_name} directory not found at {class_path}")

        if len(images) == 0:
            raise ValueError(
                f"No images found in {data_path}. Check your BUSI dataset structure."
            )

        return np.array(images), np.array(labels)

    def create_medical_model(self):
        """Create CNN model optimized for medical images"""
        model = keras.Sequential(
            [
                # First block
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=self.input_shape,
                    name="conv2d_1",
                ),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Second block
                layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Third block
                layers.Conv2D(128, (3, 3), activation="relu", name="conv2d_3"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Fourth block
                layers.Conv2D(256, (3, 3), activation="relu", name="conv2d_4"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Classifier
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        # Use different optimizer and learning rate for medical images
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        return model

    def generate_medical_gradcam(self, image_path, patient_id="patient"):
        """Generate enhanced Grad-CAM visualization for medical images with overlay"""
        try:
            # Load and preprocess the original image
            if image_path.lower().endswith(".dcm"):
                img_original = self.load_dicom_image(image_path)
                # For display, convert back to 0-255 range
                img_display = (img_original * 255).astype(np.uint8)
            else:
                img_pil = Image.open(image_path)
                if img_pil.mode != "RGB":
                    img_pil = img_pil.convert("RGB")
                img_display = np.array(
                    img_pil.resize((self.input_shape[1], self.input_shape[0]))
                )
                img_original = self.load_standard_image(image_path)

            if img_original is None:
                return None, "Failed to load image"

            # Prepare input for model
            input_tensor = np.expand_dims(img_original, axis=0)

            # Find the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break

            if last_conv_layer is None:
                return None, "No convolutional layers found for Grad-CAM"

            # Create gradient model
            grad_model = tf.keras.models.Model(
                [self.model.inputs], [last_conv_layer.output, self.model.output]
            )

            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_tensor)
                loss = predictions[:, 0]  # Binary classification

            grads = tape.gradient(loss, conv_outputs)

            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight feature maps by gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            heatmap_np = heatmap.numpy()

            # Resize heatmap to match input image size
            heatmap_resized = cv2.resize(
                heatmap_np, (self.input_shape[1], self.input_shape[0])
            )

            # Create visualization with original image + heatmap overlay
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            from datetime import datetime

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Original image
            axes[0].imshow(img_display)
            axes[0].set_title("Original Medical Image", fontsize=14, fontweight="bold")
            axes[0].axis("off")

            # Heatmap only
            colors = ["black", "purple", "blue", "cyan", "yellow", "orange", "red"]
            cmap = LinearSegmentedColormap.from_list("medical_gradcam", colors, N=256)

            im1 = axes[1].imshow(heatmap_resized, cmap=cmap, alpha=1.0)
            axes[1].set_title("Grad-CAM Activation Map", fontsize=14, fontweight="bold")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # Overlay
            axes[2].imshow(img_display)
            # Enhance heatmap for overlay
            enhanced_heatmap = np.power(heatmap_resized, 0.4)  # Enhance contrast
            enhanced_heatmap = np.clip(enhanced_heatmap * 2.0, 0, 1)

            overlay = axes[2].imshow(enhanced_heatmap, cmap=cmap, alpha=0.6)
            axes[2].set_title(
                "Medical Image + Grad-CAM Overlay", fontsize=14, fontweight="bold"
            )
            axes[2].axis("off")

            plt.tight_layout()

            # Save the visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"{patient_id}_medical_gradcam_{timestamp}.png"
            plot_path = os.path.join("results", "gradcam_plots", plot_filename)

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close()

            # Generate prediction and explanation
            prediction = float(self.model.predict(input_tensor, verbose=0)[0][0])
            predicted_class = "Malignant" if prediction > 0.5 else "Benign"
            confidence = prediction if prediction > 0.5 else (1 - prediction)

            # Create medical explanation
            explanation = f"Medical Image Analysis for {patient_id}:\n\n"
            explanation += f"• Predicted Diagnosis: {predicted_class}\n"
            explanation += f"• Model Confidence: {confidence:.1%}\n"
            explanation += f"• Risk Score: {prediction:.3f}\n\n"
            explanation += "Grad-CAM Analysis Results:\n"
            explanation += (
                "• Red/Orange regions: Areas most suspicious for malignancy\n"
            )
            explanation += "• Yellow/Cyan regions: Moderately concerning areas\n"
            explanation += "• Blue/Purple regions: Areas of mild concern\n"
            explanation += "• Dark regions: Areas not contributing to the diagnosis\n\n"
            explanation += (
                "Clinical Note: This visualization shows which regions of the "
            )
            explanation += (
                "medical image influenced the AI model's diagnostic decision. "
            )
            explanation += (
                "Areas highlighted in warm colors (red/orange/yellow) represent "
            )
            explanation += "regions the model identified as having characteristics "
            explanation += "associated with the predicted diagnosis."

            return f"/results/gradcam_plots/{plot_filename}", explanation

        except Exception as e:
            print(f"Error generating medical Grad-CAM: {str(e)}")
            import traceback

            traceback.print_exc()
            return None, f"Grad-CAM generation failed: {str(e)}"

    def predict_medical_image(self, image_path, patient_id=None):
        """Make prediction on medical image with Grad-CAM visualization"""
        try:
            if not os.path.exists(self.model_path):
                raise Exception(
                    "Medical model not found. Please train the model first."
                )

            # Load model if not already loaded
            if self.model is None:
                self.model = keras.models.load_model(self.model_path)

            # Generate patient ID if not provided
            if patient_id is None:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                patient_id = f"medical_patient_{timestamp}"

            # Load and preprocess image
            if image_path.lower().endswith(".dcm"):
                img_array = self.load_dicom_image(image_path)
            else:
                img_array = self.load_standard_image(image_path)

            if img_array is None:
                raise Exception("Failed to load or preprocess the medical image")

            # Make prediction
            input_tensor = np.expand_dims(img_array, axis=0)
            prediction_proba = self.model.predict(input_tensor, verbose=0)[0][0]
            predicted_class = "Malignant" if prediction_proba > 0.5 else "Benign"

            # Generate Grad-CAM visualization
            gradcam_url, explanation = self.generate_medical_gradcam(
                image_path, patient_id
            )

            result = {
                "patientId": patient_id,
                "predictedClass": predicted_class,
                "confidence": float(prediction_proba),
                "gradCamUrl": gradcam_url,
                "explanationText": explanation or "Medical Grad-CAM analysis completed",
                "status": "success",
            }

            return result

        except Exception as e:
            return {
                "error": str(e),
                "patientId": patient_id or "unknown",
                "predictedClass": "Unknown",
                "confidence": 0.0,
                "gradCamUrl": None,
                "explanationText": f"Medical prediction failed: {str(e)}",
                "status": "error",
            }

    def load_standard_image(self, image_path):
        """Load and preprocess standard image formats"""
        try:
            img = Image.open(image_path)

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to target shape
            img = img.resize((self.input_shape[1], self.input_shape[0]))

            # Convert to numpy array and normalize
            img_array = np.array(img).astype(np.float32) / 255.0

            return img_array

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def train_with_medical_data(
        self, data_path, dataset_type="histopathology", epochs=50
    ):
        """Train model with actual medical dataset from Kaggle or other sources"""
        print(f"Loading {dataset_type} dataset from {data_path}...")

        # Load appropriate dataset
        if dataset_type.lower() == "ddsm":
            X, y = self.load_ddsm_dataset(data_path)
        elif dataset_type.lower() == "breakhis":
            X, y = self.load_breakhis_dataset(data_path)
        elif dataset_type.lower() == "histopathology":
            X, y = self.load_kaggle_histopathology(data_path)
        elif dataset_type.lower() == "ultrasound":
            # Similar to histopathology but may have different structure
            X, y = self.load_kaggle_histopathology(data_path)
        elif dataset_type.lower() == "busi":
            X, y = self.load_busi_dataset(data_path)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        if len(X) == 0:
            raise ValueError("No images loaded. Check data path and structure.")

        print(f"Loaded {len(X)} images")
        print(f"Image shape: {X[0].shape}")
        print(f"Benign samples: {np.sum(y == 0)}")
        print(f"Malignant samples: {np.sum(y == 1)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create model
        self.model = self.create_medical_model()
        print("Model architecture:")
        self.model.summary()

        # Data augmentation for medical images (more conservative)
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,  # Reduced rotation for medical images
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,  # Only if medically appropriate
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],  # Slight brightness variation
            fill_mode="nearest",
        )

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-7, monitor="val_loss"
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_path, save_best_only=True, monitor="val_loss"
            ),
        ]

        # Train model
        print(f"Training model for {epochs} epochs...")
        history = self.model.fit(
            datagen.flow(
                X_train, y_train, batch_size=16
            ),  # Smaller batch for medical images
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=len(X_train) // 16,
        )

        # Evaluate
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba.flatten())

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "f1Score": float(f1),
            "rocAuc": float(roc_auc),
            "confusionMatrix": {
                "trueNegatives": int(tn),
                "falsePositives": int(fp),
                "falseNegatives": int(fn),
                "truePositives": int(tp),
            },
        }

        # Save metrics
        import json

        metrics_file = self.metrics_path.replace(".json", "_medical.json")
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")

        return metrics


# Example usage and CLI interface
def main():
    import sys
    import json

    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    "error": "Usage: python medical_image_model.py <command> [args]",
                    "commands": [
                        "train <data_path> <dataset_type> [epochs]",
                        "predict <image_file> [patient_id]",
                        "metrics",
                    ],
                    "supported_datasets": [
                        "ddsm",
                        "breakhis",
                        "histopathology",
                        "ultrasound",
                        "busi",
                    ],
                }
            )
        )
        sys.exit(1)

    command = sys.argv[1].lower()
    model = MedicalBreastCancerImageModel()

    try:
        if command == "train":
            if len(sys.argv) < 4:
                print(
                    json.dumps(
                        {
                            "error": "Usage: python medical_image_model.py train <data_path> <dataset_type> [epochs]",
                            "supported_datasets": [
                                "ddsm",
                                "breakhis",
                                "histopathology",
                                "ultrasound",
                                "busi",
                            ],
                        }
                    )
                )
                sys.exit(1)

            data_path = sys.argv[2]
            dataset_type = sys.argv[3]
            epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 50

            if not os.path.exists(data_path):
                print(json.dumps({"error": f"Data path not found: {data_path}"}))
                sys.exit(1)

            print("Training medical image model...")
            metrics = model.train_with_medical_data(data_path, dataset_type, epochs)
            print(json.dumps({"status": "Training completed", "metrics": metrics}))

        elif command == "predict":
            if len(sys.argv) < 3:
                print(
                    json.dumps(
                        {
                            "error": "Usage: python medical_image_model.py predict <image_file> [patient_id]"
                        }
                    )
                )
                sys.exit(1)

            image_file = sys.argv[2]
            patient_id = sys.argv[3] if len(sys.argv) > 3 else None

            if not os.path.exists(image_file):
                print(json.dumps({"error": f"Image file not found: {image_file}"}))
                sys.exit(1)

            result = model.predict_medical_image(image_file, patient_id)
            print(json.dumps(result))

        elif command == "metrics":
            # Load existing metrics
            metrics_file = model.metrics_path.replace(".json", "_medical.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                print(json.dumps({"status": "success", "metrics": metrics}))
            else:
                print(
                    json.dumps(
                        {
                            "error": "No metrics found. Please train the model first.",
                            "status": "not_found",
                        }
                    )
                )

        else:
            print(
                json.dumps(
                    {
                        "error": f"Unknown command: {command}",
                        "available_commands": ["train", "predict", "metrics"],
                    }
                )
            )
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
