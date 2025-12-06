"""
Animal Image Classification Demo
Predicts animal class using VGG16 features + SVM/Random Forest

Pipeline:
1. Load image and preprocess (resize to 224x224)
2. Extract VGG16 fc2 features (4096-dim)
3. Scale features using StandardScaler
4. Reduce dimensionality using PCA (500 components)
5. Predict using SVM or Random Forest classifier
6. Decode label and display result
"""

import numpy as np
import cv2
import pickle
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess


class AnimalClassifier:
    """Animal classifier using VGG16 features and SVM/Random Forest."""
    
    # Class labels (translation from Italian)
    CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
               'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    
    def __init__(self, modules_path: str, model_type: str = 'svm'):
        """
        Initialize the classifier.
        
        Args:
            modules_path: Path to the modules folder containing saved models
            model_type: 'svm' or 'random_forest'
        """
        self.modules_path = Path(modules_path)
        self.model_type = model_type
        self.img_size = (224, 224)
        
        # Load components
        self._load_vgg16_model()
        self._load_scaler()
        self._load_pca()
        self._load_classifier()
        self._load_label_encoder()
        
        print(f"\n✓ Classifier initialized successfully!")
        print(f"  Model type: {model_type.upper()}")
        
    def _load_vgg16_model(self):
        """Load VGG16 model for feature extraction."""
        print("Loading VGG16 model...")
        base_model = VGG16(weights='imagenet', include_top=True)
        self.feature_extractor = tf.keras.Model(
            inputs=base_model.input, 
            outputs=base_model.get_layer('fc2').output
        )
        print("  ✓ VGG16 model loaded (fc2 layer output)")
        
    def _load_scaler(self):
        """Load StandardScaler."""
        scaler_path = self.modules_path / 'vgg16_scaler.pkl'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"  ✓ Scaler loaded from {scaler_path.name}")
        
    def _load_pca(self):
        """Load PCA transformer."""
        pca_path = self.modules_path / 'vgg16_pca.pkl'
        with open(pca_path, 'rb') as f:
            self.pca = pickle.load(f)
        print(f"  ✓ PCA loaded from {pca_path.name} ({self.pca.n_components_} components)")
        
    def _load_classifier(self):
        """Load the classifier model."""
        if self.model_type == 'svm':
            model_path = self.modules_path / 'vgg16_svm_model.pkl'
        else:
            model_path = self.modules_path / 'vgg16_random_forest_model.pkl'
            
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        print(f"  ✓ {self.model_type.upper()} classifier loaded from {model_path.name}")
        
        # Display best params if GridSearchCV object
        if hasattr(self.classifier, 'best_params_'):
            print(f"    Best params: {self.classifier.best_params_}")
            
    def _load_label_encoder(self):
        """Load label encoder."""
        encoder_path = self.modules_path / 'vgg16_label_encoder.pkl'
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"  ✓ Label encoder loaded from {encoder_path.name}")
        print(f"    Classes: {list(self.label_encoder.classes_)}")
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array ready for VGG16
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Resize to VGG16 input size
        img = cv2.resize(img, self.img_size)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension and preprocess for VGG16
        img_preprocessed = vgg_preprocess(np.expand_dims(img_rgb, axis=0))
        
        return img_preprocessed
    
    def extract_features(self, image_preprocessed: np.ndarray) -> np.ndarray:
        """
        Extract VGG16 fc2 features from preprocessed image.
        
        Args:
            image_preprocessed: Preprocessed image array
            
        Returns:
            Feature vector (4096-dim)
        """
        features = self.feature_extractor.predict(image_preprocessed, verbose=0)
        return features
    
    def predict(self, image_path: str) -> dict:
        """
        Predict the animal class for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        # Step 1: Preprocess image
        img_preprocessed = self.preprocess_image(image_path)
        
        # Step 2: Extract VGG16 features
        features = self.extract_features(img_preprocessed)
        
        # Step 3: Scale features
        features_scaled = self.scaler.transform(features)
        
        # Step 4: Apply PCA
        features_pca = self.pca.transform(features_scaled)
        
        # Step 5: Predict
        prediction_encoded = self.classifier.predict(features_pca)
        
        # Step 6: Decode label
        prediction_label = self.label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Get prediction probabilities if available
        proba = None
        if hasattr(self.classifier, 'predict_proba'):
            proba = self.classifier.predict_proba(features_pca)[0]
        elif hasattr(self.classifier, 'best_estimator_') and hasattr(self.classifier.best_estimator_, 'predict_proba'):
            proba = self.classifier.best_estimator_.predict_proba(features_pca)[0]
            
        result = {
            'image_path': image_path,
            'predicted_class': prediction_label,
            'predicted_encoded': int(prediction_encoded[0]),
            'feature_shape': features.shape,
            'pca_shape': features_pca.shape
        }
        
        if proba is not None:
            result['probabilities'] = {
                self.label_encoder.inverse_transform([i])[0]: float(p) 
                for i, p in enumerate(proba)
            }
            result['confidence'] = float(max(proba))
            
        return result
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict animal classes for multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': path,
                    'error': str(e)
                })
        return results


def display_prediction(result: dict):
    """Display prediction result in a formatted way."""
    print("\n" + "="*50)
    print(f"Image: {result['image_path']}")
    print("="*50)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
        
    print(f"🎯 Predicted Class: {result['predicted_class'].upper()}")
    
    if 'confidence' in result:
        print(f"📊 Confidence: {result['confidence']*100:.2f}%")
        
    if 'probabilities' in result:
        print("\n📈 Top 5 Probabilities:")
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        for cls, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            print(f"   {cls:12s} {prob*100:5.2f}% {bar}")
    
    print(f"\n📐 Feature dimensions: {result['feature_shape'][1]} → {result['pca_shape'][1]} (after PCA)")


def main():
    """Demo main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Animal Image Classification Demo')
    parser.add_argument('image_path', type=str, nargs='?', 
                        help='Path to the image to classify')
    parser.add_argument('--model', type=str, default='svm', 
                        choices=['svm', 'random_forest'],
                        help='Model type to use (default: svm)')
    parser.add_argument('--modules', type=str, default='../modules',
                        help='Path to modules folder (default: ../modules)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    modules_path = Path(args.modules)
    if not modules_path.is_absolute():
        # Make relative to script location
        modules_path = Path(__file__).parent.parent / 'modules'
    
    classifier = AnimalClassifier(str(modules_path), model_type=args.model)
    
    if args.image_path:
        # Predict single image
        result = classifier.predict(args.image_path)
        display_prediction(result)
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("🐾 Animal Classification Demo - Interactive Mode")
        print("="*50)
        print("Enter image path to classify (or 'quit' to exit):")
        
        while True:
            try:
                image_path = input("\n> Image path: ").strip()
                if image_path.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                if not image_path:
                    continue
                    
                result = classifier.predict(image_path)
                display_prediction(result)
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
