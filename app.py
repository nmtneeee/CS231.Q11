"""
Gradio Web UI for Animal Image Classification
"""
import gradio as gr
import numpy as np
from pathlib import Path
from src.predict import AnimalClassifier


# Initialize classifier (load once at startup)
MODULES_PATH = Path(__file__).parent / 'modules'
classifier_svm = None
classifier_rf = None


def load_classifiers():
    """Load both classifiers."""
    global classifier_svm, classifier_rf
    if classifier_svm is None:
        print("Loading SVM classifier...")
        classifier_svm = AnimalClassifier(str(MODULES_PATH), model_type='svm')
    if classifier_rf is None:
        print("Loading Random Forest classifier...")
        classifier_rf = AnimalClassifier(str(MODULES_PATH), model_type='random_forest')


def predict_animal(image, model_choice):
    """
    Predict animal class from uploaded image.
    
    Args:
        image: numpy array from Gradio Image component
        model_choice: 'SVM' or 'Random Forest'
        
    Returns:
        Dictionary of class probabilities for Gradio Label component
    """
    if image is None:
        return None
    
    # Ensure classifiers are loaded
    load_classifiers()
    
    # Select classifier
    classifier = classifier_svm if model_choice == "SVM" else classifier_rf
    
    # Save temp image (classifier expects file path)
    import tempfile
    import cv2
    
    # Convert RGB to BGR for cv2
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
        cv2.imwrite(temp_path, image_bgr)
    
    try:
        # Get prediction
        result = classifier.predict(temp_path)
        
        # Return probabilities as dict for Label component
        if 'probabilities' in result:
            return result['probabilities']
        else:
            # If no probabilities, return just the predicted class with 100%
            return {result['predicted_class']: 1.0}
            
    finally:
        # Cleanup temp file
        import os
        os.unlink(temp_path)


def predict_both_models(image):
    """
    Predict using both models and compare results.
    
    Returns:
        Tuple of (svm_result, rf_result, comparison_text)
    """
    if image is None:
        return None, None, "Please upload an image"
    
    load_classifiers()
    
    import tempfile
    import cv2
    import os
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
        cv2.imwrite(temp_path, image_bgr)
    
    try:
        result_svm = classifier_svm.predict(temp_path)
        result_rf = classifier_rf.predict(temp_path)
        
        svm_probs = result_svm.get('probabilities', {result_svm['predicted_class']: 1.0})
        rf_probs = result_rf.get('probabilities', {result_rf['predicted_class']: 1.0})
        
        # Comparison text
        svm_pred = result_svm['predicted_class'].upper()
        rf_pred = result_rf['predicted_class'].upper()
        svm_conf = result_svm.get('confidence', 1.0) * 100
        rf_conf = result_rf.get('confidence', 1.0) * 100
        
        if svm_pred == rf_pred:
            comparison = f"✅ Both models agree: **{svm_pred}**\n\n"
            comparison += f"- SVM confidence: {svm_conf:.1f}%\n"
            comparison += f"- Random Forest confidence: {rf_conf:.1f}%"
        else:
            comparison = f"⚠️ Models disagree:\n\n"
            comparison += f"- SVM predicts: **{svm_pred}** ({svm_conf:.1f}%)\n"
            comparison += f"- Random Forest predicts: **{rf_pred}** ({rf_conf:.1f}%)"
        
        return svm_probs, rf_probs, comparison
        
    finally:
        os.unlink(temp_path)


# Create Gradio Interface
with gr.Blocks(title="🐾 Animal Classifier") as demo:
    gr.Markdown(
        """
        # 🐾 Animal Image Classifier
        
        Upload an image to classify the animal. This classifier can identify:
        **butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel**
        
        *Powered by VGG16 features + SVM/Random Forest classifiers*
        """
    )
    
    with gr.Tabs():
        # Tab 1: Single Model Prediction
        with gr.Tab("🎯 Single Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        height=300
                    )
                    model_choice = gr.Radio(
                        choices=["SVM", "Random Forest"],
                        value="SVM",
                        label="Select Model"
                    )
                    predict_btn = gr.Button("🔍 Predict", variant="primary")
                    
                with gr.Column(scale=1):
                    output_label = gr.Label(
                        label="Prediction Results",
                        num_top_classes=5
                    )
            
            predict_btn.click(
                fn=predict_animal,
                inputs=[image_input, model_choice],
                outputs=output_label
            )
        
        # Tab 2: Compare Both Models
        with gr.Tab("⚖️ Compare Models"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input_compare = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        height=300
                    )
                    compare_btn = gr.Button("🔍 Compare Both Models", variant="primary")
                    
                with gr.Column(scale=2):
                    with gr.Row():
                        svm_output = gr.Label(label="SVM Prediction", num_top_classes=5)
                        rf_output = gr.Label(label="Random Forest Prediction", num_top_classes=5)
                    comparison_text = gr.Markdown(label="Comparison")
            
            compare_btn.click(
                fn=predict_both_models,
                inputs=image_input_compare,
                outputs=[svm_output, rf_output, comparison_text]
            )
    
    gr.Markdown(
        """
        ---
        ### How it works:
        1. **VGG16 Feature Extraction**: Extract 4096-dimensional features from fc2 layer
        2. **StandardScaler**: Normalize features
        3. **PCA**: Reduce to 500 components
        4. **Classification**: SVM or Random Forest predicts the animal class
        """
    )


if __name__ == "__main__":
    print("Loading models... (this may take a moment)")
    load_classifiers()
    print("\nStarting Gradio server...")
    demo.launch()