import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "models/mold_model_final.keras"

def inspect_model():
    """Inspect the mold detection model architecture and details"""
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        return
    
    print("🔍 Loading model for inspection...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    print("\n" + "="*60)
    print("📊 MODEL SUMMARY")
    print("="*60)
    
    # Basic info
    print(f"Model type: {type(model).__name__}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    # Input/Output shapes
    print(f"\nInput shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    # Model architecture
    print("\n" + "="*60)
    print("🏗️  MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    
    # Layer details
    print("\n" + "="*60)
    print("📋 LAYER BREAKDOWN")
    print("="*60)
    
    for i, layer in enumerate(model.layers):
        try:
            output_shape = layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'
        except:
            output_shape = 'N/A'
        print(f"{i+1:2d}. {layer.name:<30} | {layer.__class__.__name__:<20} | Output: {output_shape}")
    
    # Test prediction to understand output
    print("\n" + "="*60)
    print("🧪 TEST PREDICTION")
    print("="*60)
    
    # Create dummy input
    dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    prediction = model.predict(dummy_input, verbose=0)
    
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Raw prediction: {prediction}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    
    # Interpret output
    if prediction.shape[-1] == 1:
        print("\n📈 BINARY CLASSIFICATION MODEL")
        print("- Single output neuron (sigmoid activation)")
        print("- Output > 0.5 = MOLD")
        print("- Output < 0.5 = CLEAN")
    elif prediction.shape[-1] == 2:
        print("\n📈 CATEGORICAL CLASSIFICATION MODEL")
        print("- Two output neurons (softmax activation)")
        print("- Class 0: CLEAN, Class 1: MOLD")
    
    # Check if it's based on a pre-trained model
    print("\n" + "="*60)
    print("🔍 MODEL ANALYSIS")
    print("="*60)
    
    layer_names = [layer.name for layer in model.layers]
    
    if any('efficientnet' in name.lower() for name in layer_names):
        print("✅ EfficientNet-based model detected")
    elif any('mobilenet' in name.lower() for name in layer_names):
        print("✅ MobileNet-based model detected")
    elif any('resnet' in name.lower() for name in layer_names):
        print("✅ ResNet-based model detected")
    else:
        print("🔧 Custom architecture or unknown base model")
    
    # Check for transfer learning
    frozen_layers = sum(1 for layer in model.layers if not layer.trainable)
    trainable_layers = sum(1 for layer in model.layers if layer.trainable)
    
    print(f"Frozen layers: {frozen_layers}")
    print(f"Trainable layers: {trainable_layers}")
    
    if frozen_layers > 0:
        print("✅ Transfer learning detected (some layers frozen)")
    else:
        print("🔧 All layers trainable (trained from scratch or fine-tuned)")

if __name__ == "__main__":
    inspect_model()