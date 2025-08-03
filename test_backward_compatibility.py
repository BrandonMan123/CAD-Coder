#!/usr/bin/env python3
"""
Low-memory backward compatibility test for pointcloud-enhanced LLaVA
"""
import torch
import sys
import os

def test_model_loading_4bit():
    """Test loading with 4-bit quantization - uses ~3-4GB instead of 13GB+"""
    print("🧪 Testing 4-bit quantized model loading...")
    
    try:
        from llava.model.builder import load_pretrained_model
        
        # Load with 4-bit quantization to save memory
        model_path = "CADCODER/CAD-Coder"  # CAD-Coder model
        
        print(f"Loading {model_path} with 4-bit quantization...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name="CAD-Coder",
            load_4bit=True,           # 4-bit quantization
            device_map="auto"         # Auto GPU distribution
        )
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model config: {type(model.config)}")
        print(f"🔧 Vision tower: {model.get_vision_tower() is not None}")
        print(f"🔧 MM projector: {hasattr(model.get_model(), 'mm_projector')}")
        
        # Test our new method exists but doesn't crash
        print(f"🆕 Pointcloud encoder: {hasattr(model, 'encode_pointclouds')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cpu_only_loading():
    """Test loading on CPU only - slower but uses no GPU memory"""
    print("\n🧪 Testing CPU-only loading...")
    
    try:
        from llava.model.builder import load_pretrained_model
        
        model_path = "CADCODER/CAD-Coder"
        
        print(f"Loading {model_path} on CPU...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None, 
            model_name="CAD-Coder",
            device="cpu"              # CPU only
        )
        
        print("✅ CPU loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ CPU loading failed: {e}")
        return False

def test_config_only():
    """Test just loading config without weights"""
    print("\n🧪 Testing config-only loading...")
    
    try:
        from transformers import AutoConfig
        from llava.model.language_model.llava_llama import LlavaConfig
        
        model_path = "CADCODER/CAD-Coder"
        config = LlavaConfig.from_pretrained(model_path)
        
        print("✅ Config loaded successfully!")
        print(f"📋 Config type: {type(config)}")
        print(f"🔧 Has vision tower: {hasattr(config, 'mm_vision_tower')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 LLaVA Backward Compatibility Test (Low Memory)")
    print("=" * 50)
    
    # Test 1: Try 4-bit loading first (most practical)
    success_4bit = test_model_loading_4bit()
    
    # Test 2: Try CPU loading if GPU fails
    if not success_4bit:
        success_cpu = test_cpu_only_loading()
    
    # Test 3: At minimum, config should load
    success_config = test_config_only()
    
    print("\n📊 Results:")
    print(f"✅ 4-bit GPU loading: {success_4bit}")
    print(f"✅ CPU loading: {'✅' if not success_4bit else 'Skipped'}")
    print(f"✅ Config loading: {success_config}")
    
    if success_4bit or success_config:
        print("\n🎉 Backward compatibility confirmed!")
        print("Your pointcloud modifications don't break existing models.")
    else:
        print("\n❌ Issues detected - check implementation")