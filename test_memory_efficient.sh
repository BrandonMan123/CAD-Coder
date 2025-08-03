#!/bin/bash

# Memory-efficient backward compatibility test using existing infrastructure
echo "🧪 Testing backward compatibility with memory optimizations..."

# Create a tiny test dataset (just 1 sample)
mkdir -p ./test_compatibility
cat > ./test_compatibility/tiny_test.jsonl << 'EOF'
{"question_id": 1, "image": "sample.jpg", "text": "What do you see?"}
EOF

# Create a dummy image
convert -size 224x224 xc:white ./inference/test100_images/sample.jpg 2>/dev/null || echo "Note: Install imagemagick for dummy image creation"

# Test with 4-bit quantization and minimal parameters
echo "Running compatibility test with 4-bit quantization..."

python -c "
import torch
import sys
sys.path.insert(0, '.')

try:
    from llava.model.builder import load_pretrained_model
    print('✅ Import successful')
    
    # Test loading with memory optimizations
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path='CADCODER/CAD-Coder',
        model_base=None,
        model_name='CAD-Coder', 
        load_4bit=True,
        device_map='auto'
    )
    print('✅ Model loaded successfully with 4-bit quantization!')
    print(f'🔧 Vision tower exists: {model.get_vision_tower() is not None}')
    print(f'🆕 Pointcloud method exists: {hasattr(model, \"encode_pointclouds\")}')
    
except Exception as e:
    print(f'❌ Loading failed: {e}')
    sys.exit(1)
"

echo "✅ Backward compatibility test complete!"