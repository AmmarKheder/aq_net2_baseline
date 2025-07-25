#!/usr/bin/env python3
import os
import sys

print("🧪 Testing PyTorch import...")
print(f"Python: {sys.version}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

try:
    import torch
    print(f"✅ PyTorch imported successfully: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA devices: {torch.cuda.device_count()}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

print("\n🧪 Testing basic PyTorch operations...")
try:
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.mm(x, y)
    print(f"✅ Matrix multiplication works: {z.shape}")
except Exception as e:
    print(f"❌ PyTorch operations failed: {e}")
    sys.exit(1)

print("\n✅ PyTorch is working correctly!")
