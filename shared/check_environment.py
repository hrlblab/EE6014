"""ECE 6041: check required local Python dependencies (no GPU required)."""
import importlib
import sys

print('Python:', sys.version.split()[0])
failed = []
for module in ['torch', 'numpy', 'pandas', 'sklearn', 'matplotlib', 'onnx', 'onnxruntime']:
    try:
        obj = importlib.import_module(module)
        print(f'PASS {module} {getattr(obj, "__version__", "")}')
    except Exception as exc:
        print(f'FAIL {module}: {exc}')
        failed.append(module)
if not failed:
    import torch
    x = torch.randn(1, 42)
    model = torch.nn.Linear(42, 4).eval()
    with torch.no_grad():
        result = model(x)
    print('PASS CPU inference shape:', tuple(result.shape))
else:
    print('Missing packages:', ', '.join(failed))
    sys.exit(1)
