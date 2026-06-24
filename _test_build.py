import sys
sys.path.insert(0, '.')
from models.model_factory import build_model
import warnings
warnings.filterwarnings('ignore')

results = []
for name in [
    'LSDT-Base', 'LSDT-Swin-T', 'LSDT-Large',
    'LSDT-Omnirad', 'LSDT-BiomedCLIP', 'LSDT-ResNet50',
    'VITClassifier', 'SWINClassifier', 'VITLargeClassifier', 'ResNetClassifier',
]:
    try:
        m = build_model(name)
        results.append(f'  ✅ {name}')
    except Exception as e:
        results.append(f'  ❌ {name}: {e}')

for r in results:
    print(r)

n_pass = sum(1 for r in results if '✅' in r)
n_fail = sum(1 for r in results if '❌' in r)
print(f'\nPass: {n_pass}, Fail: {n_fail}')
sys.exit(0 if n_fail == 0 else 1)