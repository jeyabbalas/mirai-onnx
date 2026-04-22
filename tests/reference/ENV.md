# Phase 0 environment

The Mirai upstream pins `python_requires = >=3.8, <3.9` and `torch==1.9.0`. There is no Apple-Silicon arm64 wheel for that combination, so we use **miniforge with an `osx-64` subdir** (Rosetta-translated x86_64 Python 3.8) and pip-install the official PyTorch x86_64 wheel.

## One-time host-level setup

```bash
# Rosetta 2 (run once per machine)
softwareupdate --install-rosetta --agree-to-license

# Homebrew packages
brew install --cask miniforge
brew install dcmtk        # required for the dcmtk DICOM-decoding path
```

## Conda env

```bash
# Source conda's hook in the current shell. (`conda init zsh && exec zsh` works
# too, but `exec zsh` replaces the current shell so don't run it from a script.)
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh

# Create the env (Rosetta x86_64; pin the subdir for future installs)
CONDA_SUBDIR=osx-64 conda create -n mirai-py38 python=3.8 -y
conda activate mirai-py38
conda config --env --set subdir osx-64
```

## Mirai install

```bash
pip install -e /Users/jeya/Documents/projects/mirai-onnx/external/Mirai[test]
```

### Known-bad pylibjpeg wheels

Mirai depends on `pylibjpeg[all]>=2.0.0`. The pip wheels for `pylibjpeg-libjpeg==2.2.0` and `pylibjpeg-openjpeg==2.3.0` are **mislabeled**: the wheel filenames advertise `macosx_*_x86_64`, but the `.so` inside is arm64. They cannot be loaded under our x86_64 Python and pydicom imports fail at startup as a result.

**Workaround used in this env**: uninstall all three pylibjpeg packages. The Mirai demo DICOMs are uncompressed, so pydicom can read them with its built-in handlers.

```bash
pip uninstall -y pylibjpeg-openjpeg pylibjpeg-libjpeg pylibjpeg-rle pylibjpeg
```

If a future DICOM uses a JPEG-compressed transfer syntax (e.g. JPEG 2000), pydicom will need a working decoder. Options that may work on x86_64 Python 3.8:

- `pip install python-gdcm` (often packaged as a universal wheel),
- `pip install pylibjpeg-libjpeg==1.3.0` (older wheels were correctly tagged — verify with `file` after install),
- decode with `dcmtk` upstream and feed the resulting PNG to Mirai.

Track the upstream issue at https://github.com/pydicom/pylibjpeg-libjpeg/issues if you want to wait for a real fix.

### torch.backends.mps reference

`torch==1.9.0` predates `torch.backends.mps`. Code paths in `onconet/models/mirai_full.py:313` reference it, but they're only reached when `args.cuda=True`. On a CUDA-less Mac, `MiraiModel.load_model` flips `args.cuda` to False (`mirai_full.py:102`) and the mps branch is never executed.

## Verification

```bash
python -c "
import platform, sys, torch, pydicom, numpy, torchvision, PIL
import onconet
from onconet.models.mirai_full import MiraiModel
from onconet.utils.dicom import is_dcmtk_installed
print('arch:', platform.machine())               # x86_64
print('python:', sys.version.split()[0])         # 3.8.20
print('torch:', torch.__version__)                # 1.9.0
print('torchvision:', torchvision.__version__)    # 0.10.0
print('numpy:', numpy.__version__)                # 1.24.4
print('pydicom:', pydicom.__version__)            # 2.3.0
print('Pillow:', PIL.__version__)                 # 9.0.0
print('onconet:', onconet.__version__)            # 0.14.1
print('dcmtk:', is_dcmtk_installed())             # True
"
mirai-predict --dry-run                           # downloads ~/.mirai/snapshots/...
```

## Demo data

`run_predict_demo.sh` uses `wget`, which isn't installed by default on macOS. Use curl instead:

```bash
cd /Users/jeya/Documents/projects/mirai-onnx
curl -sLO https://github.com/reginabarzilaygroup/Mirai/releases/latest/download/mirai_demo_data.zip
unzip -o mirai_demo_data.zip -d mirai_demo_data
rm mirai_demo_data.zip
```

## Baseline confirmation

```bash
mirai-predict --output-path demo_prediction.json --use-pydicom \
    mirai_demo_data/ccl1.dcm mirai_demo_data/ccr1.dcm \
    mirai_demo_data/mlol2.dcm mirai_demo_data/mlor2.dcm
```

Captured upstream output on this machine, 2026-04-22:

```
pydicom: {Year 1: 0.0314, Year 2: 0.0505, Year 3: 0.0711, Year 4: 0.0935, Year 5: 0.1052}
dcmtk:   {Year 1: 0.0298, Year 2: 0.0483, Year 3: 0.0684, Year 4: 0.09,   Year 5: 0.1016}
```

The capture script must reproduce these exactly (to the printed precision). Drift caught by `test_baseline.py`.
