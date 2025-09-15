# Vehicle Detection in CCTV with Global-Guided Self-Attention and Convolution

This repository contains the implementation code for the paper "Vehicle Detection in CCTV with Global-Guided Self-Attention and Convolution". The proposed GGMix block enhances vehicle detection performance in CCTV scenarios by combining global guidance mechanisms with self-attention and convolution operations.

## Overview

This project extends the MMYOLO framework to implement a novel architecture for vehicle detection in CCTV footage. The core contribution is the **GGMix block**, which integrates:

- **Global Guidance Module**: Provides global context awareness for better feature representation
- **Multi-Head Self-Attention**: Captures long-range dependencies in feature maps
- **Depthwise Separable Convolution**: Efficient spatial feature processing
- **Flow Warping**: Dynamic feature alignment based on learned offsets

## Architecture

The GGMix block is implemented in `mmyolo/models/backbones/base_backbone.py` and consists of several key components:

### Key Components
- **Global_Guidance**: Generates spatial offsets, attention masks, and channel/spatial attention weights
- **MultiHeadAttention**: Implements multi-head self-attention mechanism
- **DepthwiseSeparableConv**: Efficient convolution with reduced parameters
- **Flow Warping**: Applies learned geometric transformations to features

### Model Structure
```
Input Image → Backbone (YOLOv8CSPDarknet) → GGMix Blocks → Neck (YOLOv8PAFPN) → Head (YOLOv8Head) → Output
```

## Requirements

### Environment Setup
This project is based on MMYOLO. Please refer to the [MMYOLO installation guide](https://github.com/open-mmlab/mmyolo) for detailed setup instructions.

### Recommended Environment
- **Python**: 3.10
- **CUDA**: 11.8
- **PyTorch**: 2.0.0
- **Torchvision**: 0.15.1

### Dependencies
Install the required dependencies:
```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

# Install MMYOLO and dependencies
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"

# Additional dependencies for GGMix
pip install einops
pip install fairscale
pip install timm
```

## Quick Start

### Installation
1. Clone this repository:
```bash
git clone https://github.com/Flulike/vdc_yolo
cd vdv_yolo
```

2. Install the package:
```bash
pip install -r requirements.txt
mim install -v -e .
```

### Training
Train the model using the provided configuration:
```bash
python tools/train.py own/yolov8s_car.py

# Train with specific GPU
CUDA_VISIBLE_DEVICES=0 python tools/train.py own/yolov8s_car.py 
```

### Test
```bash
python tools/test.py own/yolov8s_car.py checkpoint --show-dir show_results
```

## Project Structure

```
vdv_yolo/
├── mmyolo/
│   ├── models/
│   │   └── backbones/
│   │       └── base_backbone.py      # GGMix implementation
│   └── ...
├── own/
│   └── yolov8s_car.py               # Training configuration
├── tools/
│   ├── train.py                     # Training script
│   ├── test.py                      # Testing script
│   └── ...
├── data/                            # Dataset directory
├── requirements.txt
└── README.md
```

## Configuration

The main configuration file `own/yolov8s_car.py` contains:
- Model architecture settings (YOLOv8 with GGMix blocks)
- Training hyperparameters
- Data loading and augmentation settings
- Loss function configuration

### Key Configuration Parameters
```python
model = dict(
    type='YOLODetector',
    backbone=dict(
        type='YOLOv8CSPDarknet',
        # GGMix blocks are integrated in the backbone
    ),
    neck=dict(type='YOLOv8PAFPN'),
    bbox_head=dict(type='YOLOv8Head')
)
```

## Performance

The GGMix block enhances vehicle detection performance by:
- Improving global context understanding
- Reducing false positives in complex CCTV scenarios
- Maintaining computational efficiency through depthwise separable convolutions

## Key Features

### GGMix Block Advantages
1. **Global Guidance**: Incorporates global image context for better feature representation
2. **Adaptive Attention**: Uses learnable masks to focus on relevant regions
3. **Flow Warping**: Applies geometric transformations for feature alignment
4. **Efficient Design**: Maintains real-time performance while improving accuracy

### Technical Highlights
- Multi-scale feature processing with window-based attention
- Deformable convolution through learned offsets
- Channel and spatial attention mechanisms
- Residual connections for stable training

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{guo2025vehicle,
    title={Vehicle Detection in CCTV with Global-Guided Self-Attention and Convolution},
    author={Yupei Guo, Yota Yamamoto1, Hideki Yaginuma and Yukinobu Taniguchi},
    journal={Complex & Intelligent Systems },
    year={2025}
}
```

## Acknowledgments

This project is built upon the excellent [MMYOLO](https://github.com/open-mmlab/mmyolo) framework by OpenMMLab. We thank the MMYOLO team for providing such a comprehensive and flexible codebase.

## License

This project follows the same license as MMYOLO (GPL-3.0). Please refer to the LICENSE file for details.

## Related Projects

- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision

## Contact

For questions and issues, please:
1. Check the [MMYOLO documentation](https://mmyolo.readthedocs.io/)
2. Open an issue in this repository
3. Contact the authors through the paper

---

**Note**: This implementation is designed specifically for vehicle detection in CCTV scenarios. For other object detection tasks, you may need to adjust the configuration and hyperparameters accordingly.
