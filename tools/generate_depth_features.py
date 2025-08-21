import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import time
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms

def download_midas_model():
    """下载并缓存MiDaS模型"""
    try:
        # 简化：直接使用torch hub加载，让torch hub处理缓存
        print("Loading MiDaS model...")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_midas_model():
    """Load MiDaS model for monocular depth estimation"""
    try:
        # 检查GPU可用性
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("GPU not available, using CPU")
        
        # 下载或加载模型
        model = download_midas_model()
        if model is None:
            return None
            
        model.eval()
        
        # 如果有GPU则使用GPU
        if torch.cuda.is_available():
            model = model.cuda()
            
        return model
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        return None

def create_midas_transform():
    """创建MiDaS的预处理transform"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path, transform, target_size=(256, 256)):
    """Preprocess image for MiDaS model"""
    # 读取图像
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    # 转换为PIL图像
    image_pil = Image.fromarray(image.astype(np.uint8))
    
    # 应用transform
    image_tensor = transform(image_pil)
    
    # 如果有GPU，将tensor移到GPU
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    return image_tensor.unsqueeze(0)

def estimate_depth(model, image_tensor, original_size=None):
    """Estimate depth using MiDaS model"""
    with torch.no_grad():
        # 预测深度
        prediction = model(image_tensor)
        
        # 如果输出是tuple，取第一个元素
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        
        # 确保prediction是2D或3D tensor
        if prediction.dim() == 4:
            prediction = prediction.squeeze(0).squeeze(0)
        elif prediction.dim() == 3:
            prediction = prediction.squeeze(0)
        
        # 调整大小到原始尺寸
        if original_size and prediction.shape != original_size[::-1]:
            prediction = F.interpolate(
                prediction.unsqueeze(0).unsqueeze(0),
                size=original_size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        
        # 转换为numpy数组
        depth = prediction.cpu().numpy()
        
        # 归一化到[0, 1]
        if depth.max() > depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        else:
            depth = np.zeros_like(depth)
        
        return depth

def process_dataset(input_dir, output_dir, model=None):
    """Process entire dataset to generate depth maps"""
    if model is None:
        model = load_midas_model()
        if model is None:
            print("Failed to load model")
            return
    
    # 创建transform
    print("Creating transform...")
    transform = create_midas_transform()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # 获取所有图像文件
    image_files = [f for f in input_path.rglob('*') 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    success_count = 0
    error_count = 0
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # 读取原始图像尺寸
            original_image = cv2.imread(str(image_file))
            if original_image is None:
                print(f"Failed to read image: {image_file}")
                error_count += 1
                continue
            original_size = (original_image.shape[1], original_image.shape[0])
            
            # 预处理图像
            image_tensor = preprocess_image(str(image_file), transform)
            if image_tensor is None:
                print(f"Failed to preprocess image: {image_file}")
                error_count += 1
                continue
            
            # 估计深度
            depth_map = estimate_depth(model, image_tensor, original_size)
            
            # 保存深度图
            relative_path = image_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.npy')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为numpy数组
            np.save(str(output_file), depth_map)
            
            # 可选：保存可视化图像
            vis_file = output_path / relative_path.with_name(relative_path.stem + '_depth.png')
            vis_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为可视化格式
            depth_vis = (depth_map * 255).astype(np.uint8)
            cv2.imwrite(str(vis_file), depth_vis)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            error_count += 1
            continue
    
    print(f"Processing completed: {success_count} successful, {error_count} failed")

def load_depth_map(depth_path):
    """Load depth map from saved numpy file"""
    if depth_path.endswith('.npy'):
        return np.load(depth_path)
    else:
        # 如果是图像格式，直接读取
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        return depth_img.astype(np.float32) / 255.0

def create_enhanced_features(rgb_image, depth_map):
    """Create enhanced features by combining RGB and depth"""
    # 确保深度图与RGB图像尺寸一致
    if depth_map.shape[:2] != rgb_image.shape[:2]:
        depth_map = cv2.resize(depth_map, (rgb_image.shape[1], rgb_image.shape[0]))
    
    # 如果深度图是2D，扩展为3D
    if depth_map.ndim == 2:
        depth_map = depth_map[:, :, np.newaxis]
    
    # 组合特征
    enhanced_features = np.concatenate([rgb_image, depth_map], axis=2)
    
    return enhanced_features

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate depth features for RGB images using MiDaS model')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing RGB images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save depth maps')
    parser.add_argument('--model_type', type=str, default='small',
                       choices=['small', 'large'],
                       help='MiDaS model type (small is faster, large is more accurate)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading MiDaS {args.model_type} model...")
    model = load_midas_model()
    
    if model is None:
        print("Failed to load MiDaS model.")
        exit(1)
    
    print(f"Processing images from {args.input_dir}...")
    print(f"Saving depth maps to {args.output_dir}...")
    
    # 处理数据集
    process_dataset(args.input_dir, args.output_dir, model)
    
    print("Depth feature generation completed!")
    print(f"Depth maps saved to: {args.output_dir}")
    print("You can now use these depth maps to enhance your GGMix features.") 