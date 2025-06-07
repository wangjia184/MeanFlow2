import torch
from torchvision.utils import save_image
from network import MFDiT
from meanflow import MeanFlow
from ema_pytorch import EMA
import argparse

def generate_images(checkpoint_path, output_path, device='cuda'):
    # 1. 创建模型和MeanFlow
    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=1,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10
    ).to(device)
    
    # 创建EMA包装器
    ema = EMA(model)
    
    meanflow = MeanFlow(
        channels=1,
        image_size=32,
        num_classes=10,
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0
    )
    
    # 2. 加载checkpoint - 修正的关键部分
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重到EMA包装器
    ema.load_state_dict(checkpoint['ema_model'])
    
    # 使用EMA模型的ema_model进行推理
    ema_model = ema.ema_model.eval()
    
    # 3. 生成图像
    with torch.no_grad():
        generated_images = meanflow.sample_each_class(
            ema_model, 
            n_per_class=1, 
            device=device
        )
    
    # 4. 保存图像
    save_image(generated_images, output_path, nrow=10, normalize=True, pad_value=1.0)
    print(f"Images saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from MeanFlow model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_8000.pt', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated_images.png', help='Output image path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # 处理设备选择
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    generate_images(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device
    )