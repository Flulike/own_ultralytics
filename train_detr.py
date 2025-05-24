import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch



warnings.filterwarnings('ignore')


def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).parent
    # 构建相对路径
    yaml_path = 'ultralytics/cfg/datasets/Fisheye.yaml'
    check_path(yaml_path)
    # model = RTDETR('ultralytics/cfg/models/uavdetr-r50.yaml')
    model = RTDETR('rtdetr-resnet101.yaml')
    # model = RTDETR("rtdetr-l.pt")  # load a pretrained model (recommended for training)
    # model = RTDETR("rtdetr-l.yaml").load("rtdetr-l.pt")  # build from YAML and transfer weights
    device = [3]
    optimizer = 'Adamw'
    model.train(data=str(yaml_path),
                imgsz=640,
                epochs=100,
                device=device,
                # resume='', # last.pt path
                project='results/ultralytics/RTDETR',
                name='101_fisheye_vml4_',
                )