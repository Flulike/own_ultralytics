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
    yaml_path = '/home/guo/own_ultralytics/ultralytics/cfg/datasets/_visdrone.yaml'
    check_path(yaml_path)
    # model = RTDETR('ultralytics/cfg/models/uavdetr-r50.yaml')
    model = RTDETR('/home/guo/own_ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    model = RTDETR("rtdetr-l.pt")  # load a pretrained model (recommended for training)
    model = RTDETR("/home/guo/own_ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml").load("rtdetr-l.pt")  # build from YAML and transfer weights
    device = [2]
    optimizer = 'Adamw'
    model.train(data=str(yaml_path),
                imgsz=640,
                epochs=64,
                device=device,
                # resume='', # last.pt path
                project='results/ultralytics/RTDETR',
                name='rtdetr-l_visdrone_vml6_',
                )