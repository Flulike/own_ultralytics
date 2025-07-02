import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch



import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).parent
    # 构建相对路径
    yaml_path = 'ultralytics/cfg/datasets/VisDrone.yaml'
    # model = RTDETR('ultralytics/cfg/models/uavdetr-r50.yaml')
    model = RTDETR('rtdetr-l.yaml')
    model = RTDETR("rtdetr-l.pt")  # load a pretrained model (recommended for training)
    model = RTDETR("rtdetr-l.yaml").load("rtdetr-l.pt")  # build from YAML and transfer weights
    device = [2, 3]
    optimizer = 'Adamw'
    model.train(data=str(yaml_path),
                epochs=100,
                batch=16,
                device=device,
                optimizer=optimizer,
                # resume='', # last.pt path
                project='results/ultralytics/RTDETR',
                name='l_visdrone_vml4_',
                amp=False,
                )