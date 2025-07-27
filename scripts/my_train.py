import argparse
import os
from yolov5 import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.path.abspath('../datasets/cones.yaml'))
    parser.add_argument('--cfg', type=str, default=os.path.abspath('../models/yolov5s.yaml'))
    parser.add_argument('--weights', type=str, default=os.path.abspath('../models/cone_model1.pt'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--name', type=str, default='my_model_run')
    parser.add_argument('--project', type=str, default=os.path.abspath('../runs/train'))
    
    opt = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(opt.project, exist_ok=True)
    
    # 开始训练
    train.run(**vars(opt))

if __name__ == '__main__':
    main()