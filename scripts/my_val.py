import argparse
import os
from yolov5 import val

def evaluate_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=os.path.abspath('../runs/train/my_model_run/weights/best.pt'))
    parser.add_argument('--data', type=str, default=os.path.abspath('../datasets/my_dataset.yaml'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.6)
    parser.add_argument('--task', type=str, default='val')  # 'val', 'test', 'study'
    parser.add_argument('--name', type=str, default='my_model_eval')
    parser.add_argument('--project', type=str, default=os.path.abspath('../runs/val'))
    
    opt = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(opt.project, exist_ok=True)
    
    # 运行验证
    val.run(**vars(opt))

if __name__ == '__main__':
    evaluate_model()