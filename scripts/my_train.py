import argparse
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)

from yolov5 import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.path.abspath('/home/cyy/yolov5--cones/dataset/cones.yaml'))
    parser.add_argument('--cfg', type=str, default=os.path.abspath('/home/cyy/yolov5--cones/yolov5/models/yolov5s.yaml'))
    parser.add_argument('--weights', type=str, default=os.path.abspath('/home/cyy/yolov5--cones/models/cone_model3.pt'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, nargs='+', default=[640, 640])
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument('--name', type=str, default='my_model_run')
    parser.add_argument('--project', type=str, default=os.path.abspath('/home/cyy/yolov5--cones/runs/train'))
    parser.add_argument('--device', type=str, default='0')  # 使用GPU 0
    parser.add_argument('--hyp', type=str, default=os.path.abspath('/home/cyy/yolov5--cones/yolov5/data/hyps/hyp.scratch-low.yaml'))
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    opt = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(opt.project, exist_ok=True)
    
    # 开始训练
    train.run(**vars(opt))

if __name__ == '__main__':
    main()