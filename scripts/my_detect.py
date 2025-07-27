import argparse
import os
import cv2
from yolov5 import detect
from pathlib import Path

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=os.path.abspath('../runs/train/my_model_run/weights/best.pt'))
    # 0 代表摄像头，也可以是文件/目录路径、URL
    parser.add_argument('--source', type=str, default=os.path.abspath('../datasets/images/detect'))
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--project', type=str, default=os.path.abspath('../runs/detect'))
    parser.add_argument('--name', type=str, default='my_detection_run')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--view-img', action='store_true', help='show results')
    
    opt = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(opt.project, exist_ok=True)
    
    # 运行检测
    detect.run(**vars(opt))

def detect_on_video(input_path, output_path):
    """处理视频文件"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=os.path.abspath('../runs/train/my_model_run/weights/best.pt'))
    parser.add_argument('--source', type=str, default=input_path)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--project', type=str, default=os.path.dirname(output_path))
    parser.add_argument('--name', type=str, default=os.path.basename(output_path).split('.')[0])
    
    opt = parser.parse_args()
    
    detect.run(**vars(opt))
    return os.path.join(opt.project, opt.name, input_path.split('/')[-1])

def detect_on_image(image_path, output_dir):
    """处理单张图像"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=os.path.abspath('../runs/train/my_model_run/weights/best.pt'))
    parser.add_argument('--source', type=str, default=image_path)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--project', type=str, default=output_dir)
    parser.add_argument('--name', type=str, default='image_detections')
    
    opt = parser.parse_args()
    
    detect.run(**vars(opt))
    
    # 返回处理后的图像路径
    result_dir = os.path.join(output_dir, 'image_detections')
    return next(Path(result_dir).glob('*.jpg' if image_path.lower().endswith('.jpg') else '*.png'))

if __name__ == '__main__':
    run_inference()