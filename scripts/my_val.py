import argparse
import os
import sys

# 将项目根目录添加到系统路径中
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)

from yolov5 import val

def evaluate_model():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser()
    
    # 2. 定义所有需要的参数和它们的默认值
    parser.add_argument('--weights', type=str, default=os.path.abspath('/home/cyy/yolov5--cones/runs/train/my_model_run/weights/best.pt'), help='model.pt path(s)')
    parser.add_argument('--data', type=str, default=os.path.abspath('/home/cyy/yolov5--cones/dataset/cones.yaml'), help='dataset.yaml path')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', type=str, default='val', help='val, test, or study')
    parser.add_argument('--name', type=str, default='my_model_eval', help='save to project/name')
    parser.add_argument('--project', type=str, default=os.path.abspath('/home/cyy/yolov5--cones/runs/val'), help='save to project/name')
    
    # 3. 解析参数，创建 opt 对象
    opt = parser.parse_args()
    
    # 【修正】下面这几行检查和转换的代码是多余的，已经删除。
    # 我们直接将 argparse 解析出的整数 imgsz 传递给 val.run 即可。

    # 4. 确保保存结果的目录存在
    os.makedirs(opt.project, exist_ok=True)
    
    # 5. 运行验证脚本
    print("参数解析完成，开始运行评估...")
    val.run(**vars(opt))
    print("评估完成。")

if __name__ == '__main__':
    evaluate_model()