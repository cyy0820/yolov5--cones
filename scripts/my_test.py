import argparse
import os
from yolov5 import val
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=os.path.abspath('../runs/train/my_model_run/weights/best.pt'))
    parser.add_argument('--data', type=str, default=os.path.abspath('../datasets/my_dataset.yaml'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.6)
    parser.add_argument('--name', type=str, default='my_model_test')
    parser.add_argument('--project', type=str, default=os.path.abspath('../runs/test'))
    
    opt = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(opt.project, exist_ok=True)
    
    # 运行测试
    results = val.run(**vars(opt))
    
    # 保存测试结果
    save_results(results, opt.project, opt.name)

def save_results(results, project, name):
    # 创建结果目录
    results_dir = os.path.join(project, name, 'analysis')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存主要指标
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"mAP@0.5: {results[0]:.4f}\n")
        f.write(f"mAP@0.5:0.95: {results[1]:.4f}\n")
        f.write(f"Precision: {results[2]:.4f}\n")
        f.write(f"Recall: {results[3]:.4f}\n")
    
    # 读取并处理结果CSV
    results_file = list(Path(project).glob(f'{name}/*.csv'))[0]
    df = pd.read_csv(results_file)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    plt.imshow(df.iloc[:, 1:].values, cmap='viridis')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(range(len(df.columns[1:])), df.columns[1:], rotation=45)
    plt.yticks(range(len(df)), df.iloc[:, 0])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    
    # 绘制PR曲线
    pr_curve = list(Path(project).glob(f'{name}/*_curve.png'))[0]
    os.rename(pr_curve, os.path.join(results_dir, 'pr_curve.png'))
    
    print(f"测试结果已保存到: {results_dir}")

if __name__ == '__main__':
    test_model()