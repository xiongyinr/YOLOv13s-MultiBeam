import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os

if __name__ == '__main__':
    model_path = r'D:\\毕业论文\\YOLO-HSPN\\00-结果\\yolo13s_baseline\\weights\\best.pt'
    model = YOLO(model_path)
    results = model.predict(
        source=r'D:\\毕业论文\\YOLO-HSPN\\dataset\\test\\images',
        imgsz=640,
        batch=9,
        device='0',
        save=True,        # 保存带检测框的图像
        save_txt=False,    # 保存 YOLO 格式标签 (可选)
        save_conf=True, 
        project='D:/毕业论文/YOLO-HSPN/00-结果/测试集',
        name='测试集yolo13s_predict',
    )