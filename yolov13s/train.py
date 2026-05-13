import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(model=r'D:\\YOLO\\yolov13-main\\ultralytics\\cfg\\models\\v13\\yolov13.yaml')
    model = YOLO('yolov13s.pt')
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=150,
                batch=8,
                workers=4,
                device='0',
                optimizer='auto',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='yolo13s_baseline',
                single_cls=False,
                cache='disk',
                amp=True,
                cos_lr=True,
                patience=30,
                seed=42,
                deterministic=True,
                )