import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载你训练好的模型 (改成你 best.pt 的实际路径)
    model_path = r'D:\\毕业论文\\YOLO-HSPN\\00-结果\\yolo13s_baseline\\weights\\last.pt'
    model = YOLO(model_path)

    # 2. 运行验证 (Val)
    # data: 必须和训练时用的同一个data.yaml
    # split: 可以指定 'val' 或 'test' (取决于你的 data.yaml 里有没有设置 test 集)
    metrics = model.val(
        data=r'data.yaml',  
        imgsz=640,
        batch=9,
        device='0',
        split='test', 
        verbose=True,
        project='D:/毕业论文/YOLO-HSPN/00-结果/测试集',
        name='测试集yolo13s_p2_light2',
    )

    # 3. 打印关键指标
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")