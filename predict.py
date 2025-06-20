import os
import numpy as np
import cv2
from ultralytics import YOLO

def sliding_window_predict(
    model_path,
    input_dir,
    output_dir,
    patch_size=640,
    stride=512,
    conf=0.5,
    iou=0.7,
    imgsz=640,
    device=0
):
    
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for idx, file in enumerate(files):
        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]

        nh = int(np.ceil((height - patch_size) / stride + 1))
        new_height = (nh - 1) * stride + patch_size
        pad_h = (new_height - height) // 2

        nw = int(np.ceil((width - patch_size) / stride + 1))
        new_width = (nw - 1) * stride + patch_size
        pad_w = (new_width - width) // 2

        new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        new_img[pad_h:pad_h + height, pad_w:pad_w + width, :] = img

        total_boxes = []
        total_classes = []
        for i in range(nh):
            for j in range(nw):
                y1, y2 = i * stride, i * stride + patch_size
                x1, x2 = j * stride, j * stride + patch_size
                patch = new_img[y1:y2, x1:x2, :]
                results = model.predict(
                    patch,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    verbose=False
                )
                for r in results:
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    for box, score, cls in zip(r.boxes.xywh, r.boxes.conf, r.boxes.cls):
                        if score < conf:
                            continue
                        x, y, w, h = box.cpu().numpy()
                        x += x1
                        y += y1
                        total_boxes.append([x, y, w, h])
                        total_classes.append(int(cls.cpu().item()))

        yolo_predictions = []
        for box, cls in zip(total_boxes, total_classes):
            x, y, w, h = box
            x_center = x / new_width
            y_center = y / new_height
            width_norm = w / new_width
            height_norm = h / new_height
            yolo_predictions.append(f"{cls} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")

        pred_txt_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.txt')
        with open(pred_txt_path, 'w') as f:
            for pred in yolo_predictions:
                f.write(pred + '\n')

if __name__ == "__main__":
    sliding_window_predict(
        model_path='best.pt',
        input_dir='test',
        output_dir='my_predict',
        patch_size=640,
        stride=512,
        conf=0.5,
        iou=0.7,
        imgsz=640,
        device=0
    )
