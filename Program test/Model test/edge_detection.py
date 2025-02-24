import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# **載入 YOLOv8 訓練的模型**
model = YOLO("yolov8m-pose.pt")  # 替換成你的模型

# **讀取電路圖**
image_path = "./Model test results/circuit001.png"  # 你的電路圖
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("找不到圖片，請確認 image_path 是否正確")

# **用 YOLO 偵測 Bounding Boxes**
results = model(image)
bboxes = results[0].boxes.xyxy.cpu().numpy()  # 取得 YOLO 偵測到的 bounding boxes

print(f"YOLO 偵測到 {len(bboxes)} 個元件")


# **函數：擴展 Bounding Box**
def expand_bbox(bbox, image_shape, expand_ratio=1.2):
    img_h, img_w = image_shape[:2]
    x_min, y_min, x_max, y_max = map(int, bbox)

    # 計算擴展大小
    width, height = x_max - x_min, y_max - y_min
    expand_x = int(width * (expand_ratio - 1) / 2)
    expand_y = int(height * (expand_ratio - 1) / 2)

    # 擴展 Bounding Box，確保不超過圖片邊界
    x_min = max(x_min - expand_x, 0)
    y_min = max(y_min - expand_y, 0)
    x_max = min(x_max + expand_x, img_w - 1)
    y_max = min(y_max + expand_y, img_h - 1)

    return (x_min, y_min, x_max, y_max)


# **函數：Edge Detection + Pin Extraction**
def extract_pin_points(image, bbox, expand_ratio=1.5):
    """
    依據 YOLO 偵測的 Bounding Box 提取感興趣區域（ROI），
    並使用 Edge Detection 找出 Pin Point。
    """
    # **擴展 Bounding Box**
    x_min, y_min, x_max, y_max = expand_bbox(bbox, image.shape, expand_ratio)

    # **提取 ROI**
    roi = image[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # **去雜訊**
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # **Canny 邊緣偵測**
    edges = cv2.Canny(blurred, 50, 150)

    # **尋找輪廓**
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # **初始化 Pin Points**
    pin_points = []

    # **篩選輪廓**
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 100:  # 過濾掉太小或太大的雜訊
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x_min
                cy = int(M["m01"] / M["m00"]) + y_min
                pin_points.append((cx, cy))

    return pin_points


# **畫出 YOLO 偵測框，並對每個框找 Pin Point**
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("YOLO 偵測元件 + Pin Points")

for bbox in bboxes:
    # **擴展 Bounding Box**
    x_min, y_min, x_max, y_max = expand_bbox(bbox, image.shape, expand_ratio=1.3)  # 設定擴展比例

    # **畫擴展後的 Bounding Box**
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                      linewidth=2, edgecolor="green", facecolor="none"))

    # **找出 Pin Points**
    pin_points = extract_pin_points(image, bbox, expand_ratio=1.3)

    # **畫出 Pin Points**
    for (px, py) in pin_points:
        plt.scatter(px, py, color="red", s=40)  # 紅色圓點標記 Pin

plt.show()
