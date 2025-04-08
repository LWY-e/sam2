import os

import cv2

import numpy as np

import random

from glob import glob

import matplotlib.pyplot as plt

import torch

from ultralytics import YOLO

from segment_anything import sam_model_registry, SamPredictor



# ✅ [3] 모델 경로 설정

YOLO_MODEL_PATH = "/content/drive/MyDrive/Comp/best.pt"

SAM2_MODEL_PATH = "/content/drive/MyDrive/Comp/sam2.1_hiera_large.pt"

IMAGE_DIR = "/content/drive/MyDrive/Comp/images"



# ✅ [4] 모델 로딩

# YOLO 모델 불러오기

yolo_model = YOLO(YOLO_MODEL_PATH)



def load_sam2_model(weight_path, model_type="vit_h"):

    state_dict = torch.load(weight_path, map_location=torch.device("cuda"))

    if "model" in state_dict:

        state_dict = state_dict["model"]

    sam = sam_model_registry[model_type](checkpoint=None)

    sam.load_state_dict(state_dict, strict=False)

    sam.to("cuda").eval()

    return SamPredictor(sam)



predictor = load_sam2_model(SAM2_MODEL_PATH)



# ✅ [5] 이미지 전처리 함수 및 시각화 함수



def apply_segmentation_overlay(image, mask, color=(0, 0, 255), alpha=0.5):

    overlay = image.copy()

    overlay[mask > 0] = (1 - alpha) * overlay[mask > 0] + alpha * np.array(color)

    return overlay.astype(np.uint8)



# 결과 시각화 함수

def plot_results(rows):

    fig, axs = plt.subplots(nrows=len(rows), ncols=2, figsize=(10, 5 * len(rows)))

    for i, (yolo_img, sam2_img) in enumerate(rows):

        axs[i, 0].imshow(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB))

        axs[i, 0].set_title(f"YOLO Result {i+1}")

        axs[i, 1].imshow(cv2.cvtColor(sam2_img, cv2.COLOR_BGR2RGB))

        axs[i, 1].set_title(f"YOLO + SAM2 Result {i+1}")

        axs[i, 0].axis('off')

        axs[i, 1].axis('off')

    plt.tight_layout()

    plt.show()



# 이미지 처리 및 모델 적용

image_paths = random.sample(glob(os.path.join(IMAGE_DIR, '*')), 4)

result_pairs = []



for path in image_paths:

    image = cv2.imread(path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    # YOLO 탐지

    with torch.no_grad():

        yolo_result = yolo_model(image)[0]



    yolo_annotated = image.copy()

    sam2_annotated = image.copy()



    for box in yolo_result.boxes.xyxy.cpu().numpy():

        x1, y1, x2, y2 = map(int, box)



        # 바운딩 박스 시각화

        cv2.rectangle(yolo_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)



        # ▶ 바운딩 박스 영역 Crop

        cropped_img = image_rgb[y1:y2, x1:x2]

        if cropped_img.size == 0:

            continue



        predictor.set_image(cropped_img)



        h_crop, w_crop = cropped_img.shape[:2]

        point = np.array([[w_crop // 2, h_crop // 2]])

        label = np.array([1])



        # SAM2 세그멘테이션

        masks, scores, _ = predictor.predict(

            point_coords=point,

            point_labels=label,

            multimask_output=True

        )



        if masks is None or len(masks) == 0:

            continue



        best_mask = masks[np.argmax(scores)].astype(np.uint8)



        # ▶ 원본 크기로 마스크 복원

        resized_mask = cv2.resize(best_mask, (x2 - x1, y2 - y1))

        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        full_mask[y1:y2, x1:x2] = resized_mask



        # ▶ 마스킹 적용

        sam2_annotated = apply_segmentation_overlay(sam2_annotated, full_mask)



    result_pairs.append((yolo_annotated, sam2_annotated))



# 시각화

for i, (yolo_img, sam2_img) in enumerate(result_pairs):

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)

    plt.imshow(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB))

    plt.title("YOLO Detection")

    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.imshow(cv2.cvtColor(sam2_img, cv2.COLOR_BGR2RGB))

    plt.title("YOLO + SAM2 Segmentation")

    plt.axis("off")

    plt.show()
