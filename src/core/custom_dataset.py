import os
import cv2
import dlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Any, Callable, Optional

import config


TARGET_W, TARGET_H = 178, 218
TARGET_NOSE_X = 0.5 * TARGET_W
TARGET_NOSE_Y = 0.617 * TARGET_H
TARGET_FACE_WIDTH = 0.5 * TARGET_W


class CustomDataset(Dataset):
    def __init__(
            self,
            custom_dataset_path: str,
            transform: Optional[Callable] = None,
            num_calc_samples: Optional[int] = None,
            shuffle: bool = False
    ):
        self.custom_dataset_path = custom_dataset_path
        self.transform = transform
        self.num_calc_samples = num_calc_samples

        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        self.image_list = [
            f for f in os.listdir(custom_dataset_path)
            if f.lower().endswith(self.valid_extensions)
        ]

        if shuffle:
            np.random.shuffle(self.image_list)

        if self.num_calc_samples:
            self.image_list = self.image_list[:min(self.num_calc_samples, len(self.image_list))]

        print(f"[CustomDataset] CustomDataset __init__ success ({len(self.image_list)})")

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(
            self,
            idx: int
    ) -> tuple[Any, str]:
        img_name = self.image_list[idx]
        img_path = os.path.join(self.custom_dataset_path, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


def get_custom_dataset_loader(
        custom_dataset_path: str,
        batch_size: int = 64,
        image_size: int = 64,
        shuffle: bool = True,
        num_calc_samples: Optional[int] = None
) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(
        custom_dataset_path = custom_dataset_path,
        transform = transform,
        num_calc_samples = num_calc_samples,
        shuffle = shuffle
    )

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 8,
        pin_memory = not (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    )

    print(f"[CustomDataset] get_custom_dataset_loader success")
    return dataloader


def align_and_crop_face(image_path, output_path, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[CustomDataset] 이미지를 읽을 수 없습니다: {image_path}")
        return None

    # 얼굴 감지 (그레이스케일 변환)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        print("[CustomDataset] 얼굴을 찾을 수 없습니다.")
        return None

    # 가장 큰 얼굴 사용
    rect = max(rects, key=lambda r: r.width() * r.height())
    shape = predictor(gray, rect)

    # 랜드마크 좌표 추출
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    left_eye_pts = coords[36:42]
    right_eye_pts = coords[42:48]
    left_eye_center = left_eye_pts.mean(axis=0).astype("int")
    right_eye_center = right_eye_pts.mean(axis=0).astype("int")
    nose_point = coords[30]
    left_jaw = coords[0]
    right_jaw = coords[16]

    # 회전 각도 계산
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 스케일 및 변환 행렬 생성
    current_face_width = np.linalg.norm(right_jaw - left_jaw)
    scale = TARGET_FACE_WIDTH / current_face_width

    M = cv2.getRotationMatrix2D((float(nose_point[0]), float(nose_point[1])), angle, scale)
    M[0, 2] += (TARGET_NOSE_X - nose_point[0])
    M[1, 2] += (TARGET_NOSE_Y - nose_point[1])

    # 아핀 변환 적용 (Warping)
    aligned_img = cv2.warpAffine(img, M, (TARGET_W, TARGET_H), flags=cv2.INTER_CUBIC)

    # 저장 및 결과 출력
    cv2.imwrite(output_path, aligned_img)
    print(f"[CustomDataset] 변환 완료: {output_path} (크기: {TARGET_W} * {TARGET_H})")


def measure_face_stats(image_path, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[CustomDataset] 이미지를 읽을 수 없습니다: {image_path}")
        return None

    # 얼굴 감지 (그레이스케일 변환)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        print("[CustomDataset] 얼굴을 찾을 수 없습니다.")
        return None

    # 가장 큰 얼굴 사용
    rect = max(rects, key=lambda r: r.width() * r.height())
    shape = predictor(gray, rect)

    # 랜드마크 좌표 추출
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    nose_point = coords[30]
    left_jaw = coords[0]
    right_jaw = coords[16]
    nose_bridge_top = coords[27]
    chin_bottom = coords[8]

    # 길이 측정
    face_width = np.linalg.norm(right_jaw - left_jaw)
    face_height = np.linalg.norm(chin_bottom - nose_bridge_top)

    # 비율 계산
    image_width = img.shape[1]
    image_height = img.shape[0]
    width_occupancy = face_width / image_width
    height_occupancy = face_height / image_width
    nose_ratio = nose_point[1] / image_height

    # 결과 출력
    print(f"[CustomDataset] measure_face_stats results {image_path}")
    print(f"[CustomDataset] face_width: {face_width}")
    print(f"[CustomDataset] face_height: {face_height}")
    print(f"[CustomDataset] width_occupancy: {width_occupancy}")
    print(f"[CustomDataset] height_occupancy: {height_occupancy}")
    print(f"[CustomDataset] nose_ratio: {nose_ratio}")

    return width_occupancy, height_occupancy, nose_ratio


if __name__ == "__main__":
    predictor_file = os.path.join(config.project_dir, "model", "etc" , "shape_predictor_68_face_landmarks.dat")
    input_dir = os.path.join(config.custom_dataset_path, "raw")
    output_dir = config.custom_dataset_path

    """
    result_list = []
    for i in range(1_000):
        print(f"[CustomDataset] {i + 1}")
        result = measure_face_stats(
            os.path.join(config.project_dir, "dataset", "celebA", "img_align_celeba", "img_align_celeba", f"{i + 1 : 6}.jpg".replace(" ", "0")),
            predictor_file
        )
        print()
        if result is not None:
            result_list.append(result)
    print(f"Average width_occupancy: {sum(result[0] for result in result_list) / len(result_list)}")
    print(f"Average height_occupancy: {sum(result[1] for result in result_list) / len(result_list)}")
    print(f"Average nose_ratio: {sum(result[2] for result in result_list) / len(result_list)}")
    """

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_list = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ]

    for image in image_list:
        input_image = os.path.join(input_dir, image)
        output_image = os.path.join(output_dir, image)
        align_and_crop_face(input_image, output_image, predictor_file)

