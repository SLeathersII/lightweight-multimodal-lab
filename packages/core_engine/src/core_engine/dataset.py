# packages/core_engine/src/core_engine/dataset.py
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OfficialCLEVRMultiModalDataset(Dataset):
    """
    Dataset loader for official CLEVR: Pairs RGB images with 3D point cloud scene representations.
    """

    def __init__(self, data_dir: str, split: str = "train", num_points: int = 512, img_size: int = 128):
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points

        self.img_dir = os.path.join(data_dir, "images", split)
        self.scene_json = os.path.join(data_dir, "scenes", f"CLEVR_{split}_scenes.json")

        with open(self.scene_json, 'r') as f:
            self.scenes = json.load(f)["scenes"]

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _object_to_points(self, obj: dict, points_per_obj: int = 64) -> np.ndarray:
        """Converts object 3D bounding parameters into a 3D surface point cloud."""
        pos = np.array(obj["3d_coords"], dtype=np.float32)
        shape = obj["shape"]
        size = 0.7 if obj["size"] == "large" else 0.35

        # Sample local surface geometry
        if shape == "cube":
            pts = np.random.uniform(-size, size, (points_per_obj, 3))
        elif shape == "sphere":
            vec = np.random.randn(points_per_obj, 3)
            pts = (vec / np.linalg.norm(vec, axis=1, keepdims=True)) * size
        else:  # cylinder
            theta = np.random.uniform(0, 2 * np.pi, points_per_obj)
            z = np.random.uniform(-size, size, points_per_obj)
            x = np.cos(theta) * size
            y = np.sin(theta) * size
            pts = np.stack([x, y, z], axis=1)

        return pts + pos  # Shift to 3D world coordinate center

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int):
        scene = self.scenes[idx]

        # 1. Load RGB Modality (X_RGB)
        img_path = os.path.join(self.img_dir, scene["image_filename"])
        image = Image.open(img_path).convert("RGB")
        rgb_tensor = self.img_transform(image)  # Shape: [3, H, W]

        # 2. Extract Ground Truth 3D Spatial Modality (X_3D Point Cloud)
        objects = scene["objects"]
        pts_per_obj = max(16, self.num_points // max(1, len(objects)))
        scene_points = []

        for obj in objects:
            scene_points.append(self._object_to_points(obj, points_per_obj=pts_per_obj))

        if scene_points:
            pc_array = np.vstack(scene_points)
        else:
            pc_array = np.zeros((self.num_points, 3), dtype=np.float32)

        # Resample or pad to fixed token length N
        if len(pc_array) > self.num_points:
            indices = np.random.choice(len(pc_array), self.num_points, replace=False)
            pc_array = pc_array[indices]
        elif len(pc_array) < self.num_points:
            pad = np.zeros((self.num_points - len(pc_array), 3), dtype=np.float32)
            pc_array = np.vstack([pc_array, pad])

        point_cloud_tensor = torch.tensor(pc_array, dtype=torch.float32)  # Shape: [N, 3]

        return {
            "rgb": rgb_tensor,
            "point_cloud": point_cloud_tensor
        }