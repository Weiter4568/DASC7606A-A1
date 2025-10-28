# scripts/data_augmentation.py
import logging, random
# scripts/data_augmentation.py
import logging, random
from pathlib import Path
from typing import List, Tuple
import numpy as np, cv2
from PIL import Image
import numpy as np, cv2
from PIL import Image
import albumentations as A

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAugmenter:
    def __init__(self, augmentations_per_image=2, seed=42, save_original=True,
                 image_extensions=(".png",".jpg",".jpeg")):
    def __init__(self, augmentations_per_image=2, seed=42, save_original=True,
                 image_extensions=(".png",".jpg",".jpeg")):
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions
        random.seed(seed); np.random.seed(seed)
        self.transform = A.Compose([
            # 更强的几何变换
            A.PadIfNeeded(40, 40, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.RandomCrop(32, 32, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),  # 添加垂直翻转
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),  # 增加旋转角度
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            
            # 更强的颜色变换
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
            
            # 噪声和模糊
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=7, p=0.2),
            
            # 更强的dropout
            A.CoarseDropout(max_holes=2, max_height=12, max_width=12,
                            min_holes=1, min_height=6, min_width=6,
                            fill_value=(125,123,114), p=0.4),
            A.GridDropout(ratio=0.4, p=0.2),  # 网格dropout
            
            # 混合增强
            A.RandomGridShuffle(grid=(2, 2), p=0.1),
            # A.Cutout is not available in albumentations; replace with another dropout. Already using CoarseDropout above.
        ])

    def _find_image_files(self, root: Path) -> List[Path]:
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        # 跳过已生成的文件，保证幂等
        files = [p for p in files if not (p.name.startswith("aug_") or p.name.startswith("orig_"))]
        # 跳过已生成的文件，保证幂等
        files = [p for p in files if not (p.name.startswith("aug_") or p.name.startswith("orig_"))]
        return files

    def augment_image(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img)
        out = self.transform(image=arr)["image"]
        return Image.fromarray(out.astype(np.uint8))

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        in_path, out_path = Path(input_dir), Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        done_flag = out_path / ".aug_done"
        if done_flag.exists():
            logger.info(f"Augmentation already done at {out_path}, skipping.")
            return

        imgs = self._find_image_files(in_path)
        logger.info(f"Found {len(imgs)} images to augment.")
        cnt = 0
        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}"); continue
            rel = img_path.parent.relative_to(in_path)
            tgt = out_path / rel; tgt.mkdir(parents=True, exist_ok=True)

            if self.save_original:
                img.save(tgt / f"orig_{img_path.stem}.png", format="PNG")

            for i in range(self.augmentations_per_image):
                aug = self.augment_image(img)
                aug.save(tgt / f"aug_{i}_{img_path.stem}.png", format="PNG")
                cnt += 1

        logger.info(f"Augmented {cnt} images. Output: {out_path}")
        done_flag.touch()

def augment_dataset(input_dir: str, output_dir: str, augmentations_per_image=2, seed=42) -> None:
    augmenter = ImageAugmenter(augmentations_per_image=augmentations_per_image, seed=seed, save_original=True)
    def augment_image(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img)
        out = self.transform(image=arr)["image"]
        return Image.fromarray(out.astype(np.uint8))

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        in_path, out_path = Path(input_dir), Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        done_flag = out_path / ".aug_done"
        if done_flag.exists():
            logger.info(f"Augmentation already done at {out_path}, skipping.")
            return

        imgs = self._find_image_files(in_path)
        logger.info(f"Found {len(imgs)} images to augment.")
        cnt = 0
        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}"); continue
            rel = img_path.parent.relative_to(in_path)
            tgt = out_path / rel; tgt.mkdir(parents=True, exist_ok=True)

            if self.save_original:
                img.save(tgt / f"orig_{img_path.stem}.png", format="PNG")

            for i in range(self.augmentations_per_image):
                aug = self.augment_image(img)
                aug.save(tgt / f"aug_{i}_{img_path.stem}.png", format="PNG")
                cnt += 1

        logger.info(f"Augmented {cnt} images. Output: {out_path}")
        done_flag.touch()

def augment_dataset(input_dir: str, output_dir: str, augmentations_per_image=2, seed=42) -> None:
    augmenter = ImageAugmenter(augmentations_per_image=augmentations_per_image, seed=seed, save_original=True)
    augmenter.process_directory(input_dir, output_dir)
