import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image 
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

class DataPipeline(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size=16, img_size=512, fold_idx=0, n_splits=5):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.fold_idx = fold_idx
        self.n_splits = n_splits

        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        # Load all paths and labels
        train_dir = self.dataset_dir / 'train'
        all_paths = []
        all_labels = []

        folder_map = {'Healthy': 0, 'SickNonTB': 1, 'TB': 2}
        for folder, label in folder_map.items():
            folder_path = train_dir / folder
            if folder_path.exists():
                paths = list(folder_path.rglob('*.png'))
                all_paths.extend(paths)
                all_labels.extend([label] * len(paths))
                print(f"Found {len(paths)} in {folder}")

        all_labels = np.array(all_labels)

        # Stratified split
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(np.arange(len(all_paths)), all_labels))
        train_idx, val_idx = splits[self.fold_idx]

        # Create datasets
        train_paths = [all_paths[i] for i in train_idx]
        val_paths = [all_paths[i] for i in val_idx]
        test_paths = list((self.dataset_dir / 'test').rglob('*.png'))

        # Use simple PathDataset - no Subset inheritance!
        self.train_dataset = PathDataset(train_paths, self.train_transform)
        self.val_dataset = PathDataset(val_paths, self.val_transform)
        self.test_dataset = PathDataset(test_paths, self.val_transform)

        print(f"\nDataPipeline(): Fold {self.fold_idx + 1}/{self.n_splits}:")
        print(f"DataPipeline(): Train: {len(train_paths)}")
        print(f"DataPipeline(): Val:   {len(val_paths)}")
        print(f"DataPipeline(): Test:  {len(test_paths)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


class PathDataset(Dataset):

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert('L')

        # Get label from parent folder
        folder = img_path.parent.name
        label = {'Healthy': 0, 'SickNonTB': 1, 'TB': 2}.get(folder, 2)

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label