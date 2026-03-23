import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image 
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

class DataPipeline(pl.LightningDataModule):

    def __init__(self, dataset_dir, batch_size=16, img_size=512, fold_idx=0, n_splits=5, tuning=True):

        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.fold_idx = fold_idx
        self.n_splits = n_splits
        self.tuning = tuning

        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):

        train_dir = self.dataset_dir / 'train'
        all_train_paths = self._get_image_paths(train_dir)

        print(f"\nDataPipeline.setup(): Tuning model? [{self.tuning}]")

        if (self.tuning):

            labels_for_split = [self._extract_label(p) for p in all_train_paths]
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = list(skf.split(np.arange(len(all_train_paths)), np.array(labels_for_split)))

            train_idx, val_idx = splits[self.fold_idx]
            train_paths = [all_train_paths[i] for i in train_idx]
            val_paths = [all_train_paths[i] for i in val_idx]

            self.train_dataset = PathDataset(train_paths, self.train_transform)
            self.val_dataset = PathDataset(val_paths, self.val_test_transform)
            self.test_dataset = None

            print(f"DataPipeline.setup(): Fold {self.fold_idx + 1}/{self.n_splits}:")
            print(f"DataPipeline.setup(): Train: {len(train_paths)}")
            print(f"DataPipeline.setup(): Val:   {len(val_paths)}")

        else:
            
            test_dir = self.dataset_dir / 'test'
            all_test_paths = self._get_image_paths(test_dir)

            self.train_dataset = PathDataset(all_train_paths, self.train_transform)
            self.val_dataset = None
            self.test_dataset = PathDataset(all_test_paths, self.val_test_transform) 
            
            print(f"DataPipeline.setup(): Train: {len(all_train_paths)}")
            print(f"DataPipeline.setup(): Test:  {len(all_test_paths)}")


    def _get_image_paths(self, base_dir):
        return list(base_dir.rglob('*.png'))
       
    def _extract_label(self, path):
        folder = path.parent.name
        return {'Healthy': 0, 'SickNonTB': 1, 'TB': 2}.get(folder, 2)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return None if self.val_dataset is None else DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self):
        return None if self.val_dataset is None else DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


class PathDataset(Dataset):

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert('L')

        folder = img_path.parent.name
        label = {'Healthy': 0, 'SickNonTB': 1, 'TB': 2}.get(folder, 2)

        if self.transform:
            image = self.transform(image)

        return image, label