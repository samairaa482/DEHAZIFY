import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class RESIDEDataset(Dataset):
    def __init__(self, hazy_dir, gt_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.hazy_images = sorted(os.listdir(hazy_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        self.transform = transform

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        hazy_img = Image.open(hazy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.transform:
            hazy_img = self.transform(hazy_img)
            gt_img = self.transform(gt_img)

        return hazy_img, gt_img
