import os
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class LegoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Directory with processed data
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply to images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.anno_dir = os.path.join(root_dir, split, 'annotations')
        
        self.image_files = sorted([f for f in os.listdir(self.img_dir) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Get annotation
        base_name = os.path.splitext(img_name)[0]
        anno_path = os.path.join(self.anno_dir, f"{base_name}.xml")
        
        # Parse XML annotation
        boxes, labels, masks = self._parse_annotation(anno_path, image.size)
        
        # Create target dict
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create mask tensor (dummy masks as we're only doing object detection)
        masks = torch.zeros((num_objs, *image.size), dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target
    
    def _parse_annotation(self, anno_path, img_size):
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        masks = []  # We don't have masks, this is just a placeholder
        
        width, height = img_size
        
        for obj in root.findall('object'):
            # We're using a single class 'lego', so all labels are 1
            # (0 is reserved for background in many frameworks)
            labels.append(1)
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Normalize to stay within image bounds
            xmin = max(0, min(width - 1, xmin))
            ymin = max(0, min(height - 1, ymin))
            xmax = max(0, min(width, xmax))
            ymax = max(0, min(height, ymax))
            
            boxes.append([xmin, ymin, xmax, ymax])
        
        return boxes, labels, masks

# Transforms for training with Mask R-CNN
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = torch.from_numpy(np.array(image).transpose((2, 0, 1)))
        image = image.float() / 255.0
        return image, target

def get_transform(train):
    transforms = []
    # Convert PIL image to tensor
    transforms.append(ToTensor())
    return Compose(transforms)

def collate_fn(batch):
    """Custom collate function for data loader to handle variable sized objects."""
    return tuple(zip(*batch))

def get_data_loaders(data_dir, batch_size=4):
    """Create data loaders for training, validation and testing."""
    # Create datasets
    train_dataset = LegoDataset(data_dir, 'train', transform=get_transform(train=True))
    val_dataset = LegoDataset(data_dir, 'val', transform=get_transform(train=False))
    test_dataset = LegoDataset(data_dir, 'test', transform=get_transform(train=False))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=4
    )
    
    return train_loader, val_loader, test_loader