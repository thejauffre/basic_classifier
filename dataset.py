import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
from torchvision.transforms import v2

class ClassificationDataset(Dataset):

    def __init__(self, root_dir, split='train', in_channels=3, transform=None):
        # store the image and mask filepaths, and augmentation
        self.root_dir = os.path.join(root_dir, split)
        self.in_channels = in_channels
        self.transform = transform if transform else transforms.ToTensor()

        self.classes = sorted(os.listdir(self.root_dir))
        self.class_index = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_index[class_name]
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    self.images.append(img_path)
                    self.labels.append(class_idx)


    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.images)

    def __getitem__(self, idx):
        # grab the image path from the current index
        if self.in_channels == 1:
            image = Image.open(self.images[idx]).convert('L')  # Convert to grayscale
        else:
            image = Image.open(self.images[idx]).convert('RGB')  # Convet to rgb
        
        image = self.transform(image)
        label = self.labels[idx]

        # return a tuple of the image and its label
        return image, label

def collate_fn(batch):
    images = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    return images, labels

def get_dataloaders(data_dir,
                    in_channels=3,
                    input_h=300,
                    input_w=300,
                    batch_size=8, 
                    num_workers=4):
    
    if in_channels == 3:
        transform = v2.Compose([
            v2.Resize(size=(input_h, input_w), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = v2.Compose([
            v2.Resize(size=(input_h, input_w), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])


    # Create datasets
    train_dataset = ClassificationDataset(
        root_dir=data_dir,
        in_channels=in_channels,
        split='train',
        transform=transform
    )
    
    val_dataset = ClassificationDataset(
        root_dir=data_dir,
        in_channels=in_channels,
        split='val',
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, train_dataset, val_loader, val_dataset
