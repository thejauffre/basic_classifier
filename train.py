from dataset import *
from model import *
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = "/path/to/dataset"
output_path = "output/"
input_h = 300
input_w = 300
input_channels = 3
batch_size = 8
epochs = 100

class Trainer():
    def __init__(self):
        self.init_dataset()
        self.init_model()

    def init_dataset(self):
        train_loader, train_dataset, val_loader, val_dataset = get_dataloaders(
            data_dir=data_path, 
            input_h=input_h,
            input_w=input_w,
            in_channels=input_channels,
            batch_size=batch_size)
        self.dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        self.trainSteps = len(train_dataset) // batch_size
        self.testSteps = len(val_dataset) // batch_size
        self.num_classes = train_dataset.classes
        print(f"\nNumber of classes: {len(self.num_classes)}")
        print(f"Class names: {self.num_classes}")
        self.H = {"train_loss": [], "test_loss": []}

    def init_model(self):        
        self.model = SimpleClassifier(image_height=input_h,
                                 image_width=input_w, 
                                 input_channels=input_channels,
                                 num_classes=len(self.num_classes))
        self.model = self.model.to(device)
        # Model summary
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params:,}")
        # Loss function, optimizer and scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.01)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def train(self):
        best_acc = 0.0
        self.H = {'train_loss': [], 'train_acc': [],
                    'val_loss': [], 'val_acc': []}
        start_time = time.time()
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corrects = 0
                total_samples = 0

                # Iterate over data
                for inputs, labels in tqdm(self.dataloaders[phase], desc=phase):
                    # Convert inputs from list to batch tensor
                    labels = torch.tensor(labels, dtype=torch.long).to(device)
                    inputs = [img.to(device) for img in inputs]

                    self.optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Statistics
                    running_loss += loss.item() * len(inputs)
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += len(inputs)

                if phase == 'train' and self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples
                self.H[f'{phase}_loss'].append(epoch_loss)
                self.H[f'{phase}_acc'].append(epoch_acc.item())

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # If best validation accuracy so far, save the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # Save the best model
                    torch.save({
                        'epoch': epoch,
                        'accuracy': best_acc,
                        'model_state_dict': self.model.state_dict(),
                    }, output_dir/'best.pth')
                    print(f'New best model saved with accuracy: {best_acc:.4f}')

            # Save checkpoint 
            torch.save({
                'epoch': epoch,
                'accuracy': best_acc,
                'model_state_dict': self.model.state_dict(),
            }, output_dir/f'epoch_{epoch+1}.pth')

            print()

        time_elapsed = time.time() - start_time
        
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
