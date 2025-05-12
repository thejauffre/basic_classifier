import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model
class SimpleClassifier(nn.Module):
    def __init__(self, image_height, image_width, input_channels=3, num_classes=100):
        super(SimpleClassifier, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64 * (image_height // 8) * (image_width // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_l):
        outputs = []
        for x in x_l:
            x = x.unsqueeze(0)
            x = F.relu(self.conv1(x))      # Conv2D(16)
            x = self.pool(x)               # MaxPooling2D
            x = F.relu(self.conv2(x))      # Conv2D(32)
            x = self.pool(x)               # MaxPooling2D
            x = F.relu(self.conv3(x))      # Conv2D(64 with L2 regularization)
            x = self.pool(x)               # MaxPooling2D
            x = self.dropout(x)            # Dropout(0.4)
            x = self.flatten(x)            # Flatten
            x = F.relu(self.fc1(x))        # Dense(128)
            x = self.fc2(x) 
            outputs.append(x.squeeze(0))
        return torch.stack(outputs)

def main():
    im_h = 300
    im_w = 300
    model = SimpleClassifier(image_height=im_h, image_width=im_w, num_classes=25)
    
    inputs = [
        torch.randn(3, im_h, im_w),  # square size
        torch.randn(3, im_h, im_w),  # square size
        torch.randn(3, im_h, im_w),  # square size
    ]
    
    output = model(inputs)
    print(f"Output shape: {output.shape}")  # Should be [3, num_classes]
    
    # Model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    print(output)
    preds = torch.argmax(output, dim=1)
    print(preds)
    
if __name__ == "__main__":
    main()
