import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from timm.models.vision_transformer import vit_small_patch16_224
from lightly.models.modules import MaskedVisionTransformerTIMM

class DINOv2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone.encode(x)
        cls_token = features[:, 0]
        return self.fc(cls_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

print("Preparing CIFAR-10 Dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                           shuffle=True, num_workers=4, drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                          shuffle=False, num_workers=4)

vit = vit_small_patch16_224(
    pos_embed="learn",
    dynamic_img_size=True,
    init_values=1e-5,
)
backbone = MaskedVisionTransformerTIMM(
    vit=vit,
    antialias=False,
    pos_embed_initialization="skip",
)

try:
    state_dict = torch.load("dinov2_teacher_backbone.pth", map_location=device, weights_only=True)
    backbone.load_state_dict(state_dict)
    print("Successfully loaded pre-trained DINOv2 backbone weights.")
except FileNotFoundError:
    print("Warning: 'dinov2_teacher_backbone.pth' not found. Training with randomly initialized weights.")

num_classes = 10 
classifier = DINOv2Classifier(backbone, num_classes=num_classes)
classifier = classifier.to(device)

for param in classifier.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(classifier.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 5

print("\nStarting Training...")
for epoch in range(epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 100 == 99:
            train_acc = 100 * correct / total
            avg_loss = running_loss / 100
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")
            running_loss = 0.0
            correct = 0
            total = 0

    classifier.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f"==> Epoch {epoch+1} Test Accuracy: {test_acc:.2f}% <==\n")

print("Training Complete!")