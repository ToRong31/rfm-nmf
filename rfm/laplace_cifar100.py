import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
from rfm import LaplaceRFM
import logging
import wandb

# ================== Logging setup ====================
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

file_handler = logging.FileHandler("output.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ================== Device setup ====================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM_GB = torch.cuda.get_device_properties(DEVICE).total_memory // 1024**3 - 1
else:
    DEVICE = torch.device("cpu")
    DEV_MEM_GB = 8

# ================== Transform & Dataset ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
])

full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Use subset to limit memory
subset_size = 10000
train_subset, _ = random_split(full_train_dataset, [subset_size, len(full_train_dataset) - subset_size])

# ================== Feature extraction ====================
logger.info("Loading ResNet18 and extracting features...")
resnet = models.resnet18(pretrained=True).to(DEVICE)
resnet.eval()
feature_extractor = create_feature_extractor(resnet, return_nodes={"avgpool": "feat"})

def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_feats, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            feats = feature_extractor(imgs)["feat"].squeeze()  # [B, 512, 1, 1] -> [B, 512]
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    X = torch.cat(all_feats)
    y = F.one_hot(torch.cat(all_labels), num_classes=100).float()
    return TensorDataset(X, y)

train_feat_dataset = extract_features(train_subset)
test_feat_dataset = extract_features(test_dataset)

# ================== DataLoader for RFM ====================
batch_size = 32
train_loader = DataLoader(train_feat_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_feat_dataset, batch_size=batch_size, shuffle=False)

# ================== LaplaceRFM ====================
model = LaplaceRFM(
    bandwidth=1.0,
    device=DEVICE,
    mem_gb=DEV_MEM_GB,
    diag=False
)

# ================== WandB setup ====================
wandb.login(key='cf3dc9c85e2330a83d886a54b44d32768b2d7b60')  # Replace with your key
wandb.init(project="rfm-nmf", name="LaplaceRFM-CIFAR100-ResNet18",
           config={
               "batch_size": batch_size,
               "epochs": 3,
               "bandwidth": 5.0,
               "subset_size": subset_size,
               "device": str(DEVICE),
               "feature_extractor": "ResNet18-avgpool"
           })

# ================== Train ====================
logger.info("Training LaplaceRFM with ResNet18 features")

model.fit(
    train_data=train_loader,
    test_data=test_loader,
    iters=3,
    classification=True,
    total_points_to_sample=subset_size,
    M_batch_size=64,
    method='lstsq',
    verbose=True,
    epochs=3,
    bandwidth=5.0,  # Adjusted bandwidth for Laplace kernel
)

wandb.finish()
