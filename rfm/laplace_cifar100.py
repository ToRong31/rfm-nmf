import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from rfm import LaplaceRFM
import torch.nn.functional as F
import logging
import wandb

# Cấu hình logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

file_handler = logging.FileHandler("output.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Thiết lập thiết bị
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM_GB = torch.cuda.get_device_properties(DEVICE).total_memory // 1024**3 - 1
else:
    DEVICE = torch.device("cpu")
    DEV_MEM_GB = 8

# Transform cho CIFAR-100: Resize, normalize và flatten ảnh 32x32x3 => 3072
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten ảnh
])

# Collate function với 100 lớp (one-hot)
def one_hot_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    labels = F.one_hot(labels, num_classes=100).float()
    
    if torch.cuda.is_available():
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
    
    return images, labels

# Tải CIFAR-100
full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Chia subset để tránh dùng quá nhiều RAM/GPU
subset_size = 10000
train_subset, _ = random_split(full_train_dataset, [subset_size, len(full_train_dataset) - subset_size])

# DataLoader
batch_size = 16
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=one_hot_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=one_hot_collate)

# Cấu hình mô hình Laplace RFM
laplace_model = LaplaceRFM(
    bandwidth=1.0,
    device=DEVICE,
    mem_gb=DEV_MEM_GB,
    diag=False
)

# Đăng nhập và khởi tạo wandb
wandb.login(key='cf3dc9c85e2330a83d886a54b44d32768b2d7b60')  # Thay bằng key cá nhân nếu khác
wandb.init(project="rfm-nmf", name="LaplaceRFM-CIFAR100-nmf",
           config={
               "batch_size": batch_size,
               "epochs": 3,
               "bandwidth": 1.0,
               "subset_size": subset_size,
               "device": str(DEVICE),
               "dataset": "CIFAR100"
           })

logger.info("Training LaplaceRFM on CIFAR-100")

laplace_model.fit(
    train_data=train_loader,
    test_data=test_loader,
    iters=3,
    classification=True,
    total_points_to_sample=subset_size,
    M_batch_size=64,
    method='lstsq',
    verbose=True,
    epochs=3,
    bandwidth=3.0,
)

wandb.finish()
