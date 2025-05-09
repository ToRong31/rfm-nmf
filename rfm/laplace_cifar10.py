import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from rfm import LaplaceRFM
import torch.nn.functional as F
import logging
import wandb

# Cấu hình logging: ghi log vào file và in ra console
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

# File handler: ghi log vào output.txt
file_handler = logging.FileHandler("output.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler: in log ra console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Hàm chuyển đổi batch: chuyển đổi nhãn thành one-hot encoding với 10 lớp
def one_hot_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    # Chuyển đổi nhãn sang one-hot (float)
    labels = F.one_hot(labels, num_classes=10).float()
    
    # Đảm bảo rằng cả images và labels đều ở trên cùng một thiết bị
    if torch.cuda.is_available():
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
    
    return images, labels

# Thiết lập thiết bị và bộ nhớ GPU (nếu có)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM_GB = torch.cuda.get_device_properties(DEVICE).total_memory // 1024**3 - 1
else:
    DEVICE = torch.device("cpu")
    DEV_MEM_GB = 8

# Định nghĩa transform: chuyển đổi ảnh CIFAR-10 thành tensor và chuẩn hóa
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Chuẩn hóa ảnh RGB
    transforms.Lambda(lambda x: x.view(-1))  # Làm phẳng ảnh thành vector
])

# Tải dataset CIFAR-10 cho training và testing
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Chia tập train ra thành một phần nhỏ để tránh làm đầy bộ nhớ
subset_size = 10000  # Có thể điều chỉnh theo khả năng của máy
train_subset, _ = random_split(full_train_dataset, [subset_size, len(full_train_dataset) - subset_size])

# Tạo DataLoader cho tập train và test, sử dụng collate_fn để chuyển đổi nhãn
batch_size = 16
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=one_hot_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=one_hot_collate)

# Tham số cần điều chỉnh
laplace_model = LaplaceRFM(
    bandwidth=1.,  # Nên tune parameter này (thử giá trị 0.5-5)
    device=DEVICE,
    mem_gb=DEV_MEM_GB,
    diag=False
)

# Phần huấn luyện
# wandb.login(key='cf3dc9c85e2330a83d886a54b44d32768b2d7b60')
# wandb.init(project="rfm-nmf", name="LaplaceRFM-CIFAR10-NMF_new_100_lstsq")
laplace_model.max_lstsq_size = 100  # Kích thước tối đa cho lstsq
logger.info("Training LaplaceRFM")
laplace_model.fit(
    train_data=train_loader,
    test_data=test_loader,
    iters=50,  # Tham số này có thể conflict với epochs
    classification=True,
    total_points_to_sample=20000,  # Nên để None để dùng toàn bộ data
    M_batch_size=64,  # Tăng batch size để tận dụng GPU
    method='nmf',
    verbose=True,
    epochs=10,  # Nên tăng số epochs (10-50)
    prefit_nmf=True,  # Sử dụng NMF để khởi tạo W và H
)
# wandb.finish()