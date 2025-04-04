import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from rfm import LaplaceRFM, GeneralizedLaplaceRFM, GaussRFM, NTKModel
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

# Định nghĩa transform: chuyển đổi ảnh MNIST thành tensor và làm phẳng thành vector 784 chiều
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Tải dataset MNIST cho training và testing
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

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

# Phần huấn luyện nên sửa thành

wandb.login(key='cf3dc9c85e2330a83d886a54b44d32768b2d7b60')
wandb.init(project="rfm-nmf", name="LaplaceRFM-MNIST-lstsq", 
           config={
    "batch_size": batch_size,
    "epochs": 3,
    "bandwidth": 1.0,
    "subset_size": subset_size,
    "device": str(DEVICE),
})
logger.info("Training LaplaceRFM")
laplace_model.fit(
    train_data=train_loader,
    test_data=test_loader,
    iters=3,  # Tham số này có thể conflict với epochs
    classification=True,
    total_points_to_sample=subset_size, # Nên để None để dùng toàn bộ data
    M_batch_size=64,  # Tăng batch size để tận dụng GPU
    method='lstsq',
    verbose=True,
    epochs=3,  # Nên tăng số epochs (10-50)
)
wandb.finish()