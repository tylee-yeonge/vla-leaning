# VLA 학습 가이드 - Phase 2

## 목차
- [📅 Phase 2: Bottom-Up 기초 탄탄히 (3-6개월)](#-phase-2-bottom-up-기초-탄탄히-3-6개월)
- [Month 3-4: Deep Learning 제대로](#month-3-4-deep-learning-제대로)
  - [Week 1-2: PyTorch 심화](#week-1-2-pytorch-심화)
  - [Week 3-4: CNN 깊이 파기](#week-3-4-cnn-깊이-파기)
  - [Week 5-6: Computer Vision 핵심](#week-5-6-computer-vision-핵심)
- [Month 5: Transformer & Multi-modal](#month-5-transformer--multi-modal)
  - [Week 1: Attention 메커니즘 완전 정복](#week-1-attention-메커니즘-완전-정복)
  - [Week 2: Transformer Encoder & Decoder](#week-2-transformer-encoder--decoder)
  - [Week 3: Vision Transformer (ViT)](#week-3-vision-transformer-vit)
  - [Week 4-6: Multi-modal Learning](#week-4-6-multi-modal-learning)
- [Month 6: Imitation Learning & RL 기초](#month-6-imitation-learning--rl-기초)
  - [Week 1-2: Imitation Learning 심화](#week-1-2-imitation-learning-심화)
  - [Week 3-4: RL 기초 (최소한)](#week-3-4-rl-기초-최소한)
- [수학 기초 (Phase 2 전체 병행)](#수학-기초-phase-2-전체-병행)
- [Phase 2 완료 체크](#phase-2-완료-체크)

## 📅 Phase 2: Bottom-Up 기초 탄탄히 (3-6개월)

### 목표
- Phase 1에서 부족했던 부분 체계적으로 채우기
- 수학, 딥러닝, CV 기초 제대로
- 논문 읽을 수 있는 실력
- Imitation Learning & RL 기초

---

## Month 3-4: Deep Learning 제대로

### Week 1-2: PyTorch 심화

**Phase 1에서 기본만 → 이제 제대로**

#### Custom Dataset & DataLoader
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomRobotDataset(Dataset):
    """
    복잡한 전처리가 포함된 Dataset
    """
    def __init__(self, data_dir, transform=None, augmentation=None):
        self.data_dir = data_dir
        self.transform = transform
        self.augmentation = augmentation
        
        # 데이터 로드
        self.episodes = self.load_episodes(data_dir)
        
        # 통계 계산
        self.compute_statistics()
    
    def load_episodes(self, data_dir):
        # 복잡한 로딩 로직
        pass
    
    def compute_statistics(self):
        """
        정규화를 위한 통계 계산
        """
        all_actions = []
        for episode in self.episodes:
            all_actions.extend(episode['actions'])
        
        all_actions = np.array(all_actions)
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0)
    
    def __len__(self):
        return sum(len(ep['obs']) for ep in self.episodes)
    
    def __getitem__(self, idx):
        # Episode와 frame index 찾기
        episode_idx, frame_idx = self.get_episode_frame(idx)
        
        # 데이터 추출
        obs = self.episodes[episode_idx]['obs'][frame_idx]
        action = self.episodes[episode_idx]['actions'][frame_idx]
        
        # Augmentation (선택적)
        if self.augmentation:
            obs, action = self.augmentation(obs, action)
        
        # Transform
        if self.transform:
            obs = self.transform(obs)
        
        # Normalization
        action = (action - self.action_mean) / (self.action_std + 1e-8)
        
        return {
            'observation': torch.FloatTensor(obs),
            'action': torch.FloatTensor(action),
            'episode_idx': episode_idx,
            'frame_idx': frame_idx
        }

# 효율적인 DataLoader 설정
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # CPU 코어 활용
    pin_memory=True,      # GPU 전송 빠르게
    prefetch_factor=2,    # 미리 로드
    persistent_workers=True  # Worker 재사용
)
```

---

#### Custom Loss Functions
```python
class CustomLoss(nn.Module):
    """
    복잡한 커스텀 Loss
    """
    def __init__(self, position_weight=1.0, velocity_weight=0.5):
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
    
    def forward(self, pred, target, mask=None):
        # Position loss
        pos_loss = F.mse_loss(pred['position'], target['position'], reduction='none')
        
        # Velocity loss
        vel_loss = F.mse_loss(pred['velocity'], target['velocity'], reduction='none')
        
        # Combined
        loss = self.position_weight * pos_loss + self.velocity_weight * vel_loss
        
        # Masking (for variable length sequences)
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss

# Smooth L1 Loss (Huber Loss)
class SmoothL1Loss(nn.Module):
    """
    Outlier에 robust한 loss
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()
```

---

#### Learning Rate Scheduling
```python
# 1. Warmup + Cosine Annealing
from torch.optim.lr_scheduler import CosineAnnealingLR

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

# 2. OneCycleLR (추천!)
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=num_epochs,
    steps_per_epoch=len(dataloader),
    pct_start=0.3,  # Warmup 30%
    anneal_strategy='cos',
    div_factor=25,  # initial_lr = max_lr/25
    final_div_factor=1e4  # final_lr = initial_lr/1e4
)

# 사용
for epoch in range(num_epochs):
    for batch in dataloader:
        # Training step
        optimizer.step()
        scheduler.step()  # Batch마다 호출!
```

---

#### Gradient Clipping & Accumulation
```python
# Gradient Clipping
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # Gradient norm 제한
)

# Gradient Accumulation (큰 batch size 시뮬레이션)
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward
    output = model(batch)
    loss = criterion(output, batch['target'])
    
    # Normalize loss
    loss = loss / accumulation_steps
    
    # Backward
    loss.backward()
    
    # Update every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

#### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

# Scaler 초기화
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Mixed precision forward
        with autocast():
            output = model(batch['obs'])
            loss = criterion(output, batch['action'])
        
        # Scaled backward
        scaler.scale(loss).backward()
        
        # Unscale gradients & clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step
        scaler.step(optimizer)
        scaler.update()

# 효과: 메모리 50% 절감, 속도 1.5-2배
```

---

#### GPU 메모리 최적화
```python
# 1. Gradient Checkpointing
import torch.utils.checkpoint as checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = HeavyLayer()
        self.layer2 = HeavyLayer()
        self.layer3 = HeavyLayer()
    
    def forward(self, x):
        # Checkpoint으로 메모리 절약
        x = checkpoint.checkpoint(self.layer1, x)
        x = checkpoint.checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x

# 2. 메모리 프로파일링
import torch.cuda

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 상세 분석
print(torch.cuda.memory_summary())

# 3. 메모리 정리
torch.cuda.empty_cache()
```

---

**자료:**
- PyTorch 공식 튜토리얼 전체
- "Deep Learning with PyTorch" 책 (선택)

**시간: 주 5-7시간**

---

### Week 3-4: CNN 깊이 파기

**Phase 1: ViT만 사용 → 이제 CNN도**

#### CNN 기본 구조 이해
```python
import torch.nn as nn
import torch.nn.functional as F

# 1. Basic CNN Block
class BasicCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# 2. LeNet (기초)
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

---

#### ResNet 구현 (Skip Connection의 중요성)
```python
class ResidualBlock(nn.Module):
    """
    ResNet의 핵심: Skip Connection
    
    왜 중요한가?
    - Gradient vanishing 방지
    - 깊은 네트워크 학습 가능
    - Identity mapping 학습 용이
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # Skip connection!
        out = F.relu(out)
        
        return out

class ResNet18(nn.Module):
    """
    ResNet-18 구현
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
```

---

#### MobileNet (Efficient CNN)
```python
class DepthwiseSeparableConv(nn.Module):
    """
    MobileNet의 핵심: Depthwise Separable Convolution
    
    일반 Conv: 파라미터 = K × K × C_in × C_out
    DW Conv: 파라미터 = K × K × C_in + C_in × C_out
    
    → 파라미터 수 대폭 감소!
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Depthwise: 각 채널 독립적으로 convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels  # 핵심!
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise: 1x1 conv로 채널 믹싱
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNetV2Block(nn.Module):
    """
    MobileNetV2: Inverted Residual Block
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

---

#### EfficientNet 개념
```python
"""
EfficientNet의 핵심 아이디어:
1. Compound Scaling
   - Depth (레이어 수)
   - Width (채널 수)
   - Resolution (입력 크기)
   → 세 가지를 균형있게 조절!

2. Optimal scaling coefficients
   depth: d = α^φ
   width: w = β^φ
   resolution: r = γ^φ
   
   where α·β²·γ² ≈ 2

3. MBConv (Mobile Inverted Bottleneck)
   - Squeeze-and-Excitation
   - Swish activation
"""

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    채널 간 관계 모델링
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.squeeze(x).view(b, c)
        
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        
        # Scale
        return x * y.expand_as(x)
```

---

**자료:**
- CS231n Lecture 9 (CNN Architectures)
- "Dive into Deep Learning" Chapter 7-8
- Papers: ResNet, MobileNet, EfficientNet

**시간: 주 5-7시간**

---

### Week 5-6: Computer Vision 핵심

#### Image Classification
```python
# Transfer Learning 예시
import torchvision.models as models
from torchvision import transforms

# Pre-trained model 로드
model = models.resnet50(pretrained=True)

# 마지막 layer 교체 (Fine-tuning)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Feature extraction vs Fine-tuning
# 1. Feature extraction: CNN 부분 freeze
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True  # 마지막 layer만 학습

# 2. Fine-tuning: 전체 학습 (낮은 LR)
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

#### Object Detection 기초
```python
"""
Object Detection의 발전:

1. Two-stage detectors (Faster R-CNN)
   - Region Proposal → Classification
   - 느리지만 정확

2. One-stage detectors (YOLO, SSD)
   - 한 번에 예측
   - 빠르지만 정확도 낮음

3. Modern (EfficientDet, DETR)
   - Transformer 기반
   - 빠르고 정확
"""

# YOLO 사용 예시 (실전)
from ultralytics import YOLO

# Pre-trained model
model = YOLO('yolov8n.pt')

# Fine-tuning on custom data
model.train(
    data='custom_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='custom_detector'
)

# Inference
results = model('image.jpg')
boxes = results[0].boxes  # Bounding boxes
```

---

#### Semantic Segmentation
```python
class UNet(nn.Module):
    """
    U-Net: Semantic Segmentation의 기본
    
    구조:
    - Encoder (Contracting path): Feature 추출
    - Decoder (Expanding path): Upsampling
    - Skip connections: Detail 보존
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # 1024 = 512 + 512 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)
```

---

#### 미니 프로젝트: 물류 박스 탐지기
```python
# YOLOv8로 custom object detector 만들기

# 1. 데이터 준비
"""
dataset/
  images/
    train/
      img1.jpg
      img2.jpg
    val/
      img3.jpg
  labels/
    train/
      img1.txt  # YOLO format
      img2.txt
    val/
      img3.txt
"""

# 2. data.yaml 작성
"""
train: dataset/images/train
val: dataset/images/val

nc: 3  # number of classes
names: ['small_box', 'medium_box', 'large_box']
"""

# 3. Training
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='boxes.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='box_detector',
    patience=20,  # Early stopping
    save=True,
    plots=True
)

# 4. Evaluation
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# 5. Inference
results = model.predict('test_image.jpg', save=True)

# 6. Export for deployment
model.export(format='onnx')  # For faster inference
```

---

**자료:**
- CS231n Lecture 11 (Detection and Segmentation)
- YOLO papers and tutorials
- U-Net paper

**시간: 주 5-7시간**

---

### Month 3-4 완료 체크
```
✅ PyTorch 자유자재 (80%)
✅ CNN 아키텍처 이해 (70%)
✅ CV 주요 task 이해 (60%)
✅ Transfer Learning 경험
✅ Custom Object Detector 구현

→ Transformer 준비 완료!
```

---

## Month 5: Transformer & Multi-modal

### Week 1: Attention 메커니즘 완전 정복

#### Scaled Dot-Product Attention
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    왜 scaling (√d_k)?
    - QK^T의 값이 커지면 softmax가 saturation
    - Gradient vanishing 방지
    """
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch, num_heads, seq_len, d_k)
            K: (batch, num_heads, seq_len, d_k)
            V: (batch, num_heads, seq_len, d_v)
            mask: (batch, 1, seq_len, seq_len) or None
        
        Returns:
            output: (batch, num_heads, seq_len, d_v)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Attention scores: QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, L, L)
        
        # Scaling
        scores = scores / np.sqrt(self.d_k)
        
        # Masking (for decoder, padding, etc.)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)
        
        # Weighted sum of values
        output = torch.matmul(attention_weights, V)  # (B, H, L, d_v)
        
        return output, attention_weights

# 직관적 이해를 위한 예시
def attention_example():
    """
    Attention의 직관
    """
    # 예: "The cat sat on the mat"
    # Query: "sat"이 무엇을 주목해야 하는가?
    
    Q = torch.tensor([
        [0.1, 0.2, 0.9, 0.1, 0.2, 0.1]  # "sat"의 query
    ])
    
    K = torch.tensor([
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],  # "The"
        [0.1, 0.9, 0.1, 0.1, 0.1, 0.1],  # "cat"
        [0.1, 0.2, 0.9, 0.1, 0.2, 0.1],  # "sat"
        [0.1, 0.1, 0.1, 0.9, 0.1, 0.1],  # "on"
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.1],  # "the"
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],  # "mat"
    ])
    
    # Attention scores
    scores = Q @ K.T
    attention_weights = F.softmax(scores, dim=-1)
    
    print("Attention weights:")
    print("sat attends to:")
    for i, word in enumerate(["The", "cat", "sat", "on", "the", "mat"]):
        print(f"  {word}: {attention_weights[0, i]:.3f}")
    
    # Output: "sat"은 "cat"과 "mat"에 주목!
```

---

#### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    왜 여러 head?
    - 다양한 관점에서 attention
    - Head 1: 문법적 관계
    - Head 2: 의미적 관계
    - Head 3: 거리 정보
    - ...
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def split_heads(self, x, batch_size):
        """
        (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. Linear projections
        Q = self.W_q(Q)  # (B, L, d_model)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 2. Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (B, H, L, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # 3. Attention
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        output = output.transpose(1, 2).contiguous()  # (B, L, H, d_k)
        output = output.view(batch_size, -1, self.d_model)  # (B, L, d_model)
        
        # 5. Final linear
        output = self.W_o(output)
        
        return output, attention_weights

# 시각화
def visualize_attention():
    """
    Attention weights 시각화
    """
    import matplotlib.pyplot as plt
    
    model = MultiHeadAttention(d_model=512, num_heads=8)
    
    # Dummy input
    seq_len = 10
    x = torch.randn(1, seq_len, 512)
    
    output, attn_weights = model(x, x, x)
    
    # Plot attention for each head
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for head in range(8):
        ax = axes[head // 4, head % 4]
        attn = attn_weights[0, head].detach().numpy()
        im = ax.imshow(attn, cmap='hot', interpolation='nearest')
        ax.set_title(f'Head {head+1}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('attention_heads.png')
    plt.show()
```

---

#### Position Encoding
```python
class PositionalEncoding(nn.Module):
    """
    위치 정보 인코딩
    
    왜 필요?
    - Attention은 순서 정보 없음
    - 위치에 따른 고유한 패턴 부여
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# 시각화
def visualize_positional_encoding():
    import matplotlib.pyplot as plt
    
    pe = PositionalEncoding(d_model=128, max_len=100)
    
    # Extract PE matrix
    pos_enc = pe.pe[0].numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pos_enc.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding')
    plt.colorbar()
    plt.savefig('positional_encoding.png')
    plt.show()
```

---

**자료:**
- "Attention is All You Need" 논문
- "The Annotated Transformer" (Harvard NLP)
- CS224n Lecture 9

**시간: 주 8-10시간**

---

### Week 2: Transformer Encoder & Decoder

#### Transformer Encoder Layer
```python
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    
    구조:
    1. Multi-Head Self-Attention
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len) or None
        """
        # 1. Self-Attention + Residual + Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed-Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

---

#### Transformer Decoder Layer
```python
class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    
    구조:
    1. Masked Multi-Head Self-Attention
    2. Add & Norm
    3. Multi-Head Cross-Attention (with encoder output)
    4. Add & Norm
    5. Feed-Forward Network
    6. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch, tgt_len, d_model) - decoder input
            encoder_output: (batch, src_len, d_model)
            src_mask: mask for encoder output
            tgt_mask: causal mask for decoder (prevents looking ahead)
        """
        # 1. Masked Self-Attention
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. Cross-Attention (attend to encoder output)
        cross_attn_output, _ = self.cross_attn(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

def create_causal_mask(seq_len):
    """
    Causal mask for decoder
    - Prevents attending to future tokens
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

---

#### Complete Transformer
```python
class Transformer(nn.Module):
    """
    Complete Transformer for sequence-to-sequence
    """
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout
        )
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)
        """
        # Encode
        src_emb = self.dropout(self.pos_encoding(
            self.src_embedding(src) * np.sqrt(self.d_model)
        ))
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Decode
        tgt_emb = self.dropout(self.pos_encoding(
            self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        ))
        
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, encoder_output, src_mask, tgt_mask)
        
        # Output
        output = self.fc_out(dec_output)
        
        return output

# 간단한 번역 task로 테스트
def test_transformer():
    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    
    model = Transformer(src_vocab_size, tgt_vocab_size)
    
    # Dummy data
    src = torch.randint(0, src_vocab_size, (32, 20))  # (batch, src_len)
    tgt = torch.randint(0, tgt_vocab_size, (32, 15))  # (batch, tgt_len)
    
    # Masks
    tgt_mask = create_causal_mask(15)
    
    # Forward
    output = model(src, tgt, tgt_mask=tgt_mask)
    
    print(f"Output shape: {output.shape}")  # (32, 15, tgt_vocab_size)
```

---

**프로젝트: 간단한 번역 모델**
```python
# 영어 → 한국어 번역 (작은 데이터셋)

import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = [self.src_vocab.get(w, self.src_vocab['<unk>']) 
               for w in self.src_sentences[idx].split()]
        tgt = [self.tgt_vocab.get(w, self.tgt_vocab['<unk>']) 
               for w in self.tgt_sentences[idx].split()]
        
        return torch.LongTensor(src), torch.LongTensor(tgt)

# Training
def train_translation_model():
    model = Transformer(src_vocab_size, tgt_vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    for epoch in range(num_epochs):
        for src, tgt in dataloader:
            # Prepare target input and output
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]  # Remove first token
            
            # Create masks
            tgt_mask = create_causal_mask(tgt_input.size(1))
            
            # Forward
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            
            # Loss
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

**시간: 주 8-10시간**

---

### Week 3: Vision Transformer (ViT)
```python
class PatchEmbedding(nn.Module):
    """
    이미지 → Patch → Embedding
    
    예: 224x224 이미지, 16x16 patch
    → 196개 patch (14x14)
    → 각 patch를 768차원 벡터로
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolution으로 patch embedding
        # (3, 224, 224) → (768, 14, 14)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT)
    
    구조:
    1. Patch Embedding
    2. [CLS] token 추가
    3. Position Embedding
    4. Transformer Encoder
    5. Classification Head ([CLS] token 사용)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position Embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        
        # Dropout
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        d_ff = int(embed_dim * mlp_ratio)
        self.encoder = TransformerEncoder(
            num_layers=depth,
            d_model=embed_dim,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer Encoder
        x = self.encoder(x)
        
        # Classification (use [CLS] token)
        x = self.norm(x)
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)  # (B, num_classes)
        
        return logits

# 사용 예시
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# Pre-trained ViT 사용 (실전)
from transformers import ViTModel, ViTConfig

# Option 1: From config
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_channels=3,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12
)
model = ViTModel(config)

# Option 2: Pre-trained
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
```

---

**자료:**
- "An Image is Worth 16x16 Words" 논문
- ViT 공식 코드 분석

**시간: 주 8-10시간**

---

### Week 4-6: Multi-modal Learning

#### CLIP: Contrastive Language-Image Pre-training
```python
class CLIP(nn.Module):
    """
    CLIP: 이미지와 텍스트를 같은 공간에 매핑
    
    핵심 아이디어:
    - 매칭되는 (이미지, 텍스트) 쌍: 가깝게
    - 매칭 안 되는 쌍: 멀게
    """
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # Projection heads
        self.image_projection = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_projection = nn.Linear(text_encoder.output_dim, embed_dim)
        
        # Temperature parameter (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, texts):
        """
        Args:
            images: (B, 3, H, W)
            texts: (B, seq_len)
        Returns:
            logits_per_image: (B, B)
            logits_per_text: (B, B)
        """
        # Encode
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Project to common space
        image_embeds = self.image_projection(image_features)
        text_embeds = self.text_projection(text_features)
        
        # Normalize
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

def contrastive_loss(logits_per_image, logits_per_text):
    """
    Contrastive Loss (InfoNCE)
    
    목표: Diagonal을 maximize, off-diagonal을 minimize
    """
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size).to(logits_per_image.device)
    
    # Image to text
    loss_i = F.cross_entropy(logits_per_image, labels)
    
    # Text to image
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    # Total loss
    loss = (loss_i + loss_t) / 2
    
    return loss

# Training
def train_clip():
    model = CLIP(image_encoder, text_encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for images, texts in dataloader:
            # Forward
            logits_per_image, logits_per_text = model(images, texts)
            
            # Loss
            loss = contrastive_loss(logits_per_image, logits_per_text)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Zero-shot classification
def zero_shot_classify(model, image, class_names):
    """
    CLIP의 강력한 기능: Zero-shot classification
    """
    # Encode image
    image_features = model.image_encoder(image.unsqueeze(0))
    image_embeds = F.normalize(
        model.image_projection(image_features), dim=-1
    )
    
    # Encode class names
    text_prompts = [f"a photo of a {name}" for name in class_names]
    text_tokens = tokenizer(text_prompts)
    text_features = model.text_encoder(text_tokens)
    text_embeds = F.normalize(
        model.text_projection(text_features), dim=-1
    )
    
    # Compute similarity
    similarity = (image_embeds @ text_embeds.t()).squeeze(0)
    
    # Softmax
    probs = F.softmax(similarity, dim=0)
    
    return probs

# 사용 예시
class_names = ["dog", "cat", "bird", "car"]
probs = zero_shot_classify(model, image, class_names)

for name, prob in zip(class_names, probs):
    print(f"{name}: {prob:.2%}")
```

---

#### 최신 VLM: BLIP, LLaVA
```python
# BLIP 사용 예시
from transformers import BlipProcessor, BlipForConditionalGeneration

# Model & Processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Image Captioning
from PIL import Image

image = Image.open("example.jpg")
inputs = processor(image, return_tensors="pt")

# Generate caption
generated_ids = model.generate(**inputs, max_length=50)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print(f"Caption: {caption}")

# Visual Question Answering
question = "What is in the image?"
inputs = processor(image, question, return_tensors="pt")
generated_ids = model.generate(**inputs, max_length=50)
answer = processor.decode(generated_ids[0], skip_special_tokens=True)

print(f"Answer: {answer}")
```
```python
# LLaVA 사용 예시
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Prepare inputs
prompt = "USER: <image>\nWhat is shown in this image?\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(generated_ids[0], skip_special_tokens=True)

print(response)
```

---

**프로젝트: Mini CLIP**
```python
# 작은 이미지-텍스트 데이터셋으로 CLIP 학습

# Dataset
class ImageTextDataset(Dataset):
    def __init__(self, image_paths, captions, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        caption = self.captions[idx]
        tokens = tokenizer(caption, padding='max_length', max_length=77)
        
        return image, tokens

# Training
def train_mini_clip():
    # Models
    image_encoder = torchvision.models.resnet50(pretrained=True)
    image_encoder.fc = nn.Identity()  # Remove classification head
    
    text_encoder = SimpleTextEncoder()  # Or use BERT
    
    model = CLIP(image_encoder, text_encoder, embed_dim=512)
    
    # Dataset
    dataset = ImageTextDataset(image_paths, captions, transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(30):
        for images, texts in dataloader:
            logits_per_image, logits_per_text = model(images, texts)
            loss = contrastive_loss(logits_per_image, logits_per_text)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model
```

---

**자료:**
- CLIP 논문: "Learning Transferable Visual Models From Natural Language Supervision"
- BLIP 논문
- LLaVA 논문

**시간: 주 6-8시간**

---

## Month 6: Imitation Learning & RL 기초

### Week 1-2: Imitation Learning 심화

#### Distribution Shift 문제
```python
"""
Behavioral Cloning의 근본적 한계:

Expert trajectory: s0 → s1 → s2 → s3 (goal)

Learned policy가 s1에서 약간 벗어남:
s0 → s1' → ?

Expert data에는 s1'에서의 행동이 없음!
→ 모델이 어떻게 해야 할지 모름
→ 에러가 누적됨 (compounding error)

해결책: DAgger, Behavior Regularization
"""
```

---

#### DAgger (Dataset Aggregation)
```python
class DAgger:
    """
    Interactive Imitation Learning
    
    과정:
    1. BC로 policy 학습
    2. Policy로 rollout
    3. Expert가 correction 제공
    4. 새 데이터 추가
    5. 반복
    """
    
    def __init__(self, expert, learner, env):
        self.expert = expert
        self.learner = learner
        self.env = env
        self.dataset = []
    
    def collect_expert_data(self, num_episodes):
        """
        Expert demonstration 수집
        """
        for _ in range(num_episodes):
            episode = []
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.expert.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                episode.append((state, action))
                state = next_state
            
            self.dataset.extend(episode)
        
        print(f"Collected {len(self.dataset)} expert transitions")
    
    def train_iteration(self, num_epochs=10):
        """
        한 iteration의 DAgger
        """
        # 1. Train policy on current dataset
        self.learner.train(self.dataset, num_epochs)
        
        # 2. Rollout with learned policy
        new_data = []
        num_rollouts = 10
        
        for _ in range(num_rollouts):
            state = self.env.reset()
            done = False
            
            while not done:
                # Learner's action
                learner_action = self.learner.get_action(state)
                
                # But ask expert what to do
                expert_action = self.expert.get_action(state)
                
                # Save (state, expert_action)
                new_data.append((state, expert_action))
                
                # Execute learner's action
                state, _, done, _ = self.env.step(learner_action)
        
        # 3. Add to dataset
        self.dataset.extend(new_data)
        
        print(f"Added {len(new_data)} corrections")
        print(f"Total dataset size: {len(self.dataset)}")
    
    def run(self, num_iterations=10):
        """
        Complete DAgger training
        """
        # Initial expert data
        self.collect_expert_data(num_episodes=50)
        
        # DAgger iterations
        for i in range(num_iterations):
            print(f"\n=== DAgger Iteration {i+1}/{num_iterations} ===")
            self.train_iteration()
            
            # Evaluate
            success_rate = self.evaluate()
            print(f"Success rate: {success_rate:.2%}")
            
            if success_rate > 0.9:
                print("Converged!")
                break
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate learned policy
        """
        successes = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.learner.get_action(state)
                state, reward, done, info = self.env.step(action)
                
                if info.get('success'):
                    successes += 1
                    break
        
        return successes / num_episodes
```

---

#### Behavior Regularization
```python
class BehaviorRegularizedPolicy(nn.Module):
    """
    Policy를 reference policy에서 너무 멀어지지 않도록 제약
    
    Loss = BC_loss + β * KL(π || π_ref)
    """
    
    def __init__(self, policy, reference_policy, beta=0.1):
        super().__init__()
        self.policy = policy
        self.reference_policy = reference_policy
        self.beta = beta
        
        # Freeze reference policy
        for param in self.reference_policy.parameters():
            param.requires_grad = False
    
    def compute_loss(self, obs, actions):
        # Standard BC loss
        pred_actions = self.policy(obs)
        bc_loss = F.mse_loss(pred_actions, actions)
        
        # KL divergence with reference policy
        ref_actions = self.reference_policy(obs).detach()
        kl_loss = F.mse_loss(pred_actions, ref_actions)
        
        # Combined loss
        total_loss = bc_loss + self.beta * kl_loss
        
        return total_loss, bc_loss, kl_loss
    
    def train_step(self, dataloader, optimizer):
        self.policy.train()
        
        total_loss = 0
        total_bc = 0
        total_kl = 0
        
        for obs, actions in dataloader:
            loss, bc_loss, kl_loss = self.compute_loss(obs, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_bc += bc_loss.item()
            total_kl += kl_loss.item()
        
        n = len(dataloader)
        return total_loss/n, total_bc/n, total_kl/n
```

---

#### Inverse Reinforcement Learning (개념)
```python
"""
Inverse RL의 아이디어:

기존 IL: Expert의 action을 직접 모방
IRL: Expert의 목적(reward function)을 추론

예시:
- Expert가 왜 이 경로를 선택했을까?
- 어떤 reward를 최대화하려는 걸까?

장점:
- 더 robust한 generalization
- Transfer learning 용이

단점:
- 계산 비용 높음
- 구현 복잡

VLA에서의 활용:
- 직접 사용보다는 아이디어 차용
- Reward design에 insight
- Preference learning
"""

class MaximumEntropyIRL:
    """
    Maximum Entropy IRL (개념만)
    """
    
    def __init__(self, env):
        self.env = env
        self.reward = nn.Linear(state_dim, 1)  # Learned reward
    
    def infer_reward(self, expert_trajectories):
        """
        Expert가 maximize하는 reward 추론
        
        과정:
        1. 현재 reward로 optimal policy 계산
        2. Policy로 trajectories 생성
        3. Expert와 learned의 feature 분포 비교
        4. Reward 업데이트
        5. 반복
        """
        for iteration in range(num_iterations):
            # 1. Compute optimal policy under current reward
            policy = self.compute_optimal_policy(self.reward)
            
            # 2. Sample trajectories
            learned_trajs = self.sample_trajectories(policy)
            
            # 3. Compare feature distributions
            expert_features = self.compute_features(expert_trajectories)
            learned_features = self.compute_features(learned_trajs)
            
            # 4. Update reward (gradient ascent)
            feature_diff = expert_features - learned_features
            self.reward.weight += learning_rate * feature_diff
            
            print(f"Iteration {iteration+1}, Reward updated")
        
        return self.reward
```

---

**자료:**
- "A Reduction of Imitation Learning" (DAgger 논문)
- CS285 Lecture 2-3 (Imitation Learning)

**시간: 주 6-8시간**

---

### Week 3-4: RL 기초 (최소한)

#### MDP와 Policy Gradient
```python
"""
Markov Decision Process (MDP):
- State (s): 환경의 상태
- Action (a): Agent의 행동
- Reward (r): 즉각적 보상
- Transition (P): s, a → s'
- Policy (π): s → a

목표: Expected return 최대화
J(π) = E[Σ γ^t * r_t]

VLA를 RL 관점에서:
- State: Robot observation (이미지, proprio)
- Action: Robot control command
- Reward: Task success
- Policy: VLA model 자체!
"""
```

---

#### REINFORCE Algorithm
```python
class PolicyGradient:
    """
    REINFORCE: 가장 기본적인 Policy Gradient
    
    핵심 아이디어:
    ∇J(π) = E[∇log π(a|s) * R]
    
    → 높은 return을 받은 action의 확률 증가
    """
    
    def __init__(self, policy):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    def compute_returns(self, rewards, gamma=0.99):
        """
        Discounted returns 계산
        
        R_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        """
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        
        # Normalize (학습 안정화)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self, trajectory):
        """
        Policy gradient update
        """
        states, actions, rewards = trajectory
        
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Compute log probabilities
        log_probs = []
        for state, action in zip(states, actions):
            action_dist = self.policy(state)
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs)
        
        # Policy gradient loss
        loss = -(log_probs * returns).mean()
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, env, num_episodes=1000):
        """
        Complete training loop
        """
        for episode in range(num_episodes):
            # Collect trajectory
            trajectory = self.collect_trajectory(env)
            
            # Update policy
            loss = self.update(trajectory)
            
            # Log
            episode_return = sum(trajectory[2])
            print(f"Episode {episode+1}, Return: {episode_return:.2f}, Loss: {loss:.4f}")
    
    def collect_trajectory(self, env):
        """
        환경에서 trajectory 수집
        """
        states, actions, rewards = [], [], []
        
        state = env.reset()
        done = False
        
        while not done:
            # Sample action from policy
            action_dist = self.policy(torch.FloatTensor(state))
            action = action_dist.sample()
            
            # Execute
            next_state, reward, done, _ = env.step(action.numpy())
            
            # Record
            states.append(torch.FloatTensor(state))
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        return states, actions, rewards

# Gaussian Policy (for continuous actions)
class GaussianPolicy(nn.Module):
    """
    Continuous action을 위한 Gaussian policy
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        features = self.net(state)
        mean = self.mean(features)
        std = self.log_std.exp()
        
        return torch.distributions.Normal(mean, std)
```

---

#### PPO 개념만
```python
"""
Proximal Policy Optimization (PPO)

REINFORCE의 문제점:
- 학습 불안정 (step size 조절 어려움)
- Sample efficiency 낮음

PPO의 해결책:
1. Importance sampling으로 old policy 재사용
2. Clipping으로 policy update 제한
3. Multiple epochs 학습

핵심 아이디어:
- Policy를 크게 바꾸지 않으면서 개선
- Trust region 개념

Loss:
L = min(
    ratio * advantage,
    clip(ratio, 1-ε, 1+ε) * advantage
)

where ratio = π_new(a|s) / π_old(a|s)
"""

class PPO:
    """
    PPO 개념 코드 (간소화)
    실제 구현은 더 복잡
    """
    
    def __init__(self, policy, clip_epsilon=0.2):
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    def compute_loss(self, states, actions, old_log_probs, advantages):
        """
        PPO clipped objective
        """
        # New log probs
        action_dist = self.policy(states)
        new_log_probs = action_dist.log_prob(actions)
        
        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 
                           1 - self.clip_epsilon, 
                           1 + self.clip_epsilon) * advantages
        
        # Take minimum (pessimistic bound)
        loss = -torch.min(surr1, surr2).mean()
        
        return loss
    
    def update(self, rollouts, num_epochs=4):
        """
        PPO update with multiple epochs
        """
        states, actions, old_log_probs, advantages = rollouts
        
        for epoch in range(num_epochs):
            loss = self.compute_loss(states, actions, old_log_probs, advantages)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

"""
VLA에서 PPO 활용:
- BC로 초기 policy 학습
- PPO로 fine-tuning
- 예: RT-2-X가 이 방식

하지만:
- Phase 2에서는 BC만으로 충분
- RL은 "있으면 좋은" 정도
- Phase 3에서 선택적으로 적용
"""
```

---

**VLA + RL Fine-tuning 전략**
```python
class VLAwithRLFinetuning:
    """
    VLA를 RL로 fine-tuning하는 전략
    """
    
    def __init__(self, vla_model, env):
        self.vla = vla_model
        self.env = env
        
        # RL algorithm (PPO 또는 SAC)
        self.rl_optimizer = PPO(vla_model)
    
    def stage1_bc_pretraining(self, expert_data):
        """
        Stage 1: BC로 초기 policy 학습
        """
        print("Stage 1: BC Pre-training")
        
        for epoch in range(100):
            for obs, actions in expert_data:
                pred_actions = self.vla(obs)
                loss = F.mse_loss(pred_actions, actions)
                
                # Update
                loss.backward()
                # ...
        
        print("BC pre-training done!")
        print(f"Success rate: {self.evaluate():.2%}")
    
    def stage2_rl_finetuning(self, num_iterations=1000):
        """
        Stage 2: RL로 self-improvement
        """
        print("\nStage 2: RL Fine-tuning")
        
        for iteration in range(num_iterations):
            # Collect rollouts
            rollouts = self.collect_rollouts()
            
            # PPO update
            loss = self.rl_optimizer.update(rollouts)
            
            # Evaluate
            if iteration % 10 == 0:
                success_rate = self.evaluate()
                print(f"Iteration {iteration}, Success: {success_rate:.2%}")
        
        print("RL fine-tuning done!")
    
    def collect_rollouts(self):
        """
        환경과 interaction
        """
        # Collect trajectories using current policy
        # ...
        pass
    
    def evaluate(self):
        """
        Evaluation
        """
        # Test in environment
        # ...
        pass

# 사용 예시
"""
1. Expert data로 BC 학습 (안정적 baseline)
2. 환경에서 self-play로 RL fine-tuning
3. 성능 개선 (BC 70% → RL 85%)

주의:
- RL은 불안정할 수 있음
- BC baseline이 중요
- Task에 따라 효과 다름
"""
```

---

**자료:**
- Spinning Up in Deep RL (OpenAI)
- CS285 Lecture 4-5 (Policy Gradient)

**시간: 주 6-8시간**

---

## 수학 기초 (Phase 2 전체 병행)

### Linear Algebra

**진행 중인 3Blue1Brown 시리즈 완주**

추가 학습:
- Gilbert Strang의 Linear Algebra (MIT OCW)
- 핵심 주제:
  * Eigenvalues/Eigenvectors (PCA, spectral methods)
  * SVD (데이터 압축, 추천 시스템)
  * Matrix decomposition

**시간: 주 2-3시간**

---

### Probability & Statistics

VLA에 필요한 확률론:
- 확률 분포 (Gaussian, Categorical)
- 기댓값, 분산
- Bayes' theorem
- Maximum Likelihood Estimation

자료:
- "Probability for Machine Learning" (Chris Bishop)
- Khan Academy Statistics

**시간: 주 2-3시간**

---

## Phase 2 완료 체크
```
✅ Deep Learning 기초 탄탄 (80%)
✅ PyTorch 자유자재 (85%)
✅ CNN 완전 이해 (80%)
✅ Transformer 완전 이해 (90%)
✅ Multi-modal learning 이해 (80%)
✅ Imitation Learning 심화 (70%)
✅ RL 기초 이해 (60%)
✅ 수학 기초 충분 (70%)
✅ 논문 읽기 수월함

→ 본격 VLA 프로젝트 준비 완료!
```