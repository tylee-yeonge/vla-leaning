# VLA Top-Down 학습 가이드

## 🎯 학습 철학

### Top-Down vs Bottom-Up
```
❌ Bottom-Up (전통적, 비효율):
선형대수 → 미적분 → 최적화 → ML → DL → CV → NLP → RL → VLA
└─ 6개월+ 지나도 VLA 시작 못 함

✅ Top-Down (효율적, 추천):
VLA 논문 읽기 → 막히는 부분만 학습 → 구현 → 반복
└─ 1주일 만에 VLA 코드 돌림
```

**핵심 원칙:**
1. 큰 그림을 먼저 본다
2. 이해도 30%면 진행한다
3. 필요한 것만 깊게 판다
4. 실습이 80%다

---

## 📅 8주 학습 로드맵

### Week 1: VLA의 전체 그림 보기

#### Day 1-2: RT-1 논문 첫 읽기

**목표:** 70% 이해 불가능해도 상관없음, "VLA가 뭔지" 감만 잡기

**논문:**
- "RT-1: Robotics Transformer for Real-World Control at Scale"
- [arXiv 링크](https://arxiv.org/abs/2212.06817)

**읽는 방법:**
1. Abstract만 3번 읽기
2. Introduction 정독
3. Figure 전부 보기 (그림이 핵심!)
4. Method 훑어보기 (모르는 거 메모만)
5. Results 결과만 보기

**메모할 것:**
- [ ] 이해 안 되는 용어 리스트업
- [ ] 궁금한 점 적기
- [ ] 그림 보면서 직관 잡기

**시간:** 3-4시간

**예상 반응:**
```
"Transformer가 뭐지?"
"Token이 뭐야?"
"FiLM layer는?"
"Behavioral Cloning?"

→ 정상입니다! 계속 진행하세요
```

---

#### Day 3-4: 영상으로 직관 잡기

**추천 영상:**

1. **RT-1 Official Video**
   - YouTube 검색: "RT-1 Robotics Transformer"
   - 로봇이 실제로 뭘 하는지 확인
   - 시간: 3분

2. **설명 영상**
   - "RT-1 Explained" 검색
   - 여러 개 보기 (각자 설명 방식 다름)
   - 시간: 30분-1시간

3. **Conference Talk**
   - CoRL 2022 presentation 검색
   - 연구자가 직접 설명
   - 시간: 20분

**학습 포인트:**
- [ ] VLA의 입력/출력 이해
- [ ] 어떤 문제를 푸는지 이해
- [ ] 왜 중요한지 이해

**시간:** 2-3시간

---

#### Day 5-7: HuggingFace LeRobot 실행해보기

**목표:** 코드 한 줄도 이해 못 해도 일단 돌려보기!

**설치:**
```bash
pip install lerobot
```

**첫 실행 - 데이터 시각화:**
```bash
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht
```
→ 데이터가 어떻게 생겼는지 확인

**두 번째 - 학습 실행:**
```bash
python -m lerobot.scripts.train \
    --dataset lerobot/pusht \
    --policy act
```
→ 학습 돌려보기 (이해 안 돼도 됨)

**관찰할 것:**
- [ ] 뭐가 입력인가? (이미지)
- [ ] 뭐가 출력인가? (action)
- [ ] 학습이 뭘 하는가? (loss 감소)
- [ ] 결과가 뭔가? (로봇 움직임)

**시간:** 4-6시간

---

**Week 1 완료 체크:**
```
✅ VLA가 대충 뭔지 앎
✅ 논문 1편 봤음 (이해 30%)
✅ 코드 돌려봤음 (이해 10%)
❌ 깊은 이해는 없음

→ 이게 정상! 계속 진행
```

---

### Week 2-3: 막히는 부분만 집중 학습

#### Part 1: Transformer 최소한만 (3-4일)

**목표:** VLA에 필요한 만큼만

**반드시 알아야 할 것:**

1. **Self-Attention 개념 (2시간)**
   - Query, Key, Value가 뭔지
   - Attention weight 계산
   - 직관: "어디를 볼지 결정"

2. **Transformer 구조 (2시간)**
   - Encoder-Decoder 구조
   - Position encoding
   - Multi-head attention

3. **Vision Transformer (2시간)**
   - 이미지를 어떻게 토큰으로?
   - Patch embedding
   - [CLS] token의 의미

**학습 자료:**
- "The Illustrated Transformer" (블로그) - 필독!
- 3Blue1Brown "Attention" 영상
- Andrej Karpathy "Let's build GPT" (선택)

**깊이 조절:**
- ❌ 수식 유도 → 너무 깊음
- ❌ 완벽한 이해 → 시간 낭비
- ✅ 개념과 직관 → 충분함

**간단한 실습:**
```python
import torch
import torch.nn.functional as F

# Attention 메커니즘 이해하기
Q = torch.randn(1, 10, 64)  # Query
K = torch.randn(1, 10, 64)  # Key
V = torch.randn(1, 10, 64)  # Value

# Attention weight 계산
attention_scores = Q @ K.transpose(-2, -1) / 8
attention_weights = F.softmax(attention_scores, dim=-1)

# Weighted sum
output = attention_weights @ V

print("Attention weights shape:", attention_weights.shape)
print("Output shape:", output.shape)

# 직관: attention_weights가 "어디를 볼지" 결정
```

**체크리스트:**
- [ ] Attention의 직관 이해
- [ ] Transformer 구조 그림 그릴 수 있음
- [ ] ViT가 어떻게 이미지 처리하는지 이해

---

#### Part 2: Behavioral Cloning (2-3일)

**목표:** VLA의 학습 방법 이해

**핵심 개념:**
```
BC = Supervised Learning for Actions

전통적 Supervised Learning:
입력: 이미지
출력: 라벨 (고양이/강아지)
손실: Cross-entropy

Behavioral Cloning:
입력: Observation (이미지)
출력: Action (joint angles, gripper)
손실: MSE (Mean Squared Error)

→ 완전히 똑같은 원리!
```

**코드로 이해:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# BC는 그냥 회귀(Regression)
class SimpleBC(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs):
        return self.net(obs)

# 학습 루프
model = SimpleBC(obs_dim=100, action_dim=7)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for obs, expert_action in dataloader:
    # Forward
    predicted_action = model(obs)
    
    # Loss (MSE)
    loss = F.mse_loss(predicted_action, expert_action)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")

# 이게 BC의 전부!
```

**BC vs RL 비교:**
```
Behavioral Cloning:
✅ 간단함 (Supervised Learning)
✅ 안정적 학습
✅ Expert demonstration 필요
❌ Expert보다 못함

Reinforcement Learning:
✅ Expert 없이 학습 가능
✅ 스스로 더 나아질 수 있음
❌ 복잡함
❌ 학습 불안정

VLA는 대부분 BC 사용!
```

**깊이 조절:**
- ❌ RL 전체 → 당장 불필요
- ❌ Policy gradient → 나중에
- ❌ DAgger → 심화
- ✅ BC만 → 지금은 충분!

**체크리스트:**
- [ ] BC의 원리 완전 이해
- [ ] BC와 일반 SL의 차이 이해
- [ ] 간단한 BC 코드 작성 가능

---

#### Part 3: PyTorch 필수만 (3-4일)

**목표:** VLA 코드 읽을 수 있을 정도만

**필수 개념:**

**1. Tensor 기본 (1시간)**
```python
import torch

# Tensor 생성
x = torch.randn(2, 3)
y = torch.zeros(2, 3)
z = torch.ones(2, 3)

# Shape 조작
x.shape  # torch.Size([2, 3])
x.view(3, 2)  # reshape
x.transpose(0, 1)

# 중요한 것만 알면 됨!
```

**2. nn.Module (1시간)**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 이 패턴만 알면 됨
```

**3. DataLoader (1시간)**
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset(my_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Batch 개념만 이해하면 됨
```

**4. Training Loop (1시간)**
```python
# 이 패턴이 전부!
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        # Forward
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**5. GPU 사용 (30분)**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
data = data.to(device)

# 이것만 알면 됨
```

**학습 자료:**
- PyTorch 공식 튜토리얼: "Deep Learning with PyTorch: A 60 Minute Blitz"
- https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

**깊이 조절:**
- ❌ Advanced 기능 → 나중에
- ❌ Custom CUDA → 필요 없음
- ❌ 분산 학습 → 필요 없음
- ✅ 기본만 → 충분함!

**체크리스트:**
- [ ] Tensor 조작 가능
- [ ] nn.Module 작성 가능
- [ ] Training loop 이해
- [ ] GPU 사용 가능

---

**Week 2-3 완료 체크:**
```
✅ Transformer 개념 이해 (60%)
✅ BC 완전 이해 (90%)
✅ PyTorch 기본 가능 (70%)
✅ VLA 논문 다시 읽으면 80% 이해

→ 이제 직접 만들 준비 됨!
```

---

### Week 4-6: 간단한 VLA 직접 만들기

#### 프로젝트: "Mini VLA"

**목표:** RT-1의 초간단 버전 구현

**스펙:**
- 환경: PyBullet (Isaac Sim보다 쉬움)
- 작업: 로봇 팔로 블록 밀기
- 모델: ViT-Tiny + MLP

---

#### Week 4: 환경 셋업

**Day 1-2: PyBullet 설치 및 기본**
```bash
pip install pybullet
```
```python
# basic_pybullet.py
import pybullet as p
import pybullet_data
import time

# 연결
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# 환경 로드
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
table = p.loadURDF("table/table.urdf", [0.5, 0, 0])
block = p.loadURDF("cube.urdf", [0.7, 0, 0.7], globalScaling=0.05)

# 시뮬레이션 실행
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

**체크:**
- [ ] PyBullet GUI 열림
- [ ] 로봇 팔 보임
- [ ] 블록 떨어지는 것 확인

---

**Day 3-4: 카메라 추가**
```python
# camera_setup.py
import numpy as np
from PIL import Image

def get_camera_image():
    # 카메라 설정
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1, 1, 1],
        cameraTargetPosition=[0.5, 0, 0.5],
        cameraUpVector=[0, 0, 1]
    )
    
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=1.0,
        nearVal=0.1,
        farVal=100.0
    )
    
    # 이미지 캡처
    width, height = 224, 224
    img_arr = p.getCameraImage(
        width, height,
        view_matrix,
        projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # RGB 이미지 추출
    rgb = np.array(img_arr[2]).reshape(height, width, 4)[:, :, :3]
    return rgb

# 테스트
for i in range(100):
    p.stepSimulation()
    
    if i % 10 == 0:
        img = get_camera_image()
        print(f"Image shape: {img.shape}")
```

**체크:**
- [ ] 이미지 캡처 가능
- [ ] Shape (224, 224, 3) 확인

---

**Day 5-7: Teleoperation 데이터 수집**
```python
# teleop_collect.py
import pybullet as p
import numpy as np
import pickle
from pynput import keyboard

class DataCollector:
    def __init__(self):
        self.episodes = []
        self.current_episode = {'obs': [], 'actions': []}
        self.recording = False
        self.current_action = np.zeros(7)  # 7-DOF robot
    
    def start_episode(self):
        self.recording = True
        self.current_episode = {'obs': [], 'actions': []}
        print("🔴 Recording started")
    
    def stop_episode(self):
        if self.recording:
            self.episodes.append(self.current_episode.copy())
            self.recording = False
            print(f"✅ Episode saved ({len(self.current_episode['obs'])} frames)")
    
    def add_step(self, obs, action):
        if self.recording:
            self.current_episode['obs'].append(obs)
            self.current_episode['actions'].append(action.copy())
    
    def save(self, filename='data.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.episodes, f)
        print(f"💾 Saved {len(self.episodes)} episodes")

# 키보드 제어
collector = DataCollector()
current_joint_velocities = np.zeros(7)

def on_press(key):
    global current_joint_velocities
    try:
        # 조인트 제어 (간단한 예시)
        if key.char == 'w':
            current_joint_velocities[0] = 0.5
        elif key.char == 's':
            current_joint_velocities[0] = -0.5
        # ... 다른 키 매핑
        
        # 녹화 제어
        elif key.char == 'r':
            collector.start_episode()
        elif key.char == 't':
            collector.stop_episode()
    except AttributeError:
        pass

def on_release(key):
    global current_joint_velocities
    current_joint_velocities = np.zeros(7)
    if key == keyboard.Key.esc:
        return False

# 메인 루프
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while listener.is_alive():
    # 로봇 제어
    for i in range(7):
        p.setJointMotorControl2(
            robot, i,
            p.VELOCITY_CONTROL,
            targetVelocity=current_joint_velocities[i]
        )
    
    p.stepSimulation()
    
    # 데이터 수집
    obs = get_camera_image()
    collector.add_step(obs, current_joint_velocities)

collector.save('demonstrations.pkl')
```

**목표:**
- [ ] 10+ 에피소드 수집
- [ ] 각 에피소드 50+ 프레임

---

#### Week 5: Mini VLA 모델 구현

**Day 1-3: 모델 정의**
```python
# mini_vla.py
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class MiniVLA(nn.Module):
    """
    초간단 VLA 모델
    - Vision: Pre-trained ViT
    - Policy: MLP
    """
    
    def __init__(self, action_dim=7):
        super().__init__()
        
        # Vision Encoder (ViT-Tiny 사용)
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=192,  # Tiny
            num_hidden_layers=12,
            num_attention_heads=3,
        )
        self.vision = ViTModel(config)
        
        # 또는 Pre-trained 사용
        # self.vision = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Policy Head
        self.policy = nn.Sequential(
            nn.Linear(192, 256),  # ViT hidden_size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, 224, 224)
        Returns:
            actions: (B, action_dim)
        """
        # Vision encoding
        vision_outputs = self.vision(images)
        
        # [CLS] token 사용
        image_features = vision_outputs.last_hidden_state[:, 0]
        
        # Policy
        actions = self.policy(image_features)
        
        return actions

# 테스트
if __name__ == '__main__':
    model = MiniVLA(action_dim=7)
    
    # 더미 입력
    dummy_images = torch.randn(4, 3, 224, 224)
    
    # Forward
    actions = model(dummy_images)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {actions.shape}")
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
```

**체크:**
- [ ] 모델 정의 완료
- [ ] Forward pass 작동
- [ ] 출력 shape 확인

---

**Day 4-5: 데이터 준비**
```python
# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from torchvision import transforms

class RobotDataset(Dataset):
    def __init__(self, episodes_file):
        with open(episodes_file, 'rb') as f:
            self.episodes = pickle.load(f)
        
        # 모든 (obs, action) 쌍 추출
        self.data = []
        for episode in self.episodes:
            for obs, action in zip(episode['obs'], episode['actions']):
                self.data.append((obs, action))
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        obs, action = self.data[idx]
        
        # 이미지 전처리
        obs_tensor = self.transform(obs)
        action_tensor = torch.FloatTensor(action)
        
        return obs_tensor, action_tensor

# 사용 예시
if __name__ == '__main__':
    dataset = RobotDataset('demonstrations.pkl')
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    # 테스트
    for obs, actions in dataloader:
        print(f"Batch obs shape: {obs.shape}")
        print(f"Batch actions shape: {actions.shape}")
        break
```

**체크:**
- [ ] Dataset 클래스 작동
- [ ] DataLoader로 배치 로드 가능
- [ ] 이미지 정규화 확인

---

**Day 6-7: 학습 루프**
```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from mini_vla import MiniVLA
from dataset import RobotDataset
import wandb  # 선택사항

def train_mini_vla():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 로드
    full_dataset = RobotDataset('demonstrations.pkl')
    
    # Train/Val split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    
    # 모델
    model = MiniVLA(action_dim=7).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 학습
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for obs, actions in train_loader:
            obs = obs.to(device)
            actions = actions.to(device)
            
            # Forward
            pred_actions = model(obs)
            loss = criterion(pred_actions, actions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for obs, actions in val_loader:
                obs = obs.to(device)
                actions = actions.to(device)
                
                pred_actions = model(obs)
                loss = criterion(pred_actions, actions)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✅ Best model saved!")
    
    print("\n🎉 Training complete!")

if __name__ == '__main__':
    train_mini_vla()
```

**실행:**
```bash
python train.py
```

**체크:**
- [ ] 학습 시작됨
- [ ] Loss가 감소함
- [ ] best_model.pt 저장됨

---

#### Week 6: 평가 및 개선

**Day 1-3: 시뮬레이션에서 평가**
```python
# evaluate.py
import torch
import pybullet as p
import pybullet_data
import numpy as np
from mini_vla import MiniVLA
from camera_setup import get_camera_image
from torchvision import transforms

def evaluate_model():
    # 디바이스
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = MiniVLA(action_dim=7).to(device)
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # PyBullet 초기화
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    
    success_count = 0
    num_episodes = 10
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # 환경 리셋
        p.resetSimulation()
        plane = p.loadURDF("plane.urdf")
        robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        block = p.loadURDF("cube.urdf", [0.7, 0, 0.7], globalScaling=0.05)
        
        # 목표 위치 (예시)
        goal_pos = [0.3, 0, 0.5]
        
        for step in range(100):
            # 관측
            rgb_image = get_camera_image()
            obs_tensor = transform(rgb_image).unsqueeze(0).to(device)
            
            # 예측
            with torch.no_grad():
                action = model(obs_tensor)
                action = action.cpu().numpy()[0]
            
            # 실행
            for i in range(7):
                p.setJointMotorControl2(
                    robot, i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=action[i]
                )
            
            p.stepSimulation()
            
            # 성공 체크 (블록이 목표 위치 근처에 있는지)
            block_pos, _ = p.getBasePositionAndOrientation(block)
            distance = np.linalg.norm(np.array(block_pos) - np.array(goal_pos))
            
            if distance < 0.1:  # 10cm 이내
                print(f"  ✅ Success at step {step}")
                success_count += 1
                break
        else:
            print(f"  ❌ Failed")
    
    p.disconnect()
    
    success_rate = success_count / num_episodes
    print(f"\n📊 Success Rate: {success_rate * 100:.1f}%")

if __name__ == '__main__':
    evaluate_model()
```

**실행:**
```bash
python evaluate.py
```

**기대 결과:**
- 첫 시도: 0-20% 성공률 (정상!)
- 데이터/모델 개선 후: 40-60%

---

**Day 4-7: 실패 원인 분석 & 개선**

**분석 체크리스트:**

1. **데이터 문제?**
   - [ ] 에피소드 수 충분한가? (최소 20+)
   - [ ] 다양한 시작 위치에서 수집했나?
   - [ ] Expert demonstration 품질은?
   
   개선:
```python
   # 더 많은 데이터 수집
   # 다양한 초기 조건에서
   # 더 일관된 demonstration
```

2. **모델 문제?**
   - [ ] 모델이 너무 작은가?
   - [ ] Overfitting 발생?
   - [ ] Loss가 충분히 낮아졌나?
   
   개선:
```python
   # 모델 크기 증가
   self.policy = nn.Sequential(
       nn.Linear(192, 512),  # 256 → 512
       nn.ReLU(),
       nn.Linear(512, 256),
       nn.ReLU(),
       nn.Linear(256, action_dim)
   )
```

3. **Sim-to-Sim Gap?**
   - [ ] 학습 환경 = 평가 환경?
   - [ ] 초기 조건 다른가?
   
   개선:
```python
   # 학습 시 다양한 초기 조건
   # Domain randomization
```

**개선 사이클:**
```
분석 → 가설 → 실험 → 평가 → 반복
```

---

**Week 4-6 완료 체크:**
```
✅ 나만의 VLA 모델 완성!
✅ 데이터 수집 → 학습 → 평가 전체 경험
✅ 실패도 경험 (매우 중요!)
✅ VLA 전체 파이프라인 이해
✅ 디버깅 능력 향상

→ 이제 복잡한 것도 할 수 있음!
```

---

### Week 7-8: RT-1 다시 읽고 LeRobot 마스터

#### Week 7: RT-1 논문 재구현

**Day 1-2: 논문 다시 읽기**

**이제 논문이 다르게 보입니다:**
```
Week 1에 읽었을 때:
"뭔 소린지 하나도 모르겠네..."

Week 7에 다시 읽으면:
"아! 이게 이런 뜻이었구나!"
"내가 만든 거랑 비슷한데?"
"여기가 다르네, 이래서 성능이 좋구나"

→ 같은 논문, 완전히 다른 이해도!
```

**주목할 부분:**
- [ ] FiLM layers: 언어로 vision conditioning
- [ ] TokenLearner: 효율적인 attention
- [ ] Action chunking: 여러 timestep 예측
- [ ] EfficientNet: Vision backbone

---

**Day 3-7: RT-1 핵심 요소 구현**
```python
# rt1_components.py

import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation
    언어 임베딩으로 vision feature를 조절
    """
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.scale = nn.Linear(condition_dim, feature_dim)
        self.shift = nn.Linear(condition_dim, feature_dim)
    
    def forward(self, features, condition):
        """
        Args:
            features: (B, N, feature_dim) - vision tokens
            condition: (B, condition_dim) - language embedding
        """
        gamma = self.scale(condition).unsqueeze(1)  # (B, 1, feature_dim)
        beta = self.shift(condition).unsqueeze(1)
        
        return gamma * features + beta

class TokenLearner(nn.Module):
    """
    Adaptive token selection
    많은 token → 적은 token (효율성)
    """
    def __init__(self, num_tokens, input_dim):
        super().__init__()
        self.num_tokens = num_tokens
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, num_tokens)
        )
    
    def forward(self, tokens):
        """
        Args:
            tokens: (B, N, D) - input tokens
        Returns:
            selected: (B, num_tokens, D)
        """
        # Attention weights
        attn_weights = self.attention(tokens)  # (B, N, num_tokens)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        attn_weights = attn_weights.transpose(1, 2)  # (B, num_tokens, N)
        selected = torch.bmm(attn_weights, tokens)  # (B, num_tokens, D)
        
        return selected

class SimpleRT1(nn.Module):
    """
    RT-1의 간소화 버전
    """
    def __init__(self, action_dim=7, num_tokens=8):
        super().__init__()
        
        # Vision Encoder
        from transformers import ViTModel
        self.vision = ViTModel.from_pretrained('google/vit-base-patch16-224')
        vision_dim = 768
        
        # Language Encoder (간단히)
        self.language = nn.Embedding(vocab_size=1000, embedding_dim=512)
        
        # FiLM layer
        self.film = FiLMLayer(vision_dim, 512)
        
        # Token Learner
        self.token_learner = TokenLearner(num_tokens, vision_dim)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=vision_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # Action Head
        self.action_head = nn.Linear(vision_dim, action_dim)
    
    def forward(self, images, instructions):
        """
        Args:
            images: (B, 3, 224, 224)
            instructions: (B, seq_len) - token ids
        Returns:
            actions: (B, action_dim)
        """
        # Vision
        vision_out = self.vision(images).last_hidden_state  # (B, 197, 768)
        
        # Language
        lang_embed = self.language(instructions).mean(dim=1)  # (B, 512)
        
        # FiLM conditioning
        conditioned = self.film(vision_out, lang_embed)  # (B, 197, 768)
        
        # Token selection
        selected_tokens = self.token_learner(conditioned)  # (B, 8, 768)
        
        # Decoder (간단히)
        query = selected_tokens.mean(dim=1, keepdim=True)  # (B, 1, 768)
        decoded = self.decoder(
            query.transpose(0, 1),
            selected_tokens.transpose(0, 1)
        ).transpose(0, 1).squeeze(1)
        
        # Action
        actions = self.action_head(decoded)  # (B, action_dim)
        
        return actions

# 테스트
if __name__ == '__main__':
    model = SimpleRT1()
    
    images = torch.randn(2, 3, 224, 224)
    instructions = torch.randint(0, 1000, (2, 10))
    
    actions = model(images, instructions)
    print(f"Actions shape: {actions.shape}")
```

**체크:**
- [ ] FiLM layer 이해 및 구현
- [ ] TokenLearner 구현
- [ ] 전체 모델 조립

---

#### Week 8: LeRobot 마스터

**Day 1-2: LeRobot 코드 완전 분석**
```bash
# LeRobot 저장소 클론
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 구조 파악
tree -L 2 lerobot/

# 핵심 파일들:
# - lerobot/common/policies/  # 다양한 policy 구현
# - lerobot/common/datasets/  # 데이터셋
# - lerobot/scripts/         # 학습/평가 스크립트
```

**분석할 파일:**
1. `lerobot/common/policies/act/modeling_act.py`
   - ACT (Action Chunking Transformer) 구현
   
2. `lerobot/common/policies/diffusion/modeling_diffusion.py`
   - Diffusion Policy 구현
   
3. `lerobot/common/datasets/lerobot_dataset.py`
   - 데이터 포맷 이해

**체크:**
- [ ] ACT 아키텍처 이해
- [ ] Diffusion Policy 개념 이해
- [ ] LeRobot 데이터 포맷 이해

---

**Day 3-5: 다양한 Policy 실험**
```bash
# 1. ACT Policy
python -m lerobot.scripts.train \
    --dataset lerobot/pusht \
    --policy act \
    --batch-size 32 \
    --num-epochs 100

# 2. Diffusion Policy
python -m lerobot.scripts.train \
    --dataset lerobot/pusht \
    --policy diffusion \
    --batch-size 32 \
    --num-epochs 100

# 3. 성능 비교
python -m lerobot.scripts.eval \
    --policy act \
    --checkpoint path/to/checkpoint

python -m lerobot.scripts.eval \
    --policy diffusion \
    --checkpoint path/to/checkpoint
```

**비교할 것:**
- [ ] 학습 속도
- [ ] 최종 성능
- [ ] 안정성
- [ ] 메모리 사용량

---

**Day 6-7: 자신의 데이터로 적용**
```python
# convert_to_lerobot.py
"""
Mini VLA 데이터 → LeRobot 형식 변환
"""

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from datasets import Dataset, Features, Image, Sequence, Value
import pickle
import numpy as np
from PIL import Image as PILImage

def convert_demonstrations():
    # 기존 데이터 로드
    with open('demonstrations.pkl', 'rb') as f:
        episodes = pickle.load(f)
    
    # LeRobot 형식으로 변환
    data_dict = {
        'observation.image': [],
        'action': [],
        'episode_index': [],
        'frame_index': [],
        'timestamp': [],
    }
    
    for ep_idx, episode in enumerate(episodes):
        for frame_idx, (obs, action) in enumerate(
            zip(episode['obs'], episode['actions'])
        ):
            # 이미지를 PIL Image로
            img = PILImage.fromarray(obs)
            
            data_dict['observation.image'].append(img)
            data_dict['action'].append(action.tolist())
            data_dict['episode_index'].append(ep_idx)
            data_dict['frame_index'].append(frame_idx)
            data_dict['timestamp'].append(frame_idx * 0.1)  # 예시
    
    # HuggingFace Dataset 생성
    features = Features({
        'observation.image': Image(),
        'action': Sequence(Value('float32'), length=7),
        'episode_index': Value('int64'),
        'frame_index': Value('int64'),
        'timestamp': Value('float32'),
    })
    
    dataset = Dataset.from_dict(data_dict, features=features)
    
    # 저장
    dataset.save_to_disk('./my_robot_dataset')
    
    print(f"✅ Converted {len(episodes)} episodes")
    print(f"   Total frames: {len(dataset)}")

if __name__ == '__main__':
    convert_demonstrations()
```
```bash
# LeRobot으로 학습
python -m lerobot.scripts.train \
    --dataset ./my_robot_dataset \
    --policy act \
    --output-dir ./outputs

# 평가
python -m lerobot.scripts.eval \
    --policy act \
    --checkpoint ./outputs/checkpoints/last.pt
```

**체크:**
- [ ] 데이터 변환 성공
- [ ] LeRobot으로 학습 가능
- [ ] 성능이 자신의 Mini VLA보다 나은지 확인

---

**Week 7-8 완료 체크:**
```
✅ RT-1 완전 이해 (90%)
✅ RT-1 핵심 요소 구현
✅ LeRobot 코드 읽을 수 있음
✅ 다양한 policy 비교 가능
✅ 본인 데이터에 LeRobot 적용

→ 이제 VLA "할 줄 아는" 수준!
```

---

## 📊 8주 완료 후 최종 체크리스트

### 지식
- [ ] VLA가 뭔지 명확히 설명 가능
- [ ] RT-1/RT-2 논문 이해 (80%+)
- [ ] Transformer 개념 이해 (60%+)
- [ ] Behavioral Cloning 완전 이해 (90%+)
- [ ] Vision Transformer 이해 (70%+)

### 기술
- [ ] PyTorch로 모델 자유롭게 작성
- [ ] 간단한 VLA 직접 구현 완료
- [ ] LeRobot 코드 읽고 수정 가능
- [ ] 시뮬레이션 환경 구축 가능
- [ ] 데이터 수집 → 학습 → 평가 파이프라인 이해

### 경험
- [ ] 실제로 작동하는 VLA 만들어봄
- [ ] 실패와 디버깅 경험
- [ ] 모델 성능 개선 경험
- [ ] 다양한 policy 비교 경험

### 마인드셋
- [ ] 100% 이해 안 해도 진행하는 자신감
- [ ] 막혀도 계속 전진하는 습관
- [ ] 실습 중심 학습 체화
- [ ] Top-Down 방식의 효율성 체감

---

## 🚀 9주차 이후: Isaac Sim으로 확장

### Week 9-12: Isaac Sim 전환

**Week 9: Isaac Sim 기초**
- Isaac Sim 설치
- 기본 환경 구축
- 로봇 로드 및 제어
- 카메라 셋업

**Week 10: 모바일 매니퓰레이터**
- 이동 로봇 base
- 매니퓰레이터 장착
- 통합 제어
- ROS2 연동

**Week 11: 물류 환경**
- 창고 환경 구성
- 팔레트/박스 추가
- 조명 설정
- 현실적인 환경

**Week 12: 첫 물류 VLA**
- Pallet grasping task
- 데이터 수집 (100+ episodes)
- VLA 학습
- 평가 (목표: 60%+ 성공률)

### Week 13-18: 고도화

**Week 13-15: 프로젝트 2**
- Box sorting VLA
- Multi-task learning
- ROS2 완전 통합

**Week 16-18: 포트폴리오**
- 프로젝트 3 (선택)
- GitHub 정리
- 블로그 글 작성
- 데모 비디오

---

## 💡 학습 팁

### 1. 막혀도 계속 진행
```
❌ "Transformer 완벽히 이해하고 다음"
   → 3주 낭비

✅ "Transformer 대충 알고 일단 진행"
   → 나중에 필요하면 다시 배움
   → 훨씬 효율적!
```

### 2. 이해도 30%면 충분
```
1주차: 10% → "VLA가 뭔지 모르겠다"
2주차: 30% → "대충 뭔지 알겠는데..."
4주차: 60% → "아, 이래서 이렇게 하는구나"
8주차: 80% → "논문 다시 읽으니 다 보이네"

완벽한 이해: 평생 안 옴, 필요도 없음!
```

### 3. 실습이 80%
```
시간 배분:
- 논문 읽기: 10%
- 이론 공부: 10%
- 코딩: 40%
- 실험: 30%
- 디버깅: 10%

→ 읽기 20%, 손으로 80%!
```

### 4. 작은 성공 축하하기
```
✅ 코드가 돌아감 → 축하!
✅ Loss가 감소함 → 축하!
✅ 로봇이 조금 움직임 → 축하!
✅ 한 번이라도 성공 → 축하!

→ 동기부여 유지가 가장 중요!
```

### 5. 친구 활용
```
혼자:
- 막히면 하루 종일 헤맴
- 잘못된 방향으로 며칠
- 동기부여 하락

친구와:
- 막히면 즉시 질문
- 올바른 방향 안내
- 함께 진행

→ 학습 속도 3배 차이!
```

---

## ❓ FAQ

### Q1: "정말 8주만에 가능해?"

**A: 조건부 YES**

✅ 가능한 경우:
- 프로그래밍 기본 있음
- 하루 2-3시간 투자
- 100% 이해 안 해도 진행
- 막혀도 계속 전진

❌ 불가능한 경우:
- 완벽주의자
- 순서대로 해야 하는 성격
- 하루 30분만 투자

---

### Q2: "이론이 빈약하지 않나?"

**A: 목적에 따라 다름**

VLA 연구자 목표:
- Bottom-Up 필요 (박사 과정)
- 수학/이론 깊이 필수

VLA 활용자 목표 (당신):
- Top-Down 충분 (실무)
- 이론은 필요한 만큼만

---

### Q3: "나중에 구멍 안 나?"

**A: 구멍은 생김, 하지만...**

Top-Down:
- 큰 구멍 없음 (전체 그림 O)
- 작은 구멍 있음 (세부 이론)
- 필요할 때 채우면 됨

Bottom-Up:
- 작은 구멍 없음 (완벽한 기초)
- 큰 구멍 가능 (실전 감각 X)
- 18개월 후에도 시작 못 함

→ Top-Down이 더 실용적!

---

### Q4: "육아하면서 가능한가?"

**A: 조정 필요**

이상적: 8주 (하루 2-3시간)
육아 중: 12-16주 (하루 1시간)

전략:
- 하루 30분이라도 매일
- 주말에 집중 (2-3시간)
- 아이 재울 때 1시간
- 친구와 온라인 스터디

→ 느려도 괜찮음, 방향이 중요!

---

## 🎯 최종 메시지

### Top-Down 학습의 본질
```
전통적:
기초 → 응용 → 실전
└─ 기초에만 6개월

Top-Down:
실전 → 필요한 기초 → 깊이
└─ 1주일에 전체 돌림
└─ 8주면 만들어봄
└─ 재미있어서 계속함!
```

### 당신의 18개월
```
Bottom-Up:
"18개월 후에도 이론만..."

Top-Down:
"8주 후 간단한 VLA 작동" ✅
"12주 후 물류 프로젝트 시작" ✅
"18개월 후 포트폴리오 3개" ✅
```

---

## 📝 학습 기록 템플릿

### 주간 회고
```markdown
## Week X 회고

### 완료한 것
- [ ] 

### 배운 것
- 

### 어려웠던 것
- 

### 다음 주 계획
- [ ] 

### 친구에게 질문할 것
- 
```

### 프로젝트 노트
```markdown
## 프로젝트: [이름]

### 목표
- 

### 데이터
- 에피소드 수:
- 프레임 수:
- 환경:

### 모델
- 아키텍처:
- 파라미터 수:

### 결과
- 학습 Loss:
- 검증 Loss:
- 성공률:

### 교훈
- 
```

---

## 🚀 시작하기

### 지금 당장 (30분)
```bash
# 1. 논문 다운로드
wget https://arxiv.org/pdf/2212.06817.pdf

# 2. LeRobot 설치
pip install lerobot

# 3. 첫 실행
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht
```

### 이번 주 (Week 1)

- [ ] RT-1 논문 읽기 (3시간)
- [ ] 영상 보기 (2시간)
- [ ] LeRobot 돌려보기 (4시간)
- [ ] 학습 노트 작성
- [ ] 친구에게 진행 상황 공유

### 친구에게 물어볼 것
```
1. "RT-1 논문에서 이 부분이 이해 안 가는데..."
2. "LeRobot이 이렇게 작동하는 게 맞아?"
3. "다음 주에 뭐 공부하면 좋을까?"
```

---