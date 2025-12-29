# VLA 학습 가이드 - Phase 1

## 목차
- [🎯 Phase 1: Top-Down 빠른 돌파 (1-2개월)](#-phase-1-top-down-빠른-돌파-1-2개월)
- [Week 1-2: VLA 맛보기](#week-1-2-vla-맛보기)
  - [Week 1: 논문과 실행](#week-1-논문과-실행)
    - [Day 1-3: RT-1 논문 첫 읽기](#day-1-3-rt-1-논문-첫-읽기)
    - [Day 4-5: 영상으로 직관](#day-4-5-영상으로-직관)
    - [Day 6-7: LeRobot 실행해보기](#day-6-7-lerobot-실행해보기)
- [Week 2: 최소한의 기초만](#week-2-최소한의-기초만)
  - [PyTorch 속성 (3일)](#pytorch-속성-3일)
  - [Transformer 개념만 (2일)](#transformer-개념만-2일)
  - [Behavioral Cloning (2일)](#behavioral-cloning-2일)
- [Week 3-4: Mini VLA 만들기](#week-3-4-mini-vla-만들기)
  - [프로젝트: PyBullet로 블록 밀기](#프로젝트-pybullet로-블록-밀기)
    - [Week 3: 환경 구축](#week-3-환경-구축)
    - [Week 4: 학습 및 평가](#week-4-학습-및-평가)
- [Week 5-8: LeRobot 마스터](#week-5-8-lerobot-마스터)
  - [Week 5-6: 다양한 Policy 실험](#week-5-6-다양한-policy-실험)
  - [Week 7-8: RT-1 핵심 요소 구현](#week-7-8-rt-1-핵심-요소-구현)
- [Phase 1 완료 체크](#phase-1-완료-체크)



## 🎯 Phase 1: Top-Down 빠른 돌파 (1-2개월)

### 목표
- VLA가 뭔지 감 잡기
- 간단한 VLA 직접 만들어보기
- "나도 할 수 있다" 자신감

---

## Week 1-2: VLA 맛보기

### Week 1: 논문과 실행

#### Day 1-3: RT-1 논문 첫 읽기

**목표: 30% 이해도로 읽기**
```
체크리스트:
- [ ] Abstract 3번 읽기
- [ ] Figure 전부 보기
- [ ] 모르는 용어 리스트업
- [ ] 전체 흐름만 파악

시간: 하루 1시간 × 3일
```

**읽는 방법:**
1. Abstract만 3번 읽기
2. Introduction 정독
3. Figure 전부 보기 (그림이 핵심!)
4. Method 훑어보기 (모르는 거 메모만)
5. Results 결과만 보기

**예상 반응:**
```
"Transformer가 뭐지?"
"Token이 뭐야?"
"FiLM layer는?"
"Behavioral Cloning?"

→ 정상입니다! 계속 진행하세요
```

---

#### Day 4-5: 영상으로 직관

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

**시간: 2-3시간**

---

#### Day 6-7: LeRobot 실행해보기

**목표: 코드 한 줄도 이해 못 해도 일단 돌려보기!**
```bash
# 설치
pip install lerobot

# 데이터 시각화
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht

# 학습 실행
python -m lerobot.scripts.train \
    --dataset lerobot/pusht \
    --policy act \
    --num-epochs 10
```

**관찰할 것:**
- [ ] 뭐가 입력인가? (이미지)
- [ ] 뭐가 출력인가? (action)
- [ ] 학습이 뭘 하는가? (loss 감소)
- [ ] 결과가 뭔가? (로봇 움직임)

**시간: 4-6시간**

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

## Week 2: 최소한의 기초만

### PyTorch 속성 (3일)

**Day 1: Tensor 기본**
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

**Day 2: nn.Module**
```python
import torch.nn as nn

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

**Day 3: Training Loop**
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

**자료:**
- PyTorch 공식 튜토리얼: "Deep Learning with PyTorch: A 60 Minute Blitz"

**시간: 하루 1시간 × 3일**

---

### Transformer 개념만 (2일)

**필수 개념:**
- Self-Attention이 뭔지
- Query, Key, Value
- ViT가 이미지를 어떻게 처리하는지

**자료:**
- "The Illustrated Transformer" 블로그 (필독!)
- 3Blue1Brown "Attention" 영상

**깊이 조절:**
- ❌ 수식 유도 → 너무 깊음
- ❌ 완벽한 이해 → 시간 낭비
- ✅ 개념과 직관 → 충분함

**시간: 하루 1시간 × 2일**

---

### Behavioral Cloning (2일)

**핵심 개념:**
```python
# BC = Supervised Learning for Actions

# 전통적 Supervised Learning:
# 입력: 이미지
# 출력: 라벨 (고양이/강아지)
# 손실: Cross-entropy

# Behavioral Cloning:
# 입력: Observation (이미지)
# 출력: Action (joint angles, gripper)
# 손실: MSE (Mean Squared Error)

# → 완전히 똑같은 원리!
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

# 이게 BC의 전부!
```

**시간: 하루 1시간 × 2일**

---

**Week 2 완료 체크:**
```
✅ PyTorch 기본 (50%)
✅ Transformer 개념 (30%)
✅ BC 이해 (70%)

→ Mini VLA 만들 준비 됨!
```

---

## Week 3-4: Mini VLA 만들기

### 프로젝트: PyBullet로 블록 밀기

#### Week 3: 환경 구축

**Day 1-2: PyBullet 기본**
```python
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

#### Week 4: 학습 및 평가

**Day 1-3: 모델 정의 및 학습**
```python
# mini_vla.py
import torch
import torch.nn as nn
from transformers import ViTModel

class MiniVLA(nn.Module):
    """
    초간단 VLA 모델
    - Vision: Pre-trained ViT
    - Policy: MLP
    """
    
    def __init__(self, action_dim=7):
        super().__init__()
        
        # Vision Encoder (ViT-Base)
        self.vision = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Policy Head
        self.policy = nn.Sequential(
            nn.Linear(768, 256),  # ViT hidden_size
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

# Dataset
from torch.utils.data import Dataset, DataLoader
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

# 학습
def train_mini_vla():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로드
    dataset = RobotDataset('demonstrations.pkl')
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    # 모델
    model = MiniVLA(action_dim=7).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 학습
    num_epochs = 50
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for obs, actions in dataloader:
            obs = obs.to(device)
            actions = actions.to(device)
            
            # Forward
            pred_actions = model(obs)
            loss = criterion(pred_actions, actions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoint_epoch{epoch+1}.pt')
    
    print("🎉 Training complete!")

if __name__ == '__main__':
    train_mini_vla()
```

---

**Day 4-7: 평가 및 개선**
```python
# evaluate.py
def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = MiniVLA(action_dim=7).to(device)
    model.load_state_dict(torch.load('checkpoint_epoch50.pt'))
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
        
        # 목표 위치
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
            
            # 성공 체크
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

**기대 결과:**
- 첫 시도: 20-40% 성공률 (정상!)
- 데이터/모델 개선 후: 50-60%

---

## Week 5-8: LeRobot 마스터

### Week 5-6: 다양한 Policy 실험
```bash
# ACT Policy
python -m lerobot.scripts.train \
    --dataset lerobot/pusht \
    --policy act \
    --batch-size 32 \
    --num-epochs 100

# Diffusion Policy
python -m lerobot.scripts.train \
    --dataset lerobot/pusht \
    --policy diffusion \
    --batch-size 32 \
    --num-epochs 100

# 성능 비교
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

### Week 7-8: RT-1 핵심 요소 구현
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
        
        # Decoder
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

## Phase 1 완료 체크
```
✅ VLA 전체 그림 이해 (70%)
✅ 간단한 VLA 만들어봄
✅ 실패와 성공 경험
✅ "할 수 있다" 자신감
✅ 부족한 부분 명확히 파악
✅ RT-1 핵심 요소 이해

→ 이제 제대로 배울 준비!
```