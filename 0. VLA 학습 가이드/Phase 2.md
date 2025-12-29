# VLA 학습 가이드 - Phase 2 수정본 (AMR Component 통합)

## 📋 Phase 2 수정 요약

### 변경 사항

| 항목 | 기존 | 수정 | 이유 |
|------|------|------|------|
| 기간 | 3-6개월 | 2-3개월 | ROS2 경험으로 축소 가능 |
| PyTorch 심화 | 필수 | 선택적/축소 | 기본은 Phase 1에서 완료 |
| CNN 깊이 파기 | 필수 | 축소 | ViT 중심으로 전환 |
| RL 기초 | 필수 | 최소화 | VLA는 BC 중심 |
| 수학 | 별도 진행 | 3Blue1Brown 진행중 활용 | 이미 학습 중 |

### Phase 2 위치 재정의

```
기존 로드맵:
Phase 1 (필수) → Phase 2 (필수) → Phase 3 → Phase 4

수정 로드맵:
Phase 1 (필수) → Phase 2 (선택적 보강) → Phase 3 → Phase 4
                      ↑
                필요한 부분만 선택적으로
```

---

## 🎯 Phase 2 목표 (수정)

- [x] Phase 1에서 부족했던 부분만 선택적으로 채우기
- [x] Transformer/ViT 심화 (핵심)
- [x] Imitation Learning 심화 (핵심)
- [x] PyTorch 고급 기법 (필요시)
- [ ] ~~CNN 완전 이해~~ → 기본만 (ViT 중심)
- [ ] ~~RL 기초~~ → 개념만 (BC 중심)

---

## 📅 수정된 일정

```
기존 (3-6개월):
Month 3-4: Deep Learning 제대로 (8주)
Month 5:   Transformer & Multi-modal (4주)
Month 6:   Imitation Learning & RL (4주)

수정 (2-3개월):
Week 1-3:  선택적 PyTorch 심화 (필요시만)
Week 4-6:  Transformer & ViT 심화 (핵심)
Week 7-9:  Imitation Learning 심화 (핵심)
Week 10-12: 수학 병행 + 정리
```

---

## Week 1-3: 선택적 PyTorch 심화

### 변경 내용

#### 유지: 핵심 고급 기법만

```python
# 꼭 알아야 할 것만:
# 1. Custom Dataset/DataLoader (AMR 데이터용)
# 2. Mixed Precision Training (GPU 효율)
# 3. Gradient Checkpointing (메모리 절약)

# 나머지는 필요할 때 참조
```

#### 축소 또는 제거

| 기존 내용 | 수정 | 이유 |
|----------|------|------|
| Custom Loss Functions | 유지 | 도킹 태스크용 커스텀 loss |
| LR Scheduling | 축소 | OneCycleLR만 알면 충분 |
| Gradient Clipping | 유지 | 학습 안정성 |
| GPU 메모리 최적화 | 유지 | Jetson 배포 대비 |

#### 추가: AMR 데이터셋 구현

```python
class AMRDockingDataset(Dataset):
    """
    Phase 1에서 수집한 도킹 데이터용
    
    특징:
    - ROS bag → HDF5 변환
    - 이미지 + cmd_vel + odom
    - Action normalization
    """
    
    def __init__(self, data_dir, transform=None):
        self.episodes = self.load_hdf5(data_dir)
        self.compute_action_statistics()
    
    def compute_action_statistics(self):
        """도킹 cmd_vel 통계 (정규화용)"""
        all_actions = []
        for ep in self.episodes:
            all_actions.extend(ep['cmd_vels'])
        
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.std(all_actions, axis=0)
```

---

## Week 4-6: Transformer & ViT 심화 (핵심)

### 유지/강화 내용

#### 강화: Attention 메커니즘

```python
# AMR 관점 추가 질문:
# - 도킹 시 어디에 attention이 집중되나?
# - 충전 포트 vs 배경
# - Attention 시각화로 디버깅 가능?
```

#### 강화: Vision Transformer 심화

```python
class ViTForAMR(nn.Module):
    """
    AMR 도킹용 ViT 분석
    
    주요 포인트:
    - Patch size 영향 (16 vs 32)
    - [CLS] token vs mean pooling
    - Fine-tuning vs Feature extraction
    """
    
    def visualize_attention(self, image):
        """도킹 시 attention 패턴 시각화"""
        # 어디를 보는지 확인
        # → 충전 포트에 집중되어야 함
```

#### 추가: Multi-modal Fusion (AMR 관점)

```python
class AMRMultiModalFusion(nn.Module):
    """
    AMR용 멀티모달 융합
    
    입력:
    - RGB 이미지 (도킹 카메라)
    - LiDAR 점군 또는 거리 정보
    - Proprioception (odom, joint states)
    
    기존 AMR과 차이:
    - 기존: LiDAR 중심 + 카메라 보조
    - VLA: 카메라 중심 + LiDAR 보조
    """
    
    def forward(self, rgb, lidar_range, odom):
        # Vision features
        vis_feat = self.vision_encoder(rgb)
        
        # LiDAR features (간단히)
        lidar_feat = self.lidar_encoder(lidar_range)
        
        # Proprioception
        proprio_feat = self.proprio_encoder(odom)
        
        # Fusion
        fused = self.fusion(vis_feat, lidar_feat, proprio_feat)
        
        return fused
```

---

## Week 7-9: Imitation Learning 심화 (핵심)

### 유지/강화 내용

#### 강화: Behavioral Cloning 심화

```python
# AMR 도킹에 특화된 BC 고려사항

class DockingBC:
    """
    도킹 BC 특수 고려사항:
    
    1. Covariate Shift
       - 전문가는 성공 경로만
       - 실제로는 오차 발생
       → Data Augmentation으로 보완
    
    2. Multi-modal Action
       - 접근 단계: 빠른 속도
       - 도킹 단계: 느린 정밀 속도
       → Mixture Density Network 고려
    
    3. Action Delay
       - 이미지 → 추론 → 실행 latency
       → Action chunking으로 보완
    """
```

#### 강화: Action Chunking

```python
class ActionChunkingPolicy(nn.Module):
    """
    여러 timestep action을 한번에 예측
    
    AMR 도킹에서 장점:
    - 부드러운 trajectory
    - Latency 보상
    - Temporal consistency
    
    설정:
    - chunk_size = 10 (0.5초 @ 20Hz)
    - 매 inference마다 첫 action 실행
    - 나머지는 buffer에 보관
    """
    
    def __init__(self, chunk_size=10):
        self.chunk_size = chunk_size
        self.action_buffer = deque(maxlen=chunk_size)
    
    def forward(self, obs):
        # 10개 action 예측
        actions = self.policy(obs)  # (B, 10, 3)
        return actions
    
    def get_action(self, obs):
        if len(self.action_buffer) == 0:
            # 새로 예측
            actions = self.forward(obs)
            for a in actions[0]:
                self.action_buffer.append(a)
        
        return self.action_buffer.popleft()
```

#### 추가: DAgger (선택적)

```python
"""
DAgger: Dataset Aggregation

BC의 covariate shift 문제 해결

AMR 도킹에서:
1. BC로 초기 policy 학습
2. Policy로 도킹 시도
3. 전문가가 개입하여 교정
4. 새 데이터 추가
5. 재학습
6. 반복

현실적 접근:
- 시뮬레이션에서 DAgger 적용
- 실제 로봇에서는 BC로 충분할 수도
"""
```

### 축소/제거 내용

#### 축소: RL 기초

```python
"""
기존: REINFORCE, PPO 구현
수정: 개념만 이해

이유:
- VLA는 BC 중심
- RL fine-tuning은 Phase 4에서 선택적
- 시간 효율화

알아야 할 것:
- Policy Gradient 개념
- BC와 RL의 차이
- 언제 RL이 필요한지
"""
```

---

## Week 10-12: 수학 병행 + 정리

### 변경 내용

#### 유지: Linear Algebra (진행중 활용)

```markdown
3Blue1Brown 시리즈 계속 진행

VLA 관련 핵심:
- Eigenvalues → PCA, Attention
- Matrix decomposition → 모델 압축
- Linear transformation → Layer 이해
```

#### 축소: Probability & Statistics

```markdown
기존: 깊이있는 확률론
수정: VLA에 필요한 것만

필수:
- Gaussian distribution (action modeling)
- Cross-entropy (classification)
- KL divergence (개념만)

나머지는 필요할 때 참조
```

#### 추가: Phase 3 준비 체크

```markdown
□ Isaac Sim 설치 가능한지 확인 (RTX 4070, 32GB RAM)
□ 회사 AMR URDF 확보 가능한지 확인
□ GPU 서버 사용 가능한지 확인
□ Phase 1 도킹 데이터 정리
```

---

## Phase 2 완료 체크 (수정)

```
기존:
✅ Deep Learning 기초 탄탄 (80%)
✅ PyTorch 자유자재 (85%)
✅ CNN 완전 이해 (80%)
✅ Transformer 완전 이해 (90%)
✅ Multi-modal learning 이해 (80%)
✅ Imitation Learning 심화 (70%)
✅ RL 기초 이해 (60%)
✅ 수학 기초 충분 (70%)

수정:
✅ PyTorch 고급 기법 (필요한 것만) (70%)
✅ Transformer/ViT 심화 (핵심) (85%)
✅ Multi-modal fusion 이해 (AMR 관점) (75%)
✅ Imitation Learning 심화 (핵심) (80%)
✅ Action Chunking 이해 (80%)
⬜ RL 기초 → 개념만 (40%)
✅ 수학 진행중 (3B1B 완주 목표)
✅ Phase 3 준비 완료
```

---

## 학습 시간 추정

### 주당 10-12시간 기준

| 주차 | 내용 | 예상 시간 |
|------|------|----------|
| 1-3 | PyTorch 심화 (선택적) | 15h |
| 4-6 | Transformer/ViT 심화 | 20h |
| 7-9 | Imitation Learning 심화 | 18h |
| 10-12 | 수학 + 정리 | 12h |

**총: ~65시간 (약 2-2.5개월)**

### 유연한 진행

```markdown
## 빨리 끝날 수 있는 경우
- PyTorch 이미 충분 → Week 1-3 건너뛰기
- Transformer 논문 읽기 수월 → Week 4-6 축소

## 늦어질 수 있는 경우
- 기초가 부족하다 느끼면 천천히
- 회사 업무 바쁜 시기

→ Phase 3 시작이 중요하므로 80% 이해로 넘어가도 OK
```

---

## Phase 1 → Phase 2 → Phase 3 연결

```
Phase 1 완료 시점:
- Mini VLA 동작 (Gazebo)
- ROS2 Component 설계
- 70% 이해도

Phase 2에서:
- 부족한 부분 선택적 보강
- 핵심 (Transformer, IL) 심화
- 85% 이해도

Phase 3 시작 시:
- Isaac Sim 본격 사용
- 더 복잡한 VLA 개발
- Production-ready 목표
```