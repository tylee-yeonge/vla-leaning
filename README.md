# VLA Learning Journey 🤖

> Vision-Language-Action (VLA) 모델을 학습하며 기록하는 개인 공부 저장소입니다.

## 📖 소개

안녕하세요! 이 저장소는 제가 **VLA (Vision-Language-Action)** 모델을 공부하면서 배운 내용들을 정리하고 공유하는 공간입니다. 

로보틱스와 AI에 관심이 많아서 RT-1, RT-2 같은 최신 VLA 모델들을 직접 구현해보고 싶었습니다. 전통적인 방식처럼 모든 기초 이론을 다 공부하고 시작하면 시간이 너무 오래 걸릴 것 같아서, **Top-Down 방식**으로 접근하기로 했습니다.

### 🎯 제가 선택한 학습 방법

**왜 Top-Down인가?**

```
❌ Bottom-Up (시간이 너무 오래 걸림):
선형대수 → 미적분 → 최적화 → ML → DL → CV → NLP → RL → VLA
└─ 6개월이 지나도 VLA는 시작도 못 할 것 같았습니다

✅ Top-Down (제가 선택한 방법):
VLA 논문 읽기 → 막히는 부분만 학습 → 구현 → 반복
└─ 빠르게 시작하고 필요한 것만 배우자!
```

**제 학습 원칙:**
1. 큰 그림을 먼저 본다
2. 30% 이해해도 일단 진행한다
3. 막힐 때만 깊게 판다
4. 이론보다 실습에 집중한다

## 📚 학습 계획

대략 8주 정도의 계획을 세워봤습니다.

### Week 1: VLA의 전체 그림 이해하기
- RT-1 논문 읽기
- 관련 영상 보면서 직관 만들기
- HuggingFace LeRobot 실습해보기

### Week 2-3: 필수 기초만 빠르게
- **Transformer**: Attention이 뭔지, ViT는 어떻게 작동하는지
- **Behavioral Cloning**: VLA의 핵심 학습 방법
- **PyTorch**: 필요한 기능만 빠르게 익히기

### Week 4-6: 직접 만들어보기 (Mini VLA 프로젝트)
- PyBullet으로 간단한 시뮬레이션 환경 만들기
- 데이터 수집하기 (Teleoperation)
- 간단한 VLA 모델 구현해서 학습시켜보기

### Week 7-8: 실전 VLA 모델 분석
- RT-1/RT-2 상세히 뜯어보기
- OpenVLA, π0 등 최신 모델 공부하기
- 프로젝트 확장 및 개선

## �️ 개발 환경

제가 사용하는 환경입니다:

- **Python 3.8+**
- **PyTorch** (딥러닝 프레임워크)
- **PyBullet** (로봇 시뮬레이션)
- **HuggingFace LeRobot** (VLA 실습용)
- **GPU** (학습 속도를 위해 사용 중)

## 📂 저장소 구조

```
vla-leaning/
├── LEARNING_GUIDE.md      # 상세한 학습 계획 및 노트
├── README.md              # 이 파일
└── ...                    # 학습하면서 추가되는 코드와 자료들
```

## 💭 학습하면서 느낀 점

- **완벽주의는 독**: 30% 이해하고 넘어가도 괜찮다는 걸 배웠습니다
- **일단 돌려보기**: 코드를 완전히 이해 못 해도 실행해보면 감이 옵니다
- **필요할 때 학습하기**: 막힐 때마다 그 부분만 집중적으로 공부하니 효율적입니다
- **꾸준함이 핵심**: 매일 조금씩이라도 진행하는 게 중요합니다

## 📖 주요 참고 자료

### 논문
- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- RT-2, OpenVLA 등 (공부하면서 추가 중)

### 프레임워크 & 도구
- [LeRobot (HuggingFace)](https://github.com/huggingface/lerobot)
- PyTorch
- PyBullet

## 🤝 함께 공부하시는 분들께

같은 주제에 관심 있으시다면 이 저장소가 도움이 되셨으면 좋겠습니다. 저도 배우는 단계라 완벽하지 않지만, 제가 겪은 시행착오나 배운 점들을 공유하고 싶습니다.

질문이나 피드백은 언제든 환영합니다!

---

**2025년 12월부터 시작한 VLA 학습 여정 🚀**