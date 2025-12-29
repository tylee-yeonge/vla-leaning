# VLA 하이브리드 학습 가이드 - Phase 4

## 목차
- [📅 Phase 4: 포트폴리오 완성 & 취업 준비 (13-18개월)](#-phase-4-포트폴리오-완성--취업-준비-13-18개월)
- [Month 13-14: 프로젝트 2 - Multi-Task VLA](#month-13-14-프로젝트-2---multi-task-vla)
  - [Week 1-2: Multi-Task 설계](#week-1-2-multi-task-설계)
  - [Week 3-4: Zero-Shot Generalization](#week-3-4-zero-shot-generalization)
- [Month 15-16: 프로젝트 3 - Visual SLAM Integration](#month-15-16-프로젝트-3---visual-slam-integration)
  - [Week 1-2: ORB-SLAM3 Integration](#week-1-2-orb-slam3-integration)
- [Month 17: 포트폴리오 & 블로그](#month-17-포트폴리오--블로그)
  - [Week 1-2: GitHub 포트폴리오](#week-1-2-github-포트폴리오)
  - [Week 3-4: 기술 블로그](#week-3-4-기술-블로그)
- [Month 18: 취업 준비](#month-18-취업-준비)
  - [Week 1-2: 이력서 & 포트폴리오 정리](#week-1-2-이력서--포트폴리오-정리)
  - [Week 3-4: 면접 준비](#week-3-4-면접-준비)
- [Phase 4 완료 체크](#phase-4-완료-체크)
- [최종 로드맵 요약](#최종-로드맵-요약)

## 📅 Phase 4: 포트폴리오 완성 & 취업 준비 (13-18개월)

### 목표
- 3개의 완성도 높은 프로젝트
- 기술 블로그 & GitHub 포트폴리오
- 논문 리뷰 및 재현
- 오픈소스 기여
- 이력서 & 면접 준비
- AI Perception Engineer 이직 성공!

---

## Month 13-14: 프로젝트 2 - Multi-Task VLA

### Week 1-2: Multi-Task 설계

#### Task 정의
````python
# multi_task_design.py

class MultiTaskVLA:
    """
    하나의 모델로 여러 task 수행
    
    Tasks:
    1. Box Picking (기존)
    2. Box Placing
    3. Box Stacking
    4. Object Sorting
    5. Drawer Opening
    
    목표: Task-conditioned policy
    """
    
    def __init__(self, config):
        self.config = config
        self.tasks = self.define_tasks()
    
    def define_tasks(self):
        """
        Task 정의
        """
        tasks = {
            'pick': {
                'description': 'Pick up a box from shelf',
                'success_criteria': 'gripper_holding_object',
                'reward_function': self.pick_reward,
                'difficulty': 'easy'
            },
            
            'place': {
                'description': 'Place box on target location',
                'success_criteria': 'box_on_target',
                'reward_function': self.place_reward,
                'difficulty': 'medium'
            },
            
            'stack': {
                'description': 'Stack box on top of another',
                'success_criteria': 'box_stacked_stable',
                'reward_function': self.stack_reward,
                'difficulty': 'hard'
            },
            
            'sort': {
                'description': 'Sort boxes by size/color',
                'success_criteria': 'all_boxes_sorted',
                'reward_function': self.sort_reward,
                'difficulty': 'medium'
            },
            
            'drawer': {
                'description': 'Open drawer',
                'success_criteria': 'drawer_open',
                'reward_function': self.drawer_reward,
                'difficulty': 'hard'
            }
        }
        
        return tasks
    
    def pick_reward(self, state):
        """
        Pick task reward
        """
        reward = 0.0
        
        # Distance to object
        ee_pos = state['ee_position']
        obj_pos = state['target_object']['position']
        distance = np.linalg.norm(ee_pos - obj_pos)
        
        reward += -distance  # Closer is better
        
        # Grasp attempt
        if state['gripper_closing'] and distance < 0.05:
            reward += 1.0
        
        # Success
        if state['gripper_holding_object']:
            reward += 10.0
        
        return reward
    
    def place_reward(self, state):
        """
        Place task reward
        """
        reward = 0.0
        
        # Object must be grasped
        if not state['gripper_holding_object']:
            return -10.0
        
        # Distance to target
        obj_pos = state['held_object']['position']
        target_pos = state['target_location']
        distance = np.linalg.norm(obj_pos - target_pos)
        
        reward += -distance
        
        # On target
        if distance < 0.05 and state['gripper_opening']:
            reward += 5.0
        
        # Success (stable placement)
        if state['object_on_target'] and state['object_stable']:
            reward += 10.0
        
        return reward
    
    def stack_reward(self, state):
        """
        Stacking task reward
        """
        reward = 0.0
        
        # Pick phase
        if not state['gripper_holding_object']:
            # Distance to target box
            distance = np.linalg.norm(
                state['ee_position'] - state['target_box']['position']
            )
            reward += -distance
        
        # Place phase
        else:
            # Alignment with base box
            held_pos = state['held_object']['position']
            base_pos = state['base_box']['position']
            
            # Horizontal alignment
            horizontal_offset = np.linalg.norm(held_pos[:2] - base_pos[:2])
            reward += -horizontal_offset * 2.0
            
            # Vertical distance
            vertical_distance = abs(held_pos[2] - (base_pos[2] + 0.15))
            reward += -vertical_distance
            
            # Success
            if state['box_stacked'] and state['stable']:
                reward += 15.0
        
        return reward
````

---

#### Language Conditioning
````python
# language_conditioning.py

class LanguageConditionedVLA(nn.Module):
    """
    Language-conditioned VLA
    
    Input:
    - Image
    - Proprioception  
    - Language instruction
    
    Output:
    - Action sequence
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Vision encoder
        from transformers import ViTModel
        self.vision_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224'
        )
        vision_dim = 768
        
        # Language encoder
        from transformers import BertModel
        self.language_encoder = BertModel.from_pretrained(
            'bert-base-uncased'
        )
        language_dim = 768
        
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        proprio_dim = 256
        
        # Cross-modal fusion (FiLM layer)
        self.fusion = FiLMFusion(
            vision_dim=vision_dim,
            language_dim=language_dim,
            proprio_dim=proprio_dim,
            output_dim=512
        )
        
        # Action decoder
        self.action_decoder = ActionDecoder(
            input_dim=512,
            action_dim=7,
            num_action_steps=10
        )
    
    def forward(self, observations):
        """
        Forward pass
        
        Args:
            observations: dict with
                - 'rgb': (B, 3, 224, 224)
                - 'proprio': (B, 15)
                - 'instruction': (B, max_len)
        
        Returns:
            actions: (B, num_action_steps, 7)
        """
        # Encode vision
        vision_features = self.vision_encoder(
            observations['rgb']
        ).last_hidden_state  # (B, N, 768)
        
        # Encode language
        language_features = self.language_encoder(
            observations['instruction']
        ).last_hidden_state[:, 0]  # (B, 768) - [CLS] token
        
        # Encode proprioception
        proprio_features = self.proprio_encoder(
            observations['proprio']
        )  # (B, 256)
        
        # Cross-modal fusion
        fused_features = self.fusion(
            vision_features,
            language_features,
            proprio_features
        )  # (B, 512)
        
        # Decode actions
        actions = self.action_decoder(fused_features)  # (B, T, 7)
        
        return actions

class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation
    
    Language로 vision feature를 condition
    """
    
    def __init__(self, vision_dim, language_dim, proprio_dim, output_dim):
        super().__init__()
        
        # Language projection
        self.lang_proj = nn.Linear(language_dim, output_dim)
        
        # FiLM parameters from language
        self.gamma_net = nn.Linear(language_dim, vision_dim)
        self.beta_net = nn.Linear(language_dim, vision_dim)
        
        # Vision projection
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        
        # Proprio projection
        self.proprio_proj = nn.Linear(proprio_dim, output_dim)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, vision_features, language_features, proprio_features):
        """
        Args:
            vision_features: (B, N, vision_dim)
            language_features: (B, language_dim)
            proprio_features: (B, proprio_dim)
        
        Returns:
            fused: (B, output_dim)
        """
        # FiLM conditioning
        gamma = self.gamma_net(language_features).unsqueeze(1)  # (B, 1, vision_dim)
        beta = self.beta_net(language_features).unsqueeze(1)
        
        # Modulate vision features
        modulated_vision = gamma * vision_features + beta  # (B, N, vision_dim)
        
        # Pool vision features
        pooled_vision = modulated_vision.mean(dim=1)  # (B, vision_dim)
        
        # Project all modalities
        vision_proj = self.vision_proj(pooled_vision)  # (B, output_dim)
        language_proj = self.lang_proj(language_features)
        proprio_proj = self.proprio_proj(proprio_features)
        
        # Concatenate and fuse
        combined = torch.cat([
            vision_proj,
            language_proj,
            proprio_proj
        ], dim=-1)  # (B, output_dim * 3)
        
        fused = self.fusion(combined)  # (B, output_dim)
        
        return fused

# Task instruction templates
TASK_INSTRUCTIONS = {
    'pick': [
        "pick up the {color} box",
        "grasp the {size} box from the shelf",
        "grab the box on the {position}",
    ],
    
    'place': [
        "place the box on the {location}",
        "put the box down at {coordinates}",
        "set the box on the pallet",
    ],
    
    'stack': [
        "stack the {color} box on top of the {base_color} box",
        "place this box on the stack",
        "build a stack with {number} boxes",
    ],
    
    'sort': [
        "sort boxes by {criteria}",
        "organize the {color} boxes to the {direction}",
        "separate small and large boxes",
    ],
    
    'drawer': [
        "open the {position} drawer",
        "pull the drawer {direction}",
        "access the top drawer",
    ]
}

def generate_instruction(task_type, **kwargs):
    """
    Generate task instruction
    """
    templates = TASK_INSTRUCTIONS[task_type]
    template = random.choice(templates)
    
    # Fill in placeholders
    instruction = template.format(**kwargs)
    
    return instruction
````

---

#### Multi-Task Dataset
````python
# multi_task_dataset.py

class MultiTaskDataset(Dataset):
    """
    Multi-task dataset
    
    구조:
    - Task ID
    - Instruction
    - Observation
    - Action
    """
    
    def __init__(self, data_dir, tasks=['pick', 'place', 'stack']):
        self.data_dir = data_dir
        self.tasks = tasks
        
        # Load all episodes
        self.episodes = []
        
        for task in tasks:
            task_episodes = self.load_task_episodes(task)
            self.episodes.extend(task_episodes)
        
        # Shuffle
        random.shuffle(self.episodes)
        
        # Tokenizer
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def load_task_episodes(self, task):
        """
        Load episodes for a task
        """
        task_dir = os.path.join(self.data_dir, task)
        episodes = []
        
        for episode_file in os.listdir(task_dir):
            if episode_file.endswith('.pkl'):
                with open(os.path.join(task_dir, episode_file), 'rb') as f:
                    episode = pickle.load(f)
                    episode['task'] = task
                    episodes.append(episode)
        
        print(f"Loaded {len(episodes)} episodes for task '{task}'")
        
        return episodes
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # Random timestep
        timestep = random.randint(0, len(episode['observations']) - 1)
        
        # Observation
        obs = episode['observations'][timestep]
        
        # Instruction
        instruction = episode['instruction']
        instruction_tokens = self.tokenizer(
            instruction,
            padding='max_length',
            max_length=20,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        # Action chunk
        action_chunk = episode['actions'][timestep:timestep+10]
        
        # Pad if needed
        if len(action_chunk) < 10:
            padding = [np.zeros(7)] * (10 - len(action_chunk))
            action_chunk = list(action_chunk) + padding
        
        action_chunk = np.array(action_chunk)
        
        return {
            'rgb': torch.FloatTensor(obs['rgb']),
            'proprio': torch.FloatTensor(obs['proprio']),
            'instruction': instruction_tokens,
            'action': torch.FloatTensor(action_chunk),
            'task': episode['task']
        }

# Data collection for multiple tasks
def collect_multi_task_data(env, tasks, episodes_per_task=50):
    """
    모든 task에 대한 데이터 수집
    """
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Collecting data for task: {task}")
        print(f"{'='*60}")
        
        # Configure environment for task
        env.set_task(task)
        
        # Collect episodes
        task_episodes = []
        
        for ep in range(episodes_per_task):
            # Generate instruction
            instruction = generate_instruction(task, **env.get_task_params())
            
            print(f"\nEpisode {ep+1}/{episodes_per_task}")
            print(f"Instruction: {instruction}")
            
            # Collect episode
            episode = collect_episode(env, instruction)
            
            if episode['success']:
                task_episodes.append(episode)
                print("✅ Success")
            else:
                print("❌ Failed")
        
        # Save
        save_dir = f'data/{task}'
        os.makedirs(save_dir, exist_ok=True)
        
        for i, episode in enumerate(task_episodes):
            with open(f'{save_dir}/episode_{i:03d}.pkl', 'wb') as f:
                pickle.dump(episode, f)
        
        print(f"\n💾 Saved {len(task_episodes)} episodes for {task}")
````

---

#### Multi-Task Training
````python
# multi_task_training.py

class MultiTaskTrainer:
    """
    Multi-task VLA training
    
    특징:
    - Task balancing
    - Curriculum learning
    - Multi-task metrics
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )
        
        # Loss
        self.criterion = nn.MSELoss()
        
        # Task weights (for balancing)
        self.task_weights = {
            'pick': 1.0,
            'place': 1.0,
            'stack': 1.5,  # Harder task, higher weight
            'sort': 1.2,
            'drawer': 1.5
        }
    
    def train_epoch(self, dataloader):
        """
        Single training epoch
        """
        self.model.train()
        
        task_losses = {task: [] for task in self.task_weights.keys()}
        
        for batch in tqdm(dataloader, desc='Training'):
            # Move to device
            obs = {
                'rgb': batch['rgb'].cuda(),
                'proprio': batch['proprio'].cuda(),
                'instruction': batch['instruction'].cuda()
            }
            actions = batch['action'].cuda()
            tasks = batch['task']
            
            # Forward
            pred_actions = self.model(obs)
            
            # Task-weighted loss
            loss = 0.0
            
            for task in self.task_weights.keys():
                # Mask for this task
                task_mask = torch.tensor([t == task for t in tasks]).cuda()
                
                if task_mask.sum() > 0:
                    # Task-specific loss
                    task_loss = self.criterion(
                        pred_actions[task_mask],
                        actions[task_mask]
                    )
                    
                    # Weighted
                    weighted_loss = task_loss * self.task_weights[task]
                    loss += weighted_loss
                    
                    # Record
                    task_losses[task].append(task_loss.item())
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        # Report per-task losses
        print("\nPer-task losses:")
        for task, losses in task_losses.items():
            if losses:
                print(f"  {task:10s}: {np.mean(losses):.4f}")
        
        return {task: np.mean(losses) if losses else 0 
                for task, losses in task_losses.items()}
    
    def evaluate(self, val_loader):
        """
        Multi-task evaluation
        """
        self.model.eval()
        
        task_losses = {task: [] for task in self.task_weights.keys()}
        
        with torch.no_grad():
            for batch in val_loader:
                obs = {
                    'rgb': batch['rgb'].cuda(),
                    'proprio': batch['proprio'].cuda(),
                    'instruction': batch['instruction'].cuda()
                }
                actions = batch['action'].cuda()
                tasks = batch['task']
                
                # Forward
                pred_actions = self.model(obs)
                
                # Per-task loss
                for task in self.task_weights.keys():
                    task_mask = torch.tensor([t == task for t in tasks]).cuda()
                    
                    if task_mask.sum() > 0:
                        task_loss = self.criterion(
                            pred_actions[task_mask],
                            actions[task_mask]
                        )
                        task_losses[task].append(task_loss.item())
        
        # Report
        print("\nValidation per-task losses:")
        for task, losses in task_losses.items():
            if losses:
                print(f"  {task:10s}: {np.mean(losses):.4f}")
        
        return task_losses

# Curriculum learning
class MultiTaskCurriculum:
    """
    Task별 난이도 조절
    
    Easy → Medium → Hard
    """
    
    def __init__(self, tasks):
        self.tasks = tasks
        self.difficulty = {
            'pick': 1,    # Easy
            'place': 2,   # Medium
            'sort': 2,    # Medium
            'stack': 3,   # Hard
            'drawer': 3   # Hard
        }
        
        self.current_stage = 1
    
    def get_active_tasks(self):
        """
        현재 stage의 task들
        """
        active = [task for task, diff in self.difficulty.items() 
                  if diff <= self.current_stage]
        
        return active
    
    def advance_stage(self):
        """
        다음 stage로
        """
        if self.current_stage < 3:
            self.current_stage += 1
            print(f"\n🎓 Advanced to stage {self.current_stage}")
            print(f"   Active tasks: {self.get_active_tasks()}")
    
    def should_advance(self, task_metrics):
        """
        Stage 진급 조건
        
        모든 현재 task가 threshold 이상
        """
        active_tasks = self.get_active_tasks()
        
        for task in active_tasks:
            if task_metrics.get(task, 0) < 0.7:  # 70% success
                return False
        
        return True
````

**시간: 주 8-10시간**

---

### Week 3-4: Zero-Shot Generalization
````python
# zero_shot_generalization.py

class ZeroShotEvaluator:
    """
    Zero-shot generalization 테스트
    
    학습하지 않은 task/instruction에 대한 성능
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def test_novel_instructions(self):
        """
        새로운 instruction 테스트
        
        예:
        - 학습: "pick up the red box"
        - 테스트: "grasp the crimson container"
        """
        novel_instructions = [
            # Synonym variations
            "grasp the scarlet box",
            "grab the tiny cube",
            "lift the azure container",
            
            # Different phrasing
            "I need you to pick up the green box",
            "Could you place the box on the left pallet",
            
            # Negation
            "pick up the box that is not blue",
            "place on the pallet that is not full",
            
            # Relative references
            "pick the box next to the red one",
            "place on the empty spot",
            
            # Multi-step implicit
            "move the box from shelf A to pallet B",
        ]
        
        results = {}
        
        for instruction in novel_instructions:
            print(f"\nTesting: {instruction}")
            
            success = self.execute_instruction(instruction)
            results[instruction] = success
            
            print(f"  Result: {'✅' if success else '❌'}")
        
        # Analysis
        success_rate = np.mean(list(results.values()))
        print(f"\n{'='*60}")
        print(f"Zero-shot success rate: {success_rate*100:.1f}%")
        print(f"{'='*60}")
        
        return results
    
    def test_novel_objects(self):
        """
        새로운 객체 테스트
        
        학습: 빨강/파랑/초록 박스
        테스트: 보라/노랑/검정 박스
        """
        novel_objects = [
            {'color': 'purple', 'size': 'medium'},
            {'color': 'yellow', 'size': 'small'},
            {'color': 'black', 'size': 'large'},
            {'color': 'orange', 'size': 'medium'},
        ]
        
        results = []
        
        for obj in novel_objects:
            instruction = f"pick up the {obj['color']} {obj['size']} box"
            
            # Spawn object
            self.env.spawn_object(**obj)
            
            # Test
            success = self.execute_instruction(instruction)
            results.append(success)
            
            print(f"{obj['color']:8s} box: {'✅' if success else '❌'}")
        
        success_rate = np.mean(results)
        print(f"\nNovel object success: {success_rate*100:.1f}%")
        
        return success_rate
    
    def test_novel_environments(self):
        """
        새로운 환경 테스트
        
        학습: 창고 A
        테스트: 창고 B (다른 layout)
        """
        novel_envs = [
            'warehouse_b',  # Different shelf positions
            'warehouse_c',  # Different lighting
            'warehouse_d',  # Cluttered
        ]
        
        results = {}
        
        for env_name in novel_envs:
            print(f"\nTesting environment: {env_name}")
            
            # Load environment
            self.load_environment(env_name)
            
            # Run standard tasks
            success_rates = self.run_standard_tasks()
            
            results[env_name] = success_rates
            
            print(f"  Average success: {np.mean(list(success_rates.values()))*100:.1f}%")
        
        return results
    
    def test_compositional_tasks(self):
        """
        조합 task 테스트
        
        학습: pick, place (개별)
        테스트: pick → place (조합)
        """
        compositional_tasks = [
            {
                'instruction': "pick the red box and place it on pallet A",
                'subtasks': ['pick', 'place']
            },
            {
                'instruction': "sort all small boxes to the left",
                'subtasks': ['pick', 'sort', 'place']
            },
            {
                'instruction': "stack three boxes in size order",
                'subtasks': ['pick', 'sort', 'stack']
            }
        ]
        
        results = []
        
        for task in compositional_tasks:
            print(f"\nTask: {task['instruction']}")
            
            success = self.execute_compositional_task(task)
            results.append(success)
            
            print(f"  Result: {'✅' if success else '❌'}")
        
        success_rate = np.mean(results)
        print(f"\nCompositional task success: {success_rate*100:.1f}%")
        
        return success_rate

# Few-shot adaptation
class FewShotAdapter:
    """
    Few-shot learning
    
    몇 개의 example만으로 새 task 학습
    """
    
    def __init__(self, pretrained_model):
        self.model = pretrained_model
        
        # Freeze most layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Only fine-tune task head
        self.task_head = nn.Linear(512, 7)
        self.task_head.requires_grad = True
    
    def adapt(self, demonstrations, num_epochs=10):
        """
        Few-shot adaptation
        
        Args:
            demonstrations: 5-10 examples
            num_epochs: quick fine-tuning
        """
        print(f"Adapting with {len(demonstrations)} demonstrations...")
        
        # Create mini dataset
        dataset = FewShotDataset(demonstrations)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Optimizer (only task head)
        optimizer = torch.optim.Adam(self.task_head.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Quick fine-tuning
        for epoch in range(num_epochs):
            for obs, action in dataloader:
                # Extract features (frozen)
                with torch.no_grad():
                    features = self.model.extract_features(obs)
                
                # Task head (trainable)
                pred_action = self.task_head(features)
                
                loss = criterion(pred_action, action)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        print("✅ Adaptation complete")
    
    def evaluate(self, test_env):
        """
        Evaluate adapted model
        """
        # Test on new task
        success_rate = run_evaluation(self.model, test_env)
        
        return success_rate
````

**시간: 주 6-8시간**

---

## Month 15-16: 프로젝트 3 - Visual SLAM Integration

### Week 1-2: ORB-SLAM3 Integration
````python
# orbslam_integration.py

class ORBSLAMIntegration:
    """
    ORB-SLAM3 + VLA 통합
    
    목적:
    - Real-time localization
    - Map building
    - Navigation with manipulation
    
    SLAM 관심사 활용!
    """
    
    def __init__(self, vla_model):
        self.vla_model = vla_model
        
        # ORB-SLAM3 초기화
        self.slam_system = self.initialize_orbslam()
        
        # Map
        self.map = None
        self.current_pose = None
    
    def initialize_orbslam(self):
        """
        ORB-SLAM3 초기화
        """
        import ORB_SLAM3 as orbslam
        
        # Vocabulary and settings
        vocab_path = "ORB-SLAM3/Vocabulary/ORBvoc.txt"
        settings_path = "config/camera.yaml"
        
        # Create SLAM system
        slam = orbslam.System(
            vocab_path,
            settings_path,
            orbslam.Sensor.RGBD  # RGB-D camera
        )
        
        return slam
    
    def process_frame(self, rgb, depth, timestamp):
        """
        Process single frame
        
        Args:
            rgb: RGB image
            depth: Depth image
            timestamp: Frame timestamp
        
        Returns:
            pose: 4x4 transformation matrix
        """
        # Track frame
        pose = self.slam_system.track_rgbd(
            rgb, depth, timestamp
        )
        
        if pose is not None:
            self.current_pose = pose
            
            # Update map
            self.update_map()
        
        return pose
    
    def update_map(self):
        """
        Update map from SLAM
        """
        # Get map points
        map_points = self.slam_system.get_map_points()
        
        # Get keyframes
        keyframes = self.slam_system.get_keyframes()
        
        # Update internal map
        self.map = {
            'points': map_points,
            'keyframes': keyframes,
            'current_pose': self.current_pose
        }
    
    def get_object_pose_in_map(self, object_detection):
        """
        Object의 map 좌표
        
        Args:
            object_detection: 2D bounding box + depth
        
        Returns:
            object_pose: 3D pose in map frame
        """
        # Extract 3D point from depth
        u, v = object_detection['center']
        depth = object_detection['depth']
        
        # Camera intrinsics
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']
        
        # Back-project to 3D
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        point_camera = np.array([x, y, z, 1.0])
        
        # Transform to map frame
        point_map = self.current_pose @ point_camera
        
        return point_map[:3]
    
    def plan_navigation_to_object(self, object_id):
        """
        Object까지 navigation plan
        
        Returns:
            waypoints: list of poses
        """
        # Get object pose
        object_pose = self.map['objects'][object_id]['pose']
        
        # Current robot pose
        robot_pose = self.current_pose
        
        # Plan path (A* on occupancy grid)
        path = self.plan_path(robot_pose, object_pose)
        
        return path
    
    def execute_navigation_and_manipulation(self, target_object):
        """
        Complete task: Navigate + Manipulate
        
        Workflow:
        1. Localize with SLAM
        2. Detect target object
        3. Get object pose in map
        4. Navigate to object
        5. Execute VLA manipulation
        """
        print(f"\n{'='*60}")
        print(f"Task: Navigate and pick {target_object}")
        print(f"{'='*60}")
        
        # 1. Continuous SLAM
        while not self.is_map_ready():
            rgb, depth = self.get_camera_frames()
            self.process_frame(rgb, depth, time.time())
            time.sleep(0.1)
        
        print("✅ Map ready")
        
        # 2. Object detection
        object_detections = self.detect_objects()
        target_detection = [d for d in object_detections 
                           if d['class'] == target_object][0]
        
        # 3. Get object pose in map
        object_pose_map = self.get_object_pose_in_map(target_detection)
        
        print(f"Object at: {object_pose_map}")
        
        # 4. Navigate
        waypoints = self.plan_navigation_to_object(target_detection)
        
        for waypoint in waypoints:
            self.navigate_to_pose(waypoint)
            
            # Continue SLAM during navigation
            rgb, depth = self.get_camera_frames()
            self.process_frame(rgb, depth, time.time())
        
        print("✅ Navigation complete")
        
        # 5. VLA manipulation
        print("Executing manipulation...")
        
        # Get current observation
        obs = self.get_vla_observation()
        
        # VLA inference
        success = self.execute_vla(obs)
        
        if success:
            print("✅ Manipulation successful")
        else:
            print("❌ Manipulation failed")
        
        return success

# Map visualization
def visualize_slam_map(slam_map):
    """
    SLAM map 시각화
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Map points
    points = slam_map['points']
    
    if len(points) > 0:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        
        ax.scatter(xs, ys, zs, c='gray', marker='.', s=1, alpha=0.5)
    
    # Keyframes (camera poses)
    keyframes = slam_map['keyframes']
    
    for kf in keyframes:
        pose = kf['pose']
        position = pose[:3, 3]
        
        # Draw camera
        ax.scatter(*position, c='blue', marker='o', s=50)
        
        # Draw orientation
        forward = pose[:3, 2] * 0.1
        ax.quiver(*position, *forward, color='red', length=0.1)
    
    # Current pose
    current = slam_map['current_pose']
    position = current[:3, 3]
    ax.scatter(*position, c='green', marker='*', s=200, label='Current')
    
    # Objects
    if 'objects' in slam_map:
        for obj_id, obj in slam_map['objects'].items():
            obj_pos = obj['pose']
            ax.scatter(*obj_pos, c='red', marker='x', s=100)
            ax.text(*obj_pos, obj['class'], fontsize=8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('SLAM Map')
    ax.legend()
    
    plt.show()
````

**시간: 주 8-10시간**

---

## Month 17: 포트폴리오 & 블로그

### Week 1-2: GitHub 포트폴리오

#### Repository 구조
````
vla-logistics-robot/
├── README.md
├── docs/
│   ├── installation.md
│   ├── quickstart.md
│   ├── architecture.md
│   └── results.md
├── src/
│   ├── models/
│   │   ├── act_policy.py
│   │   ├── diffusion_policy.py
│   │   └── language_conditioned_vla.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── dataset.py
│   │   └── augmentation.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── ros2/
│   │   ├── vla_node.py
│   │   ├── safety_layer.py
│   │   └── failure_recovery.py
│   └── utils/
│       ├── visualization.py
│       └── logging.py
├── configs/
│   ├── act_config.yaml
│   ├── training_config.yaml
│   └── robot_config.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── collect_data.py
│   └── deploy.py
├── tests/
│   ├── test_model.py
│   ├── test_training.py
│   └── test_ros2.py
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── model_comparison.ipynb
│   └── results_visualization.ipynb
├── assets/
│   ├── demo.gif
│   ├── architecture.png
│   └── results/
├── requirements.txt
├── setup.py
└── LICENSE
````

---

#### README.md 작성
````markdown
# VLA for Logistics Robot Manipulation

<p align="center">
  <img src="assets/demo.gif" width="600">
</p>

## 🎯 Overview

Vision-Language-Action (VLA) model for autonomous logistics robot manipulation. Achieves **75% success rate** on multi-task pick-and-place operations in simulation and **68% on real robot**.

**Key Features:**
- 🤖 Multi-task learning (pick, place, stack, sort, drawer)
- 🗣️ Language-conditioned control
- 🔄 ROS2 integration with Lifecycle management
- 🛡️ Safety layer and failure recovery
- 📊 Comprehensive evaluation framework

## 📹 Demo

| Pick | Place | Stack |
|------|-------|-------|
| ![pick](assets/pick.gif) | ![place](assets/place.gif) | ![stack](assets/stack.gif) |

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" width="800">
</p>

### Model

- **Vision Encoder**: ViT-Base (pre-trained on ImageNet)
- **Language Encoder**: BERT-Base
- **Policy**: Action Chunking Transformer (ACT)
- **Action Space**: Delta joint positions (7-DOF)
- **Observation**: RGB (224x224) + Proprioception (15-dim)

### Pipeline
Image + Language → Vision-Language Fusion → Action Decoder → Robot Control

# 🚀 Quick Start
## Installation

# Clone repository
git clone https://github.com/yourusername/vla-logistics-robot.git
cd vla-logistics-robot

# Install dependencies
pip install -r requirements.txt

# Install ROS2 packages (optional)
colcon build
source install/setup.bash

Training
# Train ACT policy
python scripts/train.py --config configs/act_config.yaml

# Multi-task training
python scripts/train.py --config configs/multitask_config.yaml --tasks pick place stack

Evaluation
# Evaluate in simulation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --num_episodes 100

# Real robot evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --real_robot

# ROS2 Deployment

## Launch VLA node
ros2 launch vla_control vla_bringup.launch.py model_path:=/path/to/model.pt

## Control via lifecycle
ros2 lifecycle set /vla_node configure
ros2 lifecycle set /vla_node activate


## 📊 Results

### Simulation Performance

| Task | Success Rate | Avg Time | Smoothness |
|------|--------------|----------|------------|
| Pick | 82% | 4.2s | 0.94 |
| Place | 78% | 5.1s | 0.91 |
| Stack | 65% | 7.8s | 0.87 |
| Sort | 73% | 12.3s | 0.89 |
| Drawer | 68% | 6.5s | 0.85 |
| **Average** | **73%** | **7.2s** | **0.89** |

### Real Robot Performance

| Task | Success Rate | Sim-Real Gap |
|------|--------------|--------------|
| Pick | 75% | -7% |
| Place | 71% | -7% |
| Stack | 58% | -7% |
| **Average** | **68%** | **-7%** |

### Generalization

- **Novel objects**: 65% (tested on unseen colors/sizes)
- **Novel instructions**: 71% (synonym variations)
- **Novel environments**: 62% (different warehouse layouts)

## 🛠️ Technical Details

### Domain Randomization

- Physics: Gravity (±5%), Friction (0.3-1.5x), Mass (0.8-1.2x)
- Visuals: Lighting (2000-8000 lux), Color temperature (3000-7000K)
- Sensors: Camera noise (σ=0.05), Proprioception noise (σ=0.01)
- Actuation: Motor noise (σ=0.02), Random delays (10% @ 1 step)

### Safety Features

- Joint limit enforcement
- Velocity limiting (< 2.0 rad/s)
- Workspace boundaries
- Collision detection
- Emergency stop

### Optimization

| Method | Inference Time | Speedup |
|--------|----------------|---------|
| Original (FP32) | 85ms | 1.0x |
| TorchScript | 52ms | 1.6x |
| Quantized (INT8) | 38ms | 2.2x |
| TensorRT (FP16) | 26ms | 3.3x |

## 📝 Publications

- [Technical Blog Post](link)
- [Medium Article](link)
- [Paper (if any)](link)

## 🙏 Acknowledgments

- ORB-SLAM3 for SLAM system
- LeRobot for VLA codebase inspiration
- RT-1/RT-2 papers for architecture ideas

## 📄 License

MIT License

## 📧 Contact

- Email: your.email@example.com
- LinkedIn: [Your Name](link)
- Website: [yourwebsite.com](link)

---

**⭐ If you find this project useful, please consider giving it a star!**
````

**시간: 주 6-8시간**

---

### Week 3-4: 기술 블로그

#### 블로그 주제들
````markdown
## 블로그 시리즈: VLA 개발 여정

### 1. "VLA란 무엇인가: Vision-Language-Action 모델 입문"
- VLA 개념 소개
- 기존 방법 (BC, RL) vs VLA
- 대표 논문 (RT-1, RT-2, OpenVLA)
- 실제 응용 사례

**예상 조회수: 1000+**

---

### 2. "Isaac Sim으로 로봇 시뮬레이션 환경 구축하기"
- Isaac Sim 소개
- 물류 창고 환경 모델링
- Domain Randomization
- 실습 코드

**난이도: 중급**

---

### 3. "Action Space 설계의 중요성: Delta vs Absolute"
- Action space 종류
- 각 방법의 장단점
- 실험 결과 비교
- 실전 팁

**독자 타겟: ML/Robotics 엔지니어**

---

### 4. "ROS2 Lifecycle으로 안전한 로봇 시스템 만들기"
- Lifecycle 패턴 설명
- VLA Node 구현
- Safety layer 통합
- 실제 적용 사례

**ROS2 경험 어필!**

---

### 5. "Sim-to-Real Transfer: 시뮬레이션에서 실제 로봇으로"
- Reality Gap 이란?
- Domain Randomization 전략
- 실험 결과 분석
- 극복 방법

**가장 중요한 주제!**

---

### 6. "Multi-Task VLA: 하나의 모델로 여러 작업 수행하기"
- Multi-task learning
- Language conditioning
- Task balancing
- Zero-shot generalization 결과

---

### 7. "VLA 모델 최적화: TensorRT로 3배 빠르게"
- 추론 속도 중요성
- 최적화 기법들
- 벤치마크 결과
- 실전 배포 팁

---

### 8. "실패에서 배운 것들: VLA 개발 시행착오"
- 초기 실패 사례
- 디버깅 과정
- 해결 방법
- 배운 교훈

**솔직한 회고, 공감 유발!**
````

---

#### 블로그 글 예시
````markdown
# VLA 모델 최적화: TensorRT로 3배 빠르게

## 🎯 왜 최적화가 필요한가?

VLA 모델을 실제 로봇에 배포하려면 **실시간 제어**가 필수입니다. 

- 목표: 10Hz 제어 (100ms 이내)
- 문제: 원본 모델 추론 시간 = 85ms
- 해결: 최적화로 26ms까지 단축 ✨

## 📊 최적화 전후 비교
```
Method          Inference Time    Speedup    Success Rate
Original (FP32)     85ms            1.0x         73%
TorchScript         52ms            1.6x         73%
Quantized (INT8)    38ms            2.2x         71%
TensorRT (FP16)     26ms            3.3x         72%
```

## 🔧 최적화 방법

### 1. TorchScript

가장 간단한 최적화!
```python
# 모델 trace
model.eval()
dummy_input = {
    'rgb': torch.randn(1, 3, 224, 224).cuda(),
    'proprio': torch.randn(1, 15).cuda()
}

scripted_model = torch.jit.trace(model, dummy_input)
scripted_model.save('model_scripted.pt')
```

**효과:**
- 속도: 1.6배 향상
- 정확도: 변화 없음
- 추천도: ⭐⭐⭐⭐

### 2. Quantization

모델 크기 1/4, 속도 2배!
```python
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

**효과:**
- 속도: 2.2배
- 모델 크기: 500MB → 125MB
- 정확도: -2% (허용 범위)
- 추천도: ⭐⭐⭐⭐

### 3. TensorRT (최고 성능!)

NVIDIA GPU 전용, 최고 속도!
```python
from torch2trt import torch2trt

model_trt = torch2trt(
    model,
    [dummy_input],
    fp16_mode=True
)
```

**효과:**
- 속도: 3.3배
- 정확도: -1%
- 추천도: ⭐⭐⭐⭐⭐

## 📈 실전 배포 결과

TensorRT 적용 후:
- ✅ 10Hz 실시간 제어 달성
- ✅ GPU 메모리 50% 절감
- ✅ 배터리 수명 20% 증가

## 💡 팁

1. **개발**: 원본 모델 사용 (디버깅 용이)
2. **테스트**: TorchScript (속도 + 안정성)
3. **배포**: TensorRT (최고 성능)

## 🎓 배운 점

- 최적화는 무조건 해야 함!
- 1-2% 정확도 손실은 충분히 허용
- 속도가 3배 빠르면 사용자 경험 완전히 달라짐

---

**전체 코드:** [GitHub](link)

**질문/피드백:** 댓글로 환영합니다! 🙌
````

**시간: 주 8-10시간 (글 3-4개)**

---

## Month 18: 취업 준비

### Week 1-2: 이력서 & 포트폴리오 정리

#### 이력서 작성
````markdown
# AI Perception Engineer 이력서

## 👤 프로필

**이름**: tylee
**경력**: ROS/ROS2 개발 5년+, 로봇 엔지니어
**목표**: AI Perception Engineer로 전환

**핵심 역량**:
- Vision-Language-Action (VLA) 모델 개발
- ROS2 시스템 설계 및 통합
- Isaac Sim 시뮬레이션
- Deep Learning (PyTorch)
- Visual SLAM (ORB-SLAM3)

---

## 💼 경력

### 로봇 엔지니어 | 물류 회사 | 2019 - 현재

**주요 업무**:
- 물류 로봇 ROS2 Application 개발
- 센서 융합 (LiDAR, IMU, Wheel Odometry)
- EKF 기반 위치 추정
- 현장 배포 및 유지보수

**성과**:
- 로봇 자율주행 정확도 15% 향상
- ROS2 Lifecycle 패턴 도입으로 안정성 30% 개선
- 센서 융합 알고리즘 최적화로 응답 속도 2배 향상

---

## 🚀 프로젝트

### 1. VLA for Logistics Robot (2024)

**개요**: 물류 로봇 manipulation을 위한 Vision-Language-Action 모델

**기술 스택**:
- PyTorch, Isaac Sim, ROS2, CUDA
- ViT, BERT, Transformer
- Domain Randomization

**성과**:
- 시뮬레이션 성공률: 73% (5개 task 평균)
- 실제 로봇 성공률: 68%
- Sim-to-Real gap: 7% (업계 평균 대비 우수)
- 추론 속도: 26ms (TensorRT 최적화)

**주요 기여**:
- Action Space 설계 및 비교 실험
- ROS2 Lifecycle 기반 안전 시스템 구현
- Domain Randomization 전략 수립
- Multi-task learning 파이프라인 구축

**링크**: [GitHub](link) | [Demo](link) | [Blog](link)

---

### 2. Multi-Task VLA with Language Conditioning (2024)

**개요**: 언어 명령으로 제어되는 Multi-task VLA

**주요 기능**:
- 5개 task (pick, place, stack, sort, drawer)
- Zero-shot generalization (65%)
- Few-shot adaptation (10 examples)

**기술적 도전**:
- FiLM layer로 vision-language fusion
- Task balancing 및 curriculum learning
- Compositional task 처리

---

### 3. SLAM-VLA Integration (2024)

**개요**: ORB-SLAM3와 VLA 통합 시스템

**기능**:
- Real-time localization & mapping
- Object pose estimation in map frame
- Navigation + Manipulation

**성과**:
- End-to-end task 성공률: 62%
- SLAM tracking 정확도: 2cm RMSE

---

## 🎓 교육

### 학사 | 기계공학 | 대학교 | 2015-2019

**관련 과목**:
- 로봇공학, 제어이론, 컴퓨터비전
- 선형대수, 확률통계, 최적화

---

## 📚 학습 & 역량

### Deep Learning
- PyTorch 숙련 (모델 설계, 학습, 최적화)
- CNN, Transformer, Diffusion Models
- TensorRT, ONNX 최적화

### Computer Vision
- Object Detection (YOLO, Faster R-CNN)
- Semantic Segmentation (U-Net)
- Visual SLAM (ORB-SLAM3)

### Robotics
- ROS2 (Lifecycle, Action, Diagnostics)
- Isaac Sim, Gazebo
- Kinematics, Dynamics, Control

### Mathematics
- Linear Algebra (eigenvalue, SVD, PCA)
- Probability & Statistics
- Optimization (Gradient Descent, Adam)

---

## 🏆 성과 & 인증

- GitHub Stars: 100+ (VLA 프로젝트)
- 기술 블로그 조회수: 5000+
- 논문 재현: RT-1, ACT

---

## 🔗 링크

- GitHub: [github.com/tylee](link)
- Blog: [blog.tylee.com](link)
- LinkedIn: [linkedin.com/in/tylee](link)
- Email: tylee@example.com
````

---

### Week 3-4: 면접 준비

#### 예상 질문 & 답변
````markdown
## 기술 면접 예상 질문

### 1. VLA 관련

**Q: VLA와 기존 Imitation Learning의 차이는?**

A: VLA는 세 가지 주요 차이가 있습니다:

1. **Multi-modal**: Vision + Language + Proprioception
   - 기존 IL: 주로 vision만
   - VLA: Language로 task conditioning 가능

2. **Generalization**: 
   - 기존 IL: Task-specific
   - VLA: Zero-shot, Few-shot 가능

3. **Scale**:
   - VLA: 대규모 데이터로 pre-training
   - 더 robust한 policy

제 프로젝트에서는 language conditioning으로 5개 task를 하나의 모델로 처리했고, 
novel instruction에 대해 71% 성공률을 달성했습니다.

---

**Q: Sim-to-Real gap을 어떻게 줄였나요?**

A: 세 가지 전략을 사용했습니다:

1. **Domain Randomization**:
   - Physics: Gravity, Friction, Mass
   - Visuals: Lighting, Colors
   - Sensors: Camera noise, Joint noise

2. **Real Data Fine-tuning**:
   - Simulation 학습 후
   - Real robot 데이터 50 에피소드로 fine-tuning
   - Gap 15% → 7%로 감소

3. **Calibration**:
   - Camera intrinsics/extrinsics
   - Robot kinematics
   - Action scaling

결과: Sim 73% → Real 68% (7% gap)

---

**Q: Action Space를 어떻게 설계했나요?**

A: 세 가지 옵션을 비교 실험했습니다:

1. **Absolute Joint**: 직접 제어, 불안정
2. **Delta Joint** (채택): 
   - 안정적 학습
   - Smooth trajectory
   - 70% 성공률

3. **Cartesian**: 직관적이지만 IK 오차

Delta Joint를 선택한 이유:
- Action chunking과 궁합 좋음
- Safety constraints 적용 용이
- Simulation→Real 전이 우수

---

### 2. ROS2 관련

**Q: ROS2 Lifecycle을 왜 사용했나요?**

A: 안전성과 재현성 때문입니다:

1. **State Management**:
   - Configure → Activate → Deactivate
   - 각 state에서 resource 관리
   - 예: Configure에서 model 로드

2. **Failure Handling**:
   - Error 시 자동 cleanup
   - 안전한 재시작

3. **System Integration**:
   - 여러 node 동기화
   - Orchestration 용이

제 VLA node는:
- Configure: Model loading
- Activate: Control loop start
- Deactivate: Safe stop
- Safety incidents 0건 달성

---

**Q: Diagnostics를 어떻게 활용했나요?**

A: 실시간 모니터링과 디버깅에 사용했습니다:
```python
def diagnostic_callback(self, stat):
    # 주요 metrics
    stat.add("Success Rate", f"{self.success_rate:.1%}")
    stat.add("Inference Time", f"{self.inference_time:.1f}ms")
    stat.add("Safety Violations", str(self.violations))
    
    # Status 결정
    if self.inference_time > 100:
        stat.summary(WARN, "Slow inference")
    else:
        stat.summary(OK, "Normal")
```

현장에서 문제 조기 발견에 매우 유용했습니다.

---

### 3. Deep Learning 관련

**Q: Transformer를 왜 사용했나요?**

A: Sequence modeling에 최적이기 때문입니다:

1. **Action Chunking**:
   - 10 timestep을 한 번에 예측
   - Temporal consistency 향상
   - Trajectory가 smooth

2. **Attention Mechanism**:
   - 중요한 visual feature에 집중
   - Long-range dependency

3. **Scalability**:
   - Pre-training 가능
   - Multi-task에 유리

실험 결과:
- MLP: 55% success
- LSTM: 63%
- Transformer (ACT): 73%

---

**Q: 모델 최적화는 어떻게 했나요?**

A: 

1. **TorchScript**: 85ms → 52ms
2. **Quantization**: 52ms → 38ms
3. **TensorRT**: 38ms → 26ms

Trade-off:
- 속도: 3.3배 향상
- 정확도: -1% (허용 범위)
- 메모리: 1/2

Production에서 TensorRT 사용 중이며,
10Hz 실시간 제어 달성했습니다.

---

### 4. 문제 해결 관련

**Q: 가장 어려웠던 기술적 도전은?**

A: Action Space 설계였습니다.

**문제**: 
- 초기 Absolute Joint: 성공률 30%
- 학습 불안정, 에러 누적

**시도**:
1. Normalization 튜닝 → 효과 없음
2. Network capacity 증가 → 여전히 불안정
3. Delta action으로 변경 → 성공!

**해결**:
- Delta joint (±0.1 rad limit)
- Action chunking (10 steps)
- 성공률 30% → 70%

**배운 점**:
- Action space가 모델만큼 중요
- 실험과 비교가 핵심
- Domain knowledge 활용

---

**Q: Failure case를 어떻게 분석했나요?**

A: 체계적 분류와 해결:

**Failure Types**:
1. Grasp failure (40%)
2. Collision (25%)
3. Trajectory deviation (20%)
4. Timeout (15%)

**해결**:
- Grasp: 더 많은 grasp data
- Collision: Safety constraints 강화
- Deviation: Action chunking 증가

결과: Overall failure 30% → 15%

---

### 5. Soft Skills

**Q: 혼자 프로젝트를 진행하면서 어려운 점은?**

A: 

**어려움**:
- Motivation 유지
- 방향성 결정
- 막힐 때 해결

**해결**:
- 주간 목표 설정
- 커뮤니티 활용 (Reddit, Discord)
- 블로그 작성 (정리 + 피드백)

**성과**:
- 18개월 프로젝트 완수
- 블로그 5000+ 조회
- GitHub 100+ stars

---

**Q: 앞으로의 계획은?**

A: 

**단기 (6개월)**:
- VLA 실제 배포 경험
- Multi-modal learning 심화
- 논문 작성/발표

**장기 (2-3년)**:
- Embodied AI 전문가
- Large-scale VLA 연구
- Open-source 기여

이 회사에서:
- 제 ROS 경험 + AI 역량 결합
- 실제 제품에 AI 적용
- 팀과 협업하며 성장
````

**시간: 주 10-12시간**

---

## Phase 4 완료 체크
````
✅ 프로젝트 3개 완성
  ├─ VLA for Logistics
  ├─ Multi-Task VLA
  └─ SLAM-VLA Integration

✅ GitHub 포트폴리오
  ├─ 깔끔한 README
  ├─ Documentation
  └─ 100+ stars 목표

✅ 기술 블로그
  ├─ 8개 포스트
  ├─ 5000+ 조회수
  └─ 기술적 깊이 + 실용성

✅ 이력서 & 면접 준비
  ├─ 프로젝트 중심 이력서
  ├─ 기술 질문 대비
  └─ 스토리 준비

✅ 네트워킹
  ├─ LinkedIn 활성화
  ├─ 컨퍼런스 참석
  └─ 커뮤니티 기여

→ AI Perception Engineer 이직 준비 완료! 🎉
````

---

## 최종 로드맵 요약
````
전체 일정: 18개월

Phase 0 (1-2개월): Top-Down 돌파
→ VLA 감 잡기, Mini VLA, RT-1 이해

Phase 1 (3-6개월): Bottom-Up 기초
→ DL, CNN, Transformer, Multi-modal

Phase 2 (7-12개월): 본격 프로젝트
→ Isaac Sim, Action/Obs 설계, VLA 학습
→ ROS2 통합, Sim-to-Real, 최적화

Phase 4 (13-18개월): 포트폴리오
→ 프로젝트 2&3, 블로그, 이직 준비

최종 성과:
- VLA 전문성 확보
- 3개 완성도 높은 프로젝트
- 기술 블로그 & GitHub
- AI Perception Engineer 취업!
````
