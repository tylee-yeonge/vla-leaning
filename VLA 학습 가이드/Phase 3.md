# VLA 학습 가이드 - Phase 3

## 목차
- [📅 Phase 3: Isaac Sim 본격 프로젝트 (7-12개월)](#-phase-3-isaac-sim-본격-프로젝트-7-12개월)
- [Month 7-8: Isaac Sim & ROS2 통합](#month-7-8-isaac-sim--ros2-통합)
  - [Week 1-2: Isaac Sim 기초](#week-1-2-isaac-sim-기초)
  - [Week 3-4: 로봇 제어](#week-3-4-로봇-제어)
  - [Week 5-8: 물류 환경 구축](#week-5-8-물류-환경-구축)
- [Month 9: Action & Observation Space 설계](#month-9-action--observation-space-설계)
  - [Week 1: Action Space 설계 ⚠️ 매우 중요!](#week-1-action-space-설계-️-매우-중요)
  - [Week 2: Observation Space 설계](#week-2-observation-space-설계)
- [Month 10: 첫 물류 VLA 개발](#month-10-첫-물류-vla-개발)
  - [Week 1-2: 데이터 수집 & 품질 관리](#week-1-2-데이터-수집--품질-관리)
  - [Week 3-4: VLA 학습](#week-3-4-vla-학습)
  - [Week 5-6: 평가 및 디버깅](#week-5-6-평가-및-디버깅)
- [Month 11-12: 고도화 및 ROS2 통합](#month-11-12-고도화-및-ros2-통합)
  - [Week 1-2: 실패 복구 시스템](#week-1-2-실패-복구-시스템)
  - [Week 3-4: Safety Layer](#week-3-4-safety-layer)
  - [Week 5-6: ROS2 완전 통합](#week-5-6-ros2-완전-통합)
  - [Week 7-8: Sim-to-Real Transfer 준비](#week-7-8-sim-to-real-transfer-준비)
- [Real Robot Deployment Checklist](#real-robot-deployment-checklist)
- [Phase 3 완료 체크](#phase-3-완료-체크)

## 📅 Phase 3: Isaac Sim 본격 프로젝트 (7-12개월)

### 목표
- Isaac Sim 환경 마스터
- 물류 로봇 VLA 개발
- ROS2 완전 통합
- 현장 적용 가능한 수준
- Sim-to-Real 준비

---

## Month 7-8: Isaac Sim & ROS2 통합

### Week 1-2: Isaac Sim 기초

#### 설치 및 환경 구축
```bash
# Isaac Sim 설치 (공식 가이드 따라)
# https://docs.omniverse.nvidia.com/app_isaacsim/

# 시스템 요구사항:
# - GPU: RTX 4070 (충분!)
# - RAM: 32GB (권장)
# - Storage: 50GB+

# ROS2 연동 확인
# Ubuntu 22.04 + ROS2 Humble
```

---

#### 기본 튜토리얼 완주

**Hello World**
```python
# hello_world.py
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

# Create world
world = World()

# Add cube
cube = DynamicCuboid(
    prim_path="/World/Cube",
    position=[0, 0, 0.5],
    size=0.1,
    color=[1.0, 0.0, 0.0]
)

# Simulation loop
for i in range(1000):
    world.step(render=True)
    
    # Print cube position
    if i % 100 == 0:
        position, _ = cube.get_world_pose()
        print(f"Step {i}: Position = {position}")

simulation_app.close()
```

**체크:**
- [ ] Isaac Sim GUI 열림
- [ ] Cube가 떨어짐
- [ ] 물리 시뮬레이션 작동

---

#### 학습 주제

**1. USD (Universal Scene Description)**
```python
"""
USD는 Isaac Sim의 기본 scene 포맷

핵심 개념:
- Prim: Scene의 기본 단위 (object, light, camera 등)
- Stage: 모든 Prim을 담는 컨테이너
- Layer: Scene의 hierarchical composition

예: /World/Robot/Link1
"""

# USD 직접 조작
from pxr import Usd, UsdGeom

stage = omni.usd.get_context().get_stage()

# Create sphere
sphere_prim = UsdGeom.Sphere.Define(stage, "/World/Sphere")
sphere_prim.GetRadiusAttr().Set(0.5)

# Set position
sphere_prim.AddTranslateOp().Set((0, 0, 1))
```

**2. Physics Simulation 설정**
```python
from omni.isaac.core.utils.physics import simulate_async

# Physics scene 설정
scene = world.scene

# Gravity
scene.set_gravity([0, 0, -9.81])

# Physics parameters
world.get_physics_context().set_solver_type("TGS")  # Temporal Gauss-Seidel
world.get_physics_context().set_broadphase_type("GPU")

# Time step
world.set_simulation_dt(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
```

**3. 렌더링 설정**
```python
# Rendering quality
import carb

settings = carb.settings.get_settings()

# Ray tracing
settings.set("/rtx/raytracing/enabled", True)
settings.set("/rtx/pathtracing/spp", 16)  # Samples per pixel

# Post-processing
settings.set("/rtx/post/aa/op", 2)  # Anti-aliasing
settings.set("/rtx/post/dlss/execMode", 1)  # DLSS
```

**시간: 주 6-8시간**

---

### Week 3-4: 로봇 제어

#### 모바일 베이스 + 매니퓰레이터
```python
# mobile_manipulator.py
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.core.articulations import Articulation

class MobileManipulator:
    """
    모바일 베이스 + 로봇 팔 통합
    """
    
    def __init__(self, world):
        self.world = world
        
        # Load mobile base
        self.setup_mobile_base()
        
        # Load manipulator
        self.setup_manipulator()
        
        # Controllers
        self.setup_controllers()
    
    def setup_mobile_base(self):
        """
        Differential drive mobile base
        """
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        
        # Add robot to scene
        self.base = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/MobileBase",
                name="mobile_base",
                wheel_dof_names=["wheel_left_joint", "wheel_right_joint"],
                create_robot=True,
                usd_path="path/to/mobile_base.usd"
            )
        )
        
        # Controller
        self.base_controller = DifferentialController(
            name="base_controller",
            wheel_radius=0.1,
            wheel_base=0.5
        )
    
    def setup_manipulator(self):
        """
        6-DOF manipulator (e.g., UR5)
        """
        self.arm = self.world.scene.add(
            SingleManipulator(
                prim_path="/World/MobileBase/UR5",
                name="ur5",
                end_effector_prim_name="tool0",
                usd_path="path/to/ur5.usd"
            )
        )
        
        # Gripper
        from omni.isaac.manipulators.grippers import ParallelGripper
        
        self.gripper = self.world.scene.add(
            ParallelGripper(
                prim_path="/World/MobileBase/UR5/gripper",
                name="gripper"
            )
        )
    
    def setup_controllers(self):
        """
        Setup arm controller
        """
        from omni.isaac.manipulators.controllers import PickPlaceController
        
        self.arm_controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self.gripper,
            robot_articulation=self.arm
        )
    
    def move_base(self, linear_velocity, angular_velocity):
        """
        Control mobile base
        
        Args:
            linear_velocity: m/s
            angular_velocity: rad/s
        """
        wheel_actions = self.base_controller.forward(
            command=[linear_velocity, angular_velocity]
        )
        self.base.apply_wheel_actions(wheel_actions)
    
    def move_arm(self, target_position, target_orientation=None):
        """
        Move arm to target pose
        
        Args:
            target_position: [x, y, z]
            target_orientation: [qx, qy, qz, qw] or None
        """
        actions = self.arm_controller.forward(
            picking_position=target_position,
            placing_position=None,
            current_joint_positions=self.arm.get_joint_positions()
        )
        self.arm.apply_action(actions)
    
    def control_gripper(self, action):
        """
        Control gripper
        
        Args:
            action: "open" or "close"
        """
        if action == "open":
            self.gripper.open()
        elif action == "close":
            self.gripper.close()

# 사용 예시
world = World()
robot = MobileManipulator(world)

# Simulation loop
for i in range(1000):
    # Move forward
    robot.move_base(linear_velocity=0.5, angular_velocity=0.0)
    
    # Move arm
    if i == 100:
        robot.move_arm(target_position=[0.5, 0.0, 0.5])
    
    if i == 200:
        robot.control_gripper("close")
    
    world.step(render=True)
```

---

#### ROS2 통합
```python
# ros2_integration.py
from omni.isaac.core.utils.extensions import enable_extension

# Enable ROS2 bridge
enable_extension("omni.isaac.ros2_bridge")

from omni.isaac.ros2_bridge import ROS2Bridge
import rclpy

class ROS2IntegratedRobot:
    """
    Isaac Sim ↔ ROS2 브릿지
    """
    
    def __init__(self, robot):
        self.robot = robot
        self.bridge = ROS2Bridge()
        
        # Initialize ROS2
        rclpy.init()
        
        # Setup publishers/subscribers
        self.setup_ros2_interface()
    
    def setup_ros2_interface(self):
        """
        ROS2 topics 설정
        """
        # 1. Camera publisher
        self.camera_pub = self.bridge.add_camera_publisher(
            topic_name="/camera/image_raw",
            camera_prim_path="/World/Robot/Camera",
            message_type="sensor_msgs/Image",
            frame_id="camera_link"
        )
        
        # 2. Joint state publisher
        self.joint_state_pub = self.bridge.add_joint_state_publisher(
            topic_name="/joint_states",
            robot_prim_path="/World/Robot"
        )
        
        # 3. Odometry publisher
        self.odom_pub = self.bridge.add_odometry_publisher(
            topic_name="/odom",
            chassis_prim_path="/World/Robot/base_link"
        )
        
        # 4. Twist subscriber (velocity commands)
        self.twist_sub = self.bridge.add_twist_subscriber(
            topic_name="/cmd_vel",
            callback=self.twist_callback
        )
        
        # 5. Joint command subscriber
        self.joint_cmd_sub = self.bridge.add_subscriber(
            topic_name="/joint_commands",
            msg_type="sensor_msgs/JointState",
            callback=self.joint_callback
        )
    
    def twist_callback(self, msg):
        """
        Velocity command callback
        """
        linear = msg.linear.x
        angular = msg.angular.z
        
        self.robot.move_base(linear, angular)
    
    def joint_callback(self, msg):
        """
        Joint command callback
        """
        positions = msg.position
        self.robot.arm.set_joint_positions(positions)

# 사용 예시
robot = MobileManipulator(world)
ros2_robot = ROS2IntegratedRobot(robot)

# Simulation loop
while simulation_app.is_running():
    world.step(render=True)
    
    # ROS2 spin (process callbacks)
    rclpy.spin_once(ros2_robot.bridge.node, timeout_sec=0.0)
```

---

#### ROS2 패턴
```python
# lifecycle_robot_node.py
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from rclpy.lifecycle import Publisher, LifecycleState

class LifecycleRobotNode(Node):
    """
    Lifecycle 패턴 활용
    
   기존 ROS2 경험 활용:
    - Lifecycle management
    - Diagnostics
    - tf2 transforms
    """
    
    def __init__(self, robot):
        super().__init__('vla_robot_node', enable_communication_interface=True)
        self.robot = robot
    
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """
        Configure state: Setup resources
        """
        self.get_logger().info('Configuring robot...')
        
        # Setup publishers
        self.cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10
        )
        
        # Setup subscribers
        self.vla_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self.vla_callback, 10
        )
        
        # Setup diagnostics
        from diagnostic_updater import Updater, DiagnosticTask
        self.diagnostics = Updater(self)
        self.diagnostics.add("Robot Status", self.diagnostic_callback)
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """
        Activate state: Start operations
        """
        self.get_logger().info('Activating robot...')
        
        # Enable robot
        self.robot.enable()
        
        # Start control loop
        self.create_timer(0.1, self.control_loop)
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """
        Deactivate state: Stop operations
        """
        self.get_logger().info('Deactivating robot...')
        
        # Stop robot
        self.robot.stop()
        
        return TransitionCallbackReturn.SUCCESS
    
    def diagnostic_callback(self, stat):
        """
        Diagnostics updater
        """
        stat.summary(DiagnosticStatus.OK, "Robot operational")
        
        # Add diagnostic info
        stat.add("Joint positions", str(self.robot.get_joint_positions()))
        stat.add("Battery", "85%")
        stat.add("Temperature", "45°C")
        
        return stat
    
    def vla_callback(self, msg):
        """
        VLA inference and control
        """
        # Image preprocessing
        image = self.bridge_image(msg)
        
        # VLA inference
        action = self.vla_model.predict(image)
        
        # Publish command
        joint_msg = JointState()
        joint_msg.position = action.tolist()
        self.cmd_pub.publish(joint_msg)

# TF2 broadcasting
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class TF2RobotBroadcaster:
    """
    Robot transforms broadcasting
    """
    
    def __init__(self, robot, node):
        self.robot = robot
        self.broadcaster = TransformBroadcaster(node)
    
    def broadcast_transforms(self):
        """
        Broadcast robot transforms
        """
        # Base link → World
        t = TransformStamped()
        t.header.stamp = node.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        
        position, orientation = self.robot.base.get_world_pose()
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        t.transform.rotation.x = orientation[0]
        t.transform.rotation.y = orientation[1]
        t.transform.rotation.z = orientation[2]
        t.transform.rotation.w = orientation[3]
        
        self.broadcaster.sendTransform(t)
        
        # Add other links...
```

**시간: 주 8-10시간**

---

### Week 5-8: 물류 환경 구축

#### 창고 환경 모델링
```python
# warehouse_environment.py
from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid, DynamicCuboid
from omni.isaac.core.prims import RigidPrim
import numpy as np

class WarehouseEnvironment:
    """
    물류 창고 환경
    
    구성:
    - 바닥 (50m x 50m)
    - 선반 5개 (2m 간격)
    - 팔레트 구역
    - 박스 (다양한 크기)
    - 조명
    """
    
    def __init__(self, world):
        self.world = world
        self.boxes = []
        
        # Setup environment
        self.setup_floor()
        self.setup_shelves()
        self.setup_pallet_area()
        self.setup_lighting()
    
    def setup_floor(self):
        """
        바닥 설정
        """
        self.floor = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Floor",
                position=[0, 0, -0.05],
                scale=[50.0, 50.0, 0.1],
                color=[0.5, 0.5, 0.5]
            )
        )
        
        # 마찰 계수 설정
        from pxr import PhysxSchema
        stage = omni.usd.get_context().get_stage()
        floor_prim = stage.GetPrimAtPath("/World/Floor")
        
        physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(floor_prim)
        physx_api.CreateLinearDampingAttr(0.1)
        physx_api.CreateAngularDampingAttr(0.1)
    
    def setup_shelves(self):
        """
        선반 설정 (5개)
        """
        self.shelves = []
        
        for i in range(5):
            x_pos = i * 2.0 - 4.0  # -4, -2, 0, 2, 4
            
            shelf = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Shelf_{i}",
                    position=[x_pos, 3.0, 1.0],
                    scale=[1.0, 0.3, 2.0],
                    color=[0.6, 0.4, 0.2]
                )
            )
            
            self.shelves.append(shelf)
    
    def setup_pallet_area(self):
        """
        팔레트 영역
        """
        # 팔레트 (1.2m x 1.0m)
        self.pallet = self.world.scene.add(
            RigidPrim(
                prim_path="/World/Pallet",
                position=[5.0, 0, 0.1],
                scale=[1.2, 1.0, 0.2]
            )
        )
        
        # 팔레트 material
        from omni.isaac.core.materials import PhysicsMaterial
        
        pallet_material = PhysicsMaterial(
            prim_path="/World/Materials/PalletMaterial",
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.1
        )
        
        self.pallet.apply_physics_material(pallet_material)
    
    def setup_lighting(self):
        """
        창고 조명 (형광등 시뮬레이션)
        """
        from omni.isaac.core.utils.prims import create_prim
        
        # 4개의 천장 조명
        for i in range(4):
            x_pos = i * 10.0 - 15.0
            
            light = create_prim(
                prim_path=f"/World/Light_{i}",
                prim_type="RectLight",
                position=[x_pos, 0, 8.0],
                attributes={
                    "intensity": 5000,
                    "width": 5.0,
                    "height": 5.0,
                    "color": (1.0, 1.0, 0.9),  # 약간 노란빛
                    "enableColorTemperature": True,
                    "colorTemperature": 5500  # Daylight
                }
            )
    
    def spawn_box(self, size="medium", position=None):
        """
        박스 생성
        
        Args:
            size: "small", "medium", "large"
            position: [x, y, z] or None (random)
        """
        # 크기 정의
        box_sizes = {
            "small": [0.2, 0.2, 0.2],
            "medium": [0.3, 0.3, 0.3],
            "large": [0.4, 0.4, 0.5]
        }
        
        scale = box_sizes[size]
        
        # 위치 (random if not specified)
        if position is None:
            position = [
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                1.0
            ]
        
        # 색상 (크기별)
        colors = {
            "small": [1.0, 0.5, 0.5],  # 빨강
            "medium": [0.5, 1.0, 0.5],  # 초록
            "large": [0.5, 0.5, 1.0]    # 파랑
        }
        
        # 박스 생성
        box_idx = len(self.boxes)
        box = self.world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Box_{box_idx}",
                position=position,
                scale=scale,
                color=colors[size],
                mass=1.0 if size == "small" else 2.0 if size == "medium" else 3.0
            )
        )
        
        self.boxes.append({
            'object': box,
            'size': size,
            'prim_path': f"/World/Box_{box_idx}"
        })
        
        return box
    
    def reset(self):
        """
        환경 리셋
        """
        # 박스 제거
        for box_info in self.boxes:
            self.world.scene.remove_object(box_info['prim_path'])
        
        self.boxes = []
        
        # 새 박스 생성 (랜덤)
        num_boxes = np.random.randint(3, 8)
        sizes = np.random.choice(["small", "medium", "large"], num_boxes)
        
        for size in sizes:
            self.spawn_box(size)

# 사용 예시
world = World()
warehouse = WarehouseEnvironment(world)

# 초기 박스 생성
warehouse.spawn_box("small", position=[1, 0, 0.5])
warehouse.spawn_box("medium", position=[2, 0, 0.5])
warehouse.spawn_box("large", position=[3, 0, 0.5])

# Simulation
for i in range(1000):
    world.step(render=True)
    
    # 주기적 리셋
    if i % 500 == 0 and i > 0:
        warehouse.reset()
```

---

#### Domain Randomization
```python
# domain_randomization.py
import random
import numpy as np

class DomainRandomizer:
    """
    Sim-to-Real을 위한 환경 다양화
    
    목적:
    - Simulation의 다양성 증가
    - Real world의 변동성 대비
    - Robust한 policy 학습
    """
    
    def __init__(self, world):
        self.world = world
    
    def randomize_environment(self):
        """
        전체 환경 랜덤화
        """
        self.randomize_physics()
        self.randomize_lighting()
        self.randomize_colors()
        self.randomize_camera()
        self.randomize_objects()
    
    def randomize_physics(self):
        """
        물리 파라미터 랜덤화
        """
        # Gravity (±5%)
        gravity_z = np.random.uniform(-10.3, -9.3)
        self.world.scene.set_gravity([0, 0, gravity_z])
        
        # Friction (전체 object에 적용)
        friction_multiplier = np.random.uniform(0.7, 1.3)
        
        for obj in self.world.scene.get_all_objects():
            if hasattr(obj, 'get_applied_physics_material'):
                material = obj.get_applied_physics_material()
                if material:
                    # Randomize friction
                    base_friction = 0.5
                    new_friction = base_friction * friction_multiplier
                    material.set_static_friction(new_friction)
                    material.set_dynamic_friction(new_friction * 0.8)
    
    def randomize_lighting(self):
        """
        조명 랜덤화
        
        변동:
        - 강도: ±30%
        - 색온도: 4000K ~ 6500K
        - 위치: ±20cm
        """
        from pxr import UsdLux
        
        stage = omni.usd.get_context().get_stage()
        
        for i in range(4):
            light_path = f"/World/Light_{i}"
            light_prim = stage.GetPrimAtPath(light_path)
            
            if light_prim:
                light = UsdLux.RectLight(light_prim)
                
                # Intensity
                base_intensity = 5000
                intensity = base_intensity * np.random.uniform(0.7, 1.3)
                light.GetIntensityAttr().Set(intensity)
                
                # Color temperature
                temp = np.random.uniform(4000, 6500)
                light.GetColorTemperatureAttr().Set(temp)
                
                # Position noise
                current_pos = light.GetPrim().GetAttribute('xformOp:translate').Get()
                noise = np.random.uniform(-0.2, 0.2, 3)
                new_pos = current_pos + noise
                light.GetPrim().GetAttribute('xformOp:translate').Set(tuple(new_pos))
    
    def randomize_colors(self):
        """
        객체 색상 랜덤화
        """
        for box_info in warehouse.boxes:
            box = box_info['object']
            
            # Random color (HSV space에서)
            hue = random.random()
            saturation = random.uniform(0.5, 1.0)
            value = random.uniform(0.5, 1.0)
            
            # HSV to RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            
            box.set_color(rgb)
    
    def randomize_camera(self):
        """
        카메라 파라미터 랜덤화
        
        변동:
        - 위치: ±2cm
        - 각도: ±5도
        - FOV: ±5도
        """
        camera_prim_path = "/World/Robot/Camera"
        camera = self.world.scene.get_object(camera_prim_path)
        
        if camera:
            # Position noise
            current_pos, current_rot = camera.get_local_pose()
            pos_noise = np.random.normal(0, 0.02, 3)  # ±2cm
            new_pos = current_pos + pos_noise
            
            # Orientation noise
            from scipy.spatial.transform import Rotation
            angle_noise = np.random.uniform(-5, 5, 3)  # ±5 degrees
            rot_noise = Rotation.from_euler('xyz', angle_noise, degrees=True)
            current_rot_obj = Rotation.from_quat(current_rot)
            new_rot = (current_rot_obj * rot_noise).as_quat()
            
            camera.set_local_pose(new_pos, new_rot)
    
    def randomize_objects(self):
        """
        객체 속성 랜덤화
        """
        for box_info in warehouse.boxes:
            box = box_info['object']
            
            # Mass variation (±20%)
            base_mass = box.get_mass()
            new_mass = base_mass * np.random.uniform(0.8, 1.2)
            box.set_mass(new_mass)
            
            # Size variation (±5%)
            current_scale = box.get_scale()
            scale_factor = np.random.uniform(0.95, 1.05)
            new_scale = current_scale * scale_factor
            box.set_scale(new_scale)
    
    def apply_texture_randomization(self):
        """
        텍스처 랜덤화 (고급)
        """
        # Add noise to textures
        # Apply different materials
        pass

# 사용 예시
randomizer = DomainRandomizer(world)

# 각 에피소드마다 환경 랜덤화
for episode in range(num_episodes):
    # Reset
    warehouse.reset()
    
    # Randomize
    randomizer.randomize_environment()
    
    # Collect data or evaluate
    # ...
```

---

#### 카메라 설정
```python
# camera_setup.py
from omni.isaac.sensor import Camera
import numpy as np

class RobotCamera:
    """
    RGB-D 카메라
    
    위치: 로봇 상단 (eye-in-hand 또는 fixed)
    """
    
    def __init__(self, world, parent_prim_path):
        self.world = world
        
        # Create camera
        self.camera = Camera(
            prim_path=f"{parent_prim_path}/Camera",
            position=[0, 0, 0.5],  # 로봇 기준 위치
            frequency=20,  # Hz
            resolution=(640, 480),
            orientation=[0, 0, 0, 1]
        )
        
        # Add to scene
        self.world.scene.add(self.camera)
        
        # Initialize
        self.camera.initialize()
        
        # Add depth
        self.camera.add_distance_to_image_plane_to_frame()
        
        # Add segmentation (선택적)
        # self.camera.add_semantic_segmentation_to_frame()
    
    def get_observation(self):
        """
        카메라 observation 획득
        
        Returns:
            dict with 'rgb' and 'depth'
        """
        # Get current frame
        frame = self.camera.get_current_frame()
        
        # RGB
        rgb = frame['rgba'][:, :, :3]  # Remove alpha channel
        
        # Depth
        depth = frame['distance_to_image_plane']
        
        # Normalize depth (0-5m → 0-1)
        depth = np.clip(depth, 0, 5.0) / 5.0
        
        return {
            'rgb': rgb,
            'depth': depth
        }
    
    def get_camera_intrinsics(self):
        """
        카메라 내부 파라미터
        """
        # Get camera parameters
        fov = self.camera.get_horizontal_fov()
        width, height = self.camera.get_resolution()
        
        # Compute focal length
        fx = (width / 2.0) / np.tan(np.radians(fov / 2.0))
        fy = fx  # Assume square pixels
        
        cx = width / 2.0
        cy = height / 2.0
        
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return intrinsics
    
    def project_3d_to_2d(self, point_3d):
        """
        3D 점을 이미지 평면에 투영
        
        Args:
            point_3d: [x, y, z] in camera frame
        
        Returns:
            [u, v] in image coordinates
        """
        K = self.get_camera_intrinsics()
        
        # Project
        point_2d_homogeneous = K @ point_3d
        u = point_2d_homogeneous[0] / point_2d_homogeneous[2]
        v = point_2d_homogeneous[1] / point_2d_homogeneous[2]
        
        return np.array([u, v])

# Multiple cameras
class MultiCameraSetup:
    """
    여러 카메라 (다양한 viewpoint)
    """
    
    def __init__(self, world, robot_prim_path):
        self.cameras = {}
        
        # Wrist camera (eye-in-hand)
        self.cameras['wrist'] = RobotCamera(
            world,
            f"{robot_prim_path}/wrist"
        )
        
        # Front camera (fixed)
        self.cameras['front'] = RobotCamera(
            world,
            f"{robot_prim_path}/base"
        )
        
        # Top-down camera (bird's eye view)
        self.cameras['top'] = RobotCamera(
            world,
            "/World/TopCamera"
        )
    
    def get_all_observations(self):
        """
        모든 카메라에서 observation
        """
        observations = {}
        
        for name, camera in self.cameras.items():
            observations[name] = camera.get_observation()
        
        return observations
```

**시간: 주 8-10시간**

---

## Month 9: Action & Observation Space 설계

### Week 1: Action Space 설계 ⚠️ 매우 중요!

#### 설계 고려사항
```python
# action_space.py

class ActionSpace:
    """
    VLA의 출력을 어떻게 정의할 것인가?
    
    핵심 결정사항:
    1. Control Space (Joint vs Cartesian)
    2. Control Mode (Position vs Velocity vs Torque)
    3. Absolute vs Delta
    4. Normalization
    5. Gripper control
    """
    
    def __init__(self, robot, control_type='delta_joint'):
        self.robot = robot
        self.control_type = control_type
        
        # Joint limits
        self.joint_min = np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0])
        self.joint_max = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 1])
        
        # Delta limits (작게!)
        self.delta_max = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
        
        # Velocity limits
        self.velocity_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])

# Option 1: Absolute Joint Position
"""
장점:
- 직접적인 제어
- Singularity 없음
- 빠른 실행

단점:
- 직관적이지 않음
- Task space reasoning 어려움

사용 시기:
- 정밀한 제어 필요
- 짧은 horizon task
"""
class AbsoluteJointSpace(ActionSpace):
    def __init__(self, robot):
        super().__init__(robot, 'absolute_joint')
        self.action_dim = 7  # 6-DOF arm + gripper
    
    def normalize(self, action):
        """
        [joint_min, joint_max] → [-1, 1]
        """
        normalized = 2 * (action - self.joint_min) / \
                    (self.joint_max - self.joint_min) - 1
        return normalized
    
    def denormalize(self, normalized_action):
        """
        [-1, 1] → [joint_min, joint_max]
        """
        action = (normalized_action + 1) / 2 * \
                (self.joint_max - self.joint_min) + self.joint_min
        return action
    
    def apply(self, normalized_action):
        """
        Apply action to robot
        """
        action = self.denormalize(normalized_action)
        
        # Safety check
        action = np.clip(action, self.joint_min, self.joint_max)
        
        # Apply
        self.robot.set_joint_positions(action)

# Option 2: Delta Joint (추천!)
"""
장점:
- 학습 안정적
- 에러 누적 방지
- Smooth control
- Safe (작은 step)

단점:
- Long-horizon planning 어려움

사용 시기:
- 대부분의 VLA
- BC 학습
"""
class DeltaJointSpace(ActionSpace):
    def __init__(self, robot):
        super().__init__(robot, 'delta_joint')
        self.action_dim = 7
    
    def normalize(self, delta_action):
        """
        [-delta_max, delta_max] → [-1, 1]
        """
        normalized = delta_action / self.delta_max
        normalized = np.clip(normalized, -1, 1)
        return normalized
    
    def denormalize(self, normalized_action):
        """
        [-1, 1] → [-delta_max, delta_max]
        """
        delta = normalized_action * self.delta_max
        return delta
    
    def apply(self, normalized_action):
        """
        Apply delta to current position
        """
        delta = self.denormalize(normalized_action)
        
        # Current position
        current = self.robot.get_joint_positions()
        
        # Target position
        target = current + delta
        
        # Safety: Joint limits
        target = np.clip(target, self.joint_min, self.joint_max)
        
        # Apply
        self.robot.set_joint_positions(target)

# Option 3: Cartesian Space
"""
장점:
- 직관적
- Task 중심
- 쉬운 demonstration

단점:
- IK 필요 (추가 오차)
- Singularity 가능
- 느림

사용 시기:
- Teleoperation
- Sparse reward task
"""
class CartesianSpace(ActionSpace):
    def __init__(self, robot):
        super().__init__(robot, 'cartesian')
        self.action_dim = 7  # [x, y, z, qx, qy, qz, qw]
        
        # Workspace limits
        self.workspace_min = np.array([0.2, -0.5, 0.0])
        self.workspace_max = np.array([0.8, 0.5, 1.0])
    
    def apply(self, normalized_action):
        """
        Apply Cartesian action
        """
        # Denormalize position
        position = normalized_action[:3]
        position = (position + 1) / 2 * \
                  (self.workspace_max - self.workspace_min) + \
                  self.workspace_min
        
        # Quaternion (already normalized)
        orientation = normalized_action[3:]
        
        # IK
        joint_positions = self.inverse_kinematics(position, orientation)
        
        if joint_positions is not None:
            self.robot.set_joint_positions(joint_positions)
    
    def inverse_kinematics(self, position, orientation):
        """
        IK solver
        """
        from omni.isaac.manipulators import IKSolver
        
        ik_solver = IKSolver(self.robot)
        joint_positions = ik_solver.solve(position, orientation)
        
        return joint_positions
```

---

#### Action Chunking (RT-1 방식)
```python
# action_chunking.py

class ActionChunking:
    """
    한 번에 여러 timestep의 action 예측
    
    장점:
    - Temporal consistency
    - Smoother trajectories
    - Less myopic (더 먼 미래 고려)
    - Training stability
    
    단점:
    - 실시간성 약간 저하
    - 메모리 증가
    """
    
    def __init__(self, model, chunk_size=10, execute_ratio=0.5):
        self.model = model
        self.chunk_size = chunk_size
        self.execute_steps = int(chunk_size * execute_ratio)
        
        self.action_buffer = []
    
    def predict_chunk(self, observation):
        """
        Model이 chunk_size만큼의 action 예측
        
        Args:
            observation: Current observation
        
        Returns:
            action_chunk: (chunk_size, action_dim)
        """
        # Model forward
        with torch.no_grad():
            action_chunk = self.model.predict_sequence(
                observation,
                sequence_length=self.chunk_size
            )
        
        return action_chunk
    
    def get_next_action(self, observation):
        """
        다음 action 반환
        
        Logic:
        1. Buffer가 비었으면 new chunk 예측
        2. Buffer에서 action 하나 pop
        3. Execute_steps만큼 실행했으면 new chunk
        """
        # Buffer empty or need refresh?
        if len(self.action_buffer) == 0:
            # Predict new chunk
            chunk = self.predict_chunk(observation)
            self.action_buffer = list(chunk)
            self.executed_count = 0
        
        # Pop action
        action = self.action_buffer.pop(0)
        self.executed_count += 1
        
        # Re-predict if executed enough
        if self.executed_count >= self.execute_steps:
            self.action_buffer = []
        
        return action

# Model with chunking
class VLAWithChunking(nn.Module):
    """
    Action chunking을 지원하는 VLA
    """
    
    def __init__(self, vision_encoder, chunk_size=10):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.chunk_size = chunk_size
        
        # Temporal decoder
        self.temporal_decoder = nn.LSTM(
            input_size=768,  # Vision feature dim
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )
        
        # Action head
        self.action_head = nn.Linear(512, 7)  # 7-DOF action
    
    def forward(self, observation):
        """
        Single observation → action chunk
        
        Args:
            observation: (B, 3, H, W)
        
        Returns:
            actions: (B, chunk_size, 7)
        """
        # Vision encoding
        vision_features = self.vision_encoder(observation)  # (B, 768)
        
        # Repeat for sequence
        vision_seq = vision_features.unsqueeze(1).repeat(
            1, self.chunk_size, 1
        )  # (B, chunk_size, 768)
        
        # Temporal decoding
        lstm_out, _ = self.temporal_decoder(vision_seq)  # (B, chunk_size, 512)
        
        # Action prediction
        actions = self.action_head(lstm_out)  # (B, chunk_size, 7)
        
        return actions
    
    def predict_sequence(self, observation, sequence_length=None):
        """
        Inference mode
        """
        if sequence_length is None:
            sequence_length = self.chunk_size
        
        self.eval()
        with torch.no_grad():
            # Adjust chunk size temporarily
            original_chunk = self.chunk_size
            self.chunk_size = sequence_length
            
            actions = self.forward(observation)
            
            self.chunk_size = original_chunk
        
        return actions[0]  # Remove batch dimension

# Training with chunking
def train_with_chunking():
    """
    Action chunking 학습
    
    Data format:
    - observation: (B, 3, H, W)
    - actions: (B, chunk_size, 7)
    """
    model = VLAWithChunking(vision_encoder, chunk_size=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        for obs, action_chunk in dataloader:
            # Forward
            pred_actions = model(obs)
            
            # Loss (모든 timestep)
            loss = criterion(pred_actions, action_chunk)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

#### 실험 가이드
```python
# 어떤 action space가 좋은가?

experiments = {
    'exp1_absolute': {
        'space': AbsoluteJointSpace,
        'expected': 'Baseline, potentially unstable'
    },
    'exp2_delta': {
        'space': DeltaJointSpace,
        'expected': 'Stable, smooth, recommended'
    },
    'exp3_delta_chunking': {
        'space': DeltaJointSpace,
        'chunking': ActionChunking(chunk_size=10),
        'expected': 'Best performance'
    },
    'exp4_cartesian': {
        'space': CartesianSpace,
        'expected': 'Intuitive but slower'
    }
}

def compare_action_spaces():
    """
    Action space 비교 실험
    """
    results = {}
    
    for exp_name, config in experiments.items():
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")
        
        # Setup
        action_space = config['space'](robot)
        
        # Train
        model = train_vla(action_space)
        
        # Evaluate
        metrics = evaluate_model(model, action_space)
        
        results[exp_name] = metrics
        
        print(f"Results:")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Smoothness: {metrics['smoothness']:.4f}")
        print(f"  Avg Time: {metrics['avg_time']:.2f}s")
    
    # Plot comparison
    plot_comparison(results)
    
    return results

# 평가 기준:
"""
1. Success Rate (가장 중요!)
2. Trajectory Smoothness (jerk)
3. Learning Stability (loss curve)
4. Inference Time
5. Sim-to-Real Gap (나중에)
"""
```

**시간: 주 8-10시간**

---

### Week 2: Observation Space 설계

#### Observation 구성
```python
# observation_space.py

class ObservationSpace:
    """
    VLA 입력을 어떻게 구성할 것인가?
    
    가능한 modality:
    1. Vision: RGB, Depth, Semantic
    2. Proprioception: Joint state, EE pose
    3. Language: Task instruction
    4. History: Past observations
    5. Goal: Target state
    """
    
    def __init__(self, config):
        self.use_rgb = config.get('use_rgb', True)
        self.use_depth = config.get('use_depth', False)
        self.use_proprio = config.get('use_proprio', True)
        self.use_language = config.get('use_language', False)
        self.history_length = config.get('history_length', 1)
        
        # History buffer
        from collections import deque
        self.history_buffer = deque(maxlen=self.history_length)
    
    def get_observation(self, camera, robot, instruction=None):
        """
        현재 observation 수집
        
        Returns:
            dict with various modalities
        """
        obs = {}
        
        # Vision
        if self.use_rgb:
            rgb = camera.get_rgb()  # (H, W, 3)
            obs['rgb'] = self.preprocess_image(rgb)
        
        if self.use_depth:
            depth = camera.get_depth()  # (H, W, 1)
            obs['depth'] = self.preprocess_depth(depth)
        
        # Proprioception
        if self.use_proprio:
            obs['joint_pos'] = robot.get_joint_positions()  # (7,)
            obs['joint_vel'] = robot.get_joint_velocities()  # (7,)
            obs['ee_pose'] = robot.get_end_effector_pose()  # (7,)
            obs['gripper_state'] = robot.get_gripper_state()  # (1,)
        
        # Language
        if self.use_language and instruction:
            obs['instruction'] = self.encode_instruction(instruction)
        
        # History
        if self.history_length > 1:
            self.history_buffer.append(obs.copy())
            obs['history'] = list(self.history_buffer)
        
        return obs
    
    def preprocess_image(self, image):
        """
        이미지 전처리
        
        Steps:
        1. Resize to 224x224
        2. Normalize [0, 255] → [0, 1]
        3. ImageNet normalization
        4. Channels first (C, H, W)
        """
        import cv2
        
        # Resize
        image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Channels first
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def preprocess_depth(self, depth):
        """
        Depth 전처리
        
        Steps:
        1. Clip far values (> 5m)
        2. Normalize to [0, 1]
        3. Resize if needed
        """
        # Clip
        depth = np.clip(depth, 0, 5.0)
        
        # Normalize
        depth = depth / 5.0
        
        # Add channel dimension if needed
        if len(depth.shape) == 2:
            depth = depth[..., np.newaxis]
        
        return depth
    
    def encode_instruction(self, instruction):
        """
        Language instruction encoding
        
        Methods:
        1. BERT tokenizer
        2. CLIP text encoder
        3. Simple word embedding
        """
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        tokens = tokenizer(
            instruction,
            padding='max_length',
            max_length=20,
            truncation=True,
            return_tensors='np'
        )
        
        return tokens['input_ids']

# Multi-modal Fusion
class MultiModalObservation(nn.Module):
    """
    다양한 observation을 어떻게 결합할까?
    
    Architecture:
    1. Separate encoders for each modality
    2. Fusion layer
    3. Joint representation
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.use_rgb = config['use_rgb']
        self.use_depth = config['use_depth']
        self.use_proprio = config['use_proprio']
        self.use_language = config['use_language']
        
        # Vision encoder (RGB)
        if self.use_rgb:
            from transformers import ViTModel
            self.rgb_encoder = ViTModel.from_pretrained(
                'google/vit-base-patch16-224'
            )
            rgb_dim = 768
        else:
            rgb_dim = 0
        
        # Depth encoder
        if self.use_depth:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 256)
            )
            depth_dim = 256
        else:
            depth_dim = 0
        
        # Proprioception encoder
        if self.use_proprio:
            # 7 pos + 7 vel + 7 ee_pose + 1 gripper = 22
            self.proprio_encoder = nn.Sequential(
                nn.Linear(22, 128),
                nn.ReLU(),
                nn.Linear(128, 256)
            )
            proprio_dim = 256
        else:
            proprio_dim = 0
        
        # Language encoder
        if self.use_language:
            from transformers import AutoModel
            self.language_encoder = AutoModel.from_pretrained(
                "bert-base-uncased"
            )
            lang_dim = 768
        else:
            lang_dim = 0
        
        # Fusion
        total_dim = rgb_dim + depth_dim + proprio_dim + lang_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512)
        )
    
    def forward(self, obs):
        """
        Encode and fuse all modalities
        
        Args:
            obs: dict with various keys
        
        Returns:
            fused_features: (B, 512)
        """
        features = []
        
        # RGB
        if self.use_rgb and 'rgb' in obs:
            rgb_feat = self.rgb_encoder(obs['rgb']).last_hidden_state[:, 0]
            features.append(rgb_feat)
        
        # Depth
        if self.use_depth and 'depth' in obs:
            depth_feat = self.depth_encoder(obs['depth'])
            features.append(depth_feat)
        
        # Proprioception
        if self.use_proprio:
            proprio = torch.cat([
                obs['joint_pos'],
                obs['joint_vel'],
                obs['ee_pose'],
                obs['gripper_state']
            ], dim=-1)
            proprio_feat = self.proprio_encoder(proprio)
            features.append(proprio_feat)
        
        # Language
        if self.use_language and 'instruction' in obs:
            lang_feat = self.language_encoder(
                obs['instruction']
            ).last_hidden_state[:, 0]
            features.append(lang_feat)
        
        # Concatenate
        combined = torch.cat(features, dim=-1)
        
        # Fuse
        fused = self.fusion(combined)
        
        return fused
```

---

#### Ablation Study
```python
# observation_ablation.py

def observation_ablation_study():
    """
    Observation modality ablation
    
    목적: 각 modality의 기여도 측정
    """
    
    configs = {
        'rgb_only': {
            'use_rgb': True,
            'use_depth': False,
            'use_proprio': False,
            'use_language': False
        },
        'rgb_proprio': {
            'use_rgb': True,
            'use_depth': False,
            'use_proprio': True,
            'use_language': False
        },
        'rgb_depth': {
            'use_rgb': True,
            'use_depth': True,
            'use_proprio': False,
            'use_language': False
        },
        'rgb_depth_proprio': {
            'use_rgb': True,
            'use_depth': True,
            'use_proprio': True,
            'use_language': False
        },
        'full': {
            'use_rgb': True,
            'use_depth': True,
            'use_proprio': True,
            'use_language': True
        }
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {name}")
        print(f"{'='*60}")
        
        # Train model
        model = train_vla_with_config(config)
        
        # Evaluate
        metrics = evaluate_model(model)
        
        results[name] = metrics
        
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Training Time: {metrics['train_time']:.1f}s")
        print(f"Inference Time: {metrics['inference_time']:.3f}s")
    
    # Analysis
    print("\n" + "="*60)
    print("ABLATION ANALYSIS")
    print("="*60)
    
    # Best configuration
    best_config = max(results.items(), key=lambda x: x[1]['success_rate'])
    print(f"Best: {best_config[0]} ({best_config[1]['success_rate']:.2%})")
    
    # Contribution of each modality
    baseline = results['rgb_only']['success_rate']
    
    print(f"\nContributions:")
    print(f"  Proprio: +{results['rgb_proprio']['success_rate'] - baseline:.1%}")
    print(f"  Depth: +{results['rgb_depth']['success_rate'] - baseline:.1%}")
    
    return results

# 일반적인 결과:
"""
RGB only: 40-50%
RGB + Proprio: 60-70% (큰 향상!)
RGB + Depth: 50-60%
RGB + Depth + Proprio: 70-80% (Best balance)
Full (with Language): 75-85% (task dependent)

Recommendation:
- 대부분 task: RGB + Proprio
- Depth 필요한 경우: 3D reasoning, occlusion
- Language: Multi-task, zero-shot generalization
"""
```

**시간: 주 8-10시간**

---

## Month 10: 첫 물류 VLA 개발

### Week 1-2: 데이터 수집 & 품질 관리

#### Teleoperation 시스템
```python
# teleoperation.py
import numpy as np
from pynput import keyboard, mouse

class TeleoperationSystem:
    """
    사람이 로봇을 제어하며 demonstration 수집
    
    입력 방법:
    1. Keyboard (간단, 부정확)
    2. SpaceMouse (3D input, 추천!)
    3. VR Controller (가장 직관적)
    """
    
    def __init__(self, robot, camera):
        self.robot = robot
        self.camera = camera
        
        # Recording state
        self.recording = False
        self.current_episode = []
        self.episodes = []
        
        # Control state
        self.current_velocity = np.zeros(7)
        
        # Setup input device
        self.setup_keyboard_control()
    
    def setup_keyboard_control(self):
        """
        키보드 제어 설정
        
        키 매핑:
        - W/S: Joint 1 +/-
        - A/D: Joint 2 +/-
        - Q/E: Joint 3 +/-
        - ...
        - Space: Gripper toggle
        - R: Start recording
        - T: Stop recording
        """
        self.key_mapping = {
            'w': (0, 0.5),
            's': (0, -0.5),
            'a': (1, 0.5),
            'd': (1, -0.5),
            'q': (2, 0.5),
            'e': (2, -0.5),
            # ... more keys
        }
        
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
    
    def on_press(self, key):
        """
        Key press handler
        """
        try:
            char = key.char
            
            # Control joints
            if char in self.key_mapping:
                joint_idx, velocity = self.key_mapping[char]
                self.current_velocity[joint_idx] = velocity
            
            # Recording control
            elif char == 'r':
                self.start_recording()
            elif char == 't':
                self.stop_recording()
            
            # Gripper
            elif char == ' ':
                self.toggle_gripper()
        
        except AttributeError:
            pass
    
    def on_release(self, key):
        """
        Key release handler
        """
        # Stop movement
        self.current_velocity = np.zeros(7)
        
        if key == keyboard.Key.esc:
            return False
    
    def start_recording(self):
        """
        Start recording episode
        """
        self.recording = True
        self.current_episode = []
        print("🔴 Recording started")
    
    def stop_recording(self):
        """
        Stop recording episode
        """
        if self.recording and len(self.current_episode) > 10:
            self.episodes.append(self.current_episode.copy())
            print(f"✅ Episode {len(self.episodes)} saved ({len(self.current_episode)} frames)")
        else:
            print("❌ Episode too short, discarded")
        
        self.recording = False
        self.current_episode = []
    
    def step(self):
        """
        Single control step
        
        Call this in simulation loop
        """
        # Apply control
        self.robot.apply_joint_velocities(self.current_velocity)
        
        # Record if active
        if self.recording:
            obs = self.get_observation()
            action = self.current_velocity.copy()
            
            self.current_episode.append({
                'observation': obs,
                'action': action,
                'timestamp': time.time()
            })
    
    def get_observation(self):
        """
        Get current observation
        """
        return {
            'rgb': self.camera.get_rgb(),
            'depth': self.camera.get_depth(),
            'joint_pos': self.robot.get_joint_positions(),
            'joint_vel': self.robot.get_joint_velocities(),
            'gripper': self.robot.get_gripper_state()
        }
    
    def save_demonstrations(self, filename='demonstrations.pkl'):
        """
        Save collected demonstrations
        """
        import pickle
        
        with open(filename, 'wb') as f:
            pickle.dump(self.episodes, f)
        
        print(f"💾 Saved {len(self.episodes)} episodes to {filename}")
    
    def toggle_gripper(self):
        """
        Toggle gripper state
        """
        current = self.robot.get_gripper_state()
        if current > 0.5:
            self.robot.close_gripper()
        else:
            self.robot.open_gripper()

# 사용 예시
teleop = TeleoperationSystem(robot, camera)

print("""
=== Teleoperation Controls ===
W/S: Joint 1
A/D: Joint 2
Q/E: Joint 3
...
Space: Toggle gripper
R: Start recording
T: Stop recording
ESC: Exit
==============================
""")

# Simulation loop
while simulation_app.is_running():
    teleop.step()
    world.step(render=True)
```

---

#### 데이터 품질 관리
```python
# data_quality.py

class DataQualityChecker:
    """
    수집한 데이터 검증
    
    체크 항목:
    1. Episode length
    2. Action variance
    3. Success/failure
    4. Image quality
    5. Trajectory smoothness
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'min_length': 20,
            'max_length': 500,
            'min_action_std': 0.01,
            'min_brightness': 10,
            'max_jerk': 5.0
        }
    
    def check_episode(self, episode, success=True):
        """
        Single episode 검증
        
        Returns:
            is_good: bool
            issues: list of issue strings
        """
        issues = []
        
        # 1. Length check
        length = len(episode)
        if length < self.quality_thresholds['min_length']:
            issues.append(f"Too short ({length} < {self.quality_thresholds['min_length']})")
        
        if length > self.quality_thresholds['max_length']:
            issues.append(f"Too long ({length} > {self.quality_thresholds['max_length']})")
        
        # 2. Action variance
        actions = np.array([step['action'] for step in episode])
        action_std = np.std(actions, axis=0).mean()
        
        if action_std < self.quality_thresholds['min_action_std']:
            issues.append(f"Static actions (std={action_std:.4f})")
        
        # 3. Success check
        if not success:
            # 실패 에피소드도 일부 포함 (10-15%)
            if np.random.random() > 0.15:
                issues.append("Failed episode")
        
        # 4. Image quality
        first_obs = episode[0]['observation']
        image = first_obs['rgb']
        
        brightness = np.mean(image)
        if brightness < self.quality_thresholds['min_brightness']:
            issues.append(f"Too dark (brightness={brightness:.1f})")
        
        contrast = np.std(image)
        if contrast < 5:
            issues.append(f"Low contrast (std={contrast:.1f})")
        
        # 5. Smoothness (jerk)
        velocities = np.diff(actions, axis=0)
        jerks = np.diff(velocities, axis=0)
        max_jerk = np.max(np.abs(jerks))
        
        if max_jerk > self.quality_thresholds['max_jerk']:
            issues.append(f"Jerky movements (max_jerk={max_jerk:.2f})")
        
        is_good = len(issues) == 0
        
        return is_good, issues
    
    def clean_dataset(self, episodes):
        """
        Dataset 정제
        
        Returns:
            clean_episodes: list
            statistics: dict
        """
        clean_episodes = []
        
        statistics = {
            'total': len(episodes),
            'removed_short': 0,
            'removed_long': 0,
            'removed_static': 0,
            'removed_failed': 0,
            'removed_quality': 0,
            'removed_jerky': 0,
            'kept': 0
        }
        
        for i, episode in enumerate(episodes):
            success = episode.get('success', True)
            is_good, issues = self.check_episode(episode['data'], success)
            
            if is_good:
                clean_episodes.append(episode)
                statistics['kept'] += 1
            else:
                # Update statistics
                for issue in issues:
                    if 'short' in issue.lower():
                        statistics['removed_short'] += 1
                    elif 'long' in issue.lower():
                        statistics['removed_long'] += 1
                    elif 'static' in issue.lower():
                        statistics['removed_static'] += 1
                    elif 'failed' in issue.lower():
                        statistics['removed_failed'] += 1
                    elif 'jerky' in issue.lower():
                        statistics['removed_jerky'] += 1
                    else:
                        statistics['removed_quality'] += 1
                
                print(f"Episode {i+1} removed: {', '.join(issues)}")
        
        # Print report
        print("\n" + "="*60)
        print("DATA CLEANING REPORT")
        print("="*60)
        for key, val in statistics.items():
            percentage = (val / statistics['total'] * 100) if statistics['total'] > 0 else 0
            print(f"{key:20s}: {val:4d} ({percentage:5.1f}%)")
        print("="*60)
        
        return clean_episodes, statistics
    
    def visualize_quality(self, episodes):
        """
        데이터 품질 시각화
        """
        import matplotlib.pyplot as plt
        
        # Episode lengths
        lengths = [len(ep['data']) for ep in episodes]
        
        # Action statistics
        all_actions = []
        for ep in episodes:
            actions = [step['action'] for step in ep['data']]
            all_actions.extend(actions)
        all_actions = np.array(all_actions)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode length distribution
        axes[0, 0].hist(lengths, bins=30)
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].set_xlabel('Length (frames)')
        axes[0, 0].set_ylabel('Count')
        
        # Action distribution
        for i in range(min(7, all_actions.shape[1])):
            axes[0, 1].hist(all_actions[:, i], bins=50, alpha=0.5, label=f'Joint {i+1}')
        axes[0, 1].set_title('Action Distribution')
        axes[0, 1].set_xlabel('Action value')
        axes[0, 1].legend()
        
        # Action correlation
        correlation = np.corrcoef(all_actions.T)
        im = axes[1, 0].imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Action Correlation')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Success rate over time
        success_rate = []
        window = 10
        for i in range(0, len(episodes), window):
            batch = episodes[i:i+window]
            rate = sum(ep.get('success', True) for ep in batch) / len(batch)
            success_rate.append(rate)
        
        axes[1, 1].plot(success_rate)
        axes[1, 1].set_title('Success Rate Over Collection')
        axes[1, 1].set_xlabel(f'Batch (window={window})')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('data_quality_report.png')
        plt.show()
```

---

#### Data Augmentation
```python
# data_augmentation.py

class RobotDataAugmentation:
    """
    이미지 + Action 동시 augmentation
    
    주의: 이미지만 augment하면 안 됨!
           Action도 같이 변환해야 consistency 유지
    """
    
    def __init__(self, config):
        self.config = config
        
        # Augmentation probabilities
        self.p_flip = config.get('p_flip', 0.3)
        self.p_rotate = config.get('p_rotate', 0.2)
        self.p_brightness = config.get('p_brightness', 0.3)
        self.p_noise = config.get('p_noise', 0.2)
        self.p_color_jitter = config.get('p_color_jitter', 0.3)
    
    def augment(self, observation, action):
        """
        Augment observation and action together
        
        Args:
            observation: dict with 'rgb', 'depth', etc.
            action: numpy array (7,)
        
        Returns:
            aug_observation: augmented observation
            aug_action: augmented action
        """
        # Copy to avoid in-place modification
        obs = observation.copy()
        act = action.copy()
        
        # 1. Horizontal flip (조심! action도 변환)
        if np.random.random() < self.p_flip:
            obs, act = self.horizontal_flip(obs, act)
        
        # 2. Small rotation (조심! action도 변환)
        if np.random.random() < self.p_rotate:
            angle = np.random.uniform(-5, 5)
            obs, act = self.rotate(obs, act, angle)
        
        # 3. Brightness (action 불변)
        if np.random.random() < self.p_brightness:
            obs = self.adjust_brightness(obs)
        
        # 4. Gaussian noise (action 불변)
        if np.random.random() < self.p_noise:
            obs = self.add_noise(obs)
        
        # 5. Color jitter (action 불변)
        if np.random.random() < self.p_color_jitter:
            obs = self.color_jitter(obs)
        
        return obs, act
    
    def horizontal_flip(self, obs, action):
        """
        Horizontal flip
        
        Image: 좌우 반전
        Action: y축 관련 joint 반전
        """
        # Flip image
        if 'rgb' in obs:
            obs['rgb'] = np.fliplr(obs['rgb'])
        
        if 'depth' in obs:
            obs['depth'] = np.fliplr(obs['depth'])
        
        # Flip action (예시, 실제는 robot kinematics에 따라)
        # Joint 2, 4, 6: y축 관련
        action[[1, 3, 5]] = -action[[1, 3, 5]]
        
        return obs, action
    
    def rotate(self, obs, action, angle):
        """
        Small rotation
        
        Image: 회전
        Action: base frame 기준 회전
        """
        import cv2
        from scipy.spatial.transform import Rotation
        
        # Rotate image
        if 'rgb' in obs:
            h, w = obs['rgb'].shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            obs['rgb'] = cv2.warpAffine(obs['rgb'], M, (w, h))
        
        # Rotate action (Cartesian space에서)
        # Joint space면 FK → rotate → IK 필요
        # 여기서는 간단히 approximation
        rot = Rotation.from_euler('z', angle, degrees=True)
        
        # Action의 처음 3개가 xyz라고 가정
        if len(action) >= 3:
            action[:3] = rot.apply(action[:3])
        
        return obs, action
    
    def adjust_brightness(self, obs):
        """
        Brightness adjustment
        """
        if 'rgb' in obs:
            factor = np.random.uniform(0.7, 1.3)
            obs['rgb'] = np.clip(obs['rgb'] * factor, 0, 255).astype(np.uint8)
        
        return obs
    
    def add_noise(self, obs):
        """
        Gaussian noise
        """
        if 'rgb' in obs:
            noise = np.random.normal(0, 5, obs['rgb'].shape)
            obs['rgb'] = np.clip(obs['rgb'] + noise, 0, 255).astype(np.uint8)
        
        return obs
    
    def color_jitter(self, obs):
        """
        Color jittering
        """
        if 'rgb' in obs:
            # Hue, Saturation, Value adjustments
            import cv2
            
            hsv = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Hue shift
            hsv[:, :, 0] += np.random.uniform(-10, 10)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
            
            # Saturation
            hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Value
            hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            
            obs['rgb'] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return obs

# Dataset with augmentation
class AugmentedRobotDataset(Dataset):
    def __init__(self, episodes, augmentation=None):
        self.episodes = episodes
        self.augmentation = augmentation
        
        # Flatten to (obs, action) pairs
        self.data = []
        for ep in episodes:
            for step in ep['data']:
                self.data.append((
                    step['observation'],
                    step['action']
                ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        obs, action = self.data[idx]
        
        # Augmentation
        if self.augmentation:
            obs, action = self.augmentation.augment(obs, action)
        
        # Preprocessing
        obs_tensor = self.preprocess(obs)
        action_tensor = torch.FloatTensor(action)
        
        return obs_tensor, action_tensor
```

---

**데이터 수집 가이드**
```
목표: 50-100 성공 에피소드

다양성 확보:
┌─────────────────────────────────┐
│ 1. 박스 위치: 5-7 위치          │
│ 2. 박스 방향: 4 방향 (90도씩)   │
│ 3. 조명 조건: 3 단계            │
│ 4. 시작 자세: 다양하게           │
│ 5. 속도: 빠르게/느리게          │
└─────────────────────────────────┘

시간 투자:
- 주말 4-6시간 × 2주
- 한 에피소드: 2-3분
- → 50+ 에피소드 충분

체크리스트:
- [ ] Teleoperation 시스템 작동
- [ ] 최소 50 에피소드 수집
- [ ] 다양한 조건 포함
- [ ] 품질 검증 완료
- [ ] 데이터 저장 확인
```

**시간: 주 8-10시간 (실제 수집 포함)**

### Week 3-4: VLA 학습

#### 모델 선택 및 설정
```python
# model_selection.py

"""
VLA Model 선택 가이드

Option 1: ACT (Action Chunking Transformer) - 추천!
- 장점: 안정적 학습, Temporal consistency, 좋은 성능
- 단점: 메모리 사용량 높음
- 적합: 대부분의 manipulation tasks

Option 2: Diffusion Policy
- 장점: Multi-modal action, 매우 안정적
- 단점: 추론 느림 (iterative denoising)
- 적합: High-precision tasks

Option 3: OpenVLA
- 장점: Pre-trained, Language conditioning
- 단점: 크고 무거움, Fine-tuning 어려움
- 적합: Multi-task, zero-shot generalization

추천: ACT로 시작
→ 안정적이고 빠름
→ 나중에 다른 policy 비교
"""

# ACT Configuration
config = {
    'policy': 'act',
    'dataset': 'box_picking_v1',
    
    # Model architecture
    'vision_encoder': 'vit-base',
    'hidden_dim': 512,
    'n_heads': 8,
    'n_encoder_layers': 4,
    'n_decoder_layers': 1,
    'n_action_steps': 10,  # Action chunking
    'n_obs_steps': 1,
    
    # Action space
    'action_dim': 7,
    'action_type': 'delta_joint',
    
    # Observation
    'use_rgb': True,
    'use_depth': False,
    'use_proprio': True,
    
    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 500,
    'weight_decay': 1e-4,
    'gradient_clip': 1.0,
    
    # Data augmentation
    'use_augmentation': True,
    'aug_prob': 0.5,
    
    # Hardware
    'device': 'cuda',
    'num_workers': 4,
    'mixed_precision': True,
    
    # Checkpointing
    'save_every': 50,
    'validate_every': 10,
}
```

---

#### ACT 모델 구현
```python
# act_model.py
import torch
import torch.nn as nn
from transformers import ViTModel

class ACTPolicy(nn.Module):
    """
    Action Chunking Transformer
    
    Architecture:
    1. Vision Encoder (ViT)
    2. Proprioception Encoder (MLP)
    3. Encoder Transformer
    4. Decoder Transformer (action chunking)
    5. Action Head
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.n_action_steps = config['n_action_steps']
        self.action_dim = config['action_dim']
        
        # Vision Encoder
        self.vision_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224'
        )
        vision_dim = 768
        
        # Vision projection
        self.vision_proj = nn.Linear(vision_dim, self.hidden_dim)
        
        # Proprioception Encoder
        proprio_dim = 7 + 7 + 1  # joint_pos + joint_vel + gripper
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        )
        
        # Encoder (process current observation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config['n_heads'],
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['n_encoder_layers']
        )
        
        # Decoder (generate action sequence)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=config['n_heads'],
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['n_decoder_layers']
        )
        
        # Action queries (learnable)
        self.action_queries = nn.Parameter(
            torch.randn(1, self.n_action_steps, self.hidden_dim)
        )
        
        # Action Head
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, observations):
        """
        Forward pass
        
        Args:
            observations: dict with
                - 'rgb': (B, 3, 224, 224)
                - 'proprio': (B, 15)
        
        Returns:
            actions: (B, n_action_steps, action_dim)
        """
        batch_size = observations['rgb'].shape[0]
        
        # 1. Encode vision
        vision_features = self.vision_encoder(
            observations['rgb']
        ).last_hidden_state  # (B, num_patches, 768)
        
        vision_features = self.vision_proj(vision_features)  # (B, N, hidden_dim)
        
        # 2. Encode proprioception
        proprio_features = self.proprio_encoder(
            observations['proprio']
        ).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # 3. Concatenate features
        encoder_input = torch.cat([
            vision_features,
            proprio_features
        ], dim=1)  # (B, N+1, hidden_dim)
        
        # 4. Encoder
        encoder_output = self.encoder(encoder_input)  # (B, N+1, hidden_dim)
        
        # 5. Prepare action queries
        action_queries = self.action_queries.expand(
            batch_size, -1, -1
        )  # (B, n_action_steps, hidden_dim)
        
        # 6. Decoder
        decoder_output = self.decoder(
            action_queries,
            encoder_output
        )  # (B, n_action_steps, hidden_dim)
        
        # 7. Action prediction
        actions = self.action_head(decoder_output)  # (B, n_action_steps, action_dim)
        
        return actions
    
    def predict(self, observations):
        """
        Inference mode
        """
        self.eval()
        with torch.no_grad():
            actions = self.forward(observations)
        return actions
```

---

#### 학습 스크립트
```python
# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm

class VLATrainer:
    """
    VLA Training Pipeline
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Model
        self.model = ACTPolicy(config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] / 10
        )
        
        # Loss
        self.criterion = nn.MSELoss()
        
        # Mixed precision
        self.scaler = GradScaler() if config['mixed_precision'] else None
        
        # Logging
        if config.get('use_wandb', True):
            wandb.init(
                project='box-picking-vla',
                config=config
            )
        
        # Metrics
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """
        Single training epoch
        """
        self.model.train()
        
        total_loss = 0
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (observations, actions) in enumerate(pbar):
            # Move to device
            observations = {
                k: v.to(self.device) for k, v in observations.items()
            }
            actions = actions.to(self.device)  # (B, n_action_steps, action_dim)
            
            # Forward
            if self.scaler:
                with autocast():
                    pred_actions = self.model(observations)
                    loss = self.criterion(pred_actions, actions)
                
                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
                
                # Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_actions = self.model(observations)
                loss = self.criterion(pred_actions, actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to wandb
            if self.config.get('use_wandb') and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validation
        """
        self.model.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for observations, actions in tqdm(val_loader, desc='Validation'):
                # Move to device
                observations = {
                    k: v.to(self.device) for k, v in observations.items()
                }
                actions = actions.to(self.device)
                
                # Forward
                pred_actions = self.model(observations)
                loss = self.criterion(pred_actions, actions)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        
        # Log
        if self.config.get('use_wandb'):
            wandb.log({'val/loss': avg_loss})
        
        return avg_loss
    
    def train(self, train_loader, val_loader):
        """
        Complete training loop
        """
        print("="*60)
        print("Starting Training")
        print("="*60)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Epochs: {self.config['num_epochs']}")
        print("="*60)
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            if (epoch + 1) % self.config['validate_every'] == 0:
                val_loss = self.validate(val_loader)
                
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt', epoch, val_loss)
                    print(f"✅ Best model saved! (Val Loss: {val_loss:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pt', epoch, train_loss)
            
            # Update scheduler
            self.scheduler.step()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("="*60)
    
    def save_checkpoint(self, filename, epoch, loss):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Checkpoint loaded: {filename}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Loss: {checkpoint['loss']:.4f}")
        
        return checkpoint['epoch']

# Main training script
def main():
    # Load data
    from dataset import BoxPickingDataset
    
    train_dataset = BoxPickingDataset(
        'data/train_episodes.pkl',
        augmentation=True
    )
    
    val_dataset = BoxPickingDataset(
        'data/val_episodes.pkl',
        augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Train
    trainer = VLATrainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
```

---

#### Hyperparameter Tuning
```python
# hyperparameter_tuning.py

class LRFinder:
    """
    Learning Rate Finder
    
    목적: Optimal learning rate 찾기
    방법: Exponentially increasing LR, plot loss
    """
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def find(self, train_loader, min_lr=1e-7, max_lr=1, num_steps=100):
        """
        Run LR finder
        
        Returns:
            lrs: list of learning rates
            losses: list of losses
        """
        # Save initial state
        initial_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        # LR schedule
        lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps)
        losses = []
        
        # Iterator
        data_iter = iter(train_loader)
        
        for lr in tqdm(lrs, desc='LR Finder'):
            # Set LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Get batch
            try:
                observations, actions = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                observations, actions = next(data_iter)
            
            # Move to device
            observations = {k: v.to(self.device) for k, v in observations.items()}
            actions = actions.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            pred_actions = self.model(observations)
            loss = self.criterion(pred_actions, actions)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Record
            losses.append(loss.item())
            
            # Stop if loss explodes
            if len(losses) > 1 and losses[-1] > losses[0] * 10:
                break
        
        # Restore initial state
        self.model.load_state_dict(initial_state['model'])
        self.optimizer.load_state_dict(initial_state['optimizer'])
        
        # Plot
        self.plot(lrs[:len(losses)], losses)
        
        # Find optimal LR
        optimal_lr = self.find_optimal_lr(lrs[:len(losses)], losses)
        
        return lrs[:len(losses)], losses, optimal_lr
    
    def plot(self, lrs, losses):
        """
        Plot LR vs Loss
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        plt.savefig('lr_finder.png')
        plt.show()
    
    def find_optimal_lr(self, lrs, losses):
        """
        Find optimal LR (steepest descent point)
        """
        # Smooth losses
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(losses, sigma=2)
        
        # Find steepest descent
        gradients = np.gradient(smoothed)
        optimal_idx = np.argmin(gradients)
        
        optimal_lr = lrs[optimal_idx]
        
        print(f"Optimal LR: {optimal_lr:.2e}")
        
        return optimal_lr

# Hyperparameter Grid Search
def grid_search():
    """
    Grid search for hyperparameters
    """
    
    param_grid = {
        'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4],
        'batch_size': [16, 32, 64],
        'hidden_dim': [256, 512, 768],
        'n_action_steps': [5, 10, 15]
    }
    
    results = []
    
    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for hd in param_grid['hidden_dim']:
                for steps in param_grid['n_action_steps']:
                    print(f"\n{'='*60}")
                    print(f"Testing: lr={lr}, bs={bs}, hd={hd}, steps={steps}")
                    print(f"{'='*60}")
                    
                    # Update config
                    config.update({
                        'learning_rate': lr,
                        'batch_size': bs,
                        'hidden_dim': hd,
                        'n_action_steps': steps
                    })
                    
                    # Train
                    trainer = VLATrainer(config)
                    trainer.train(train_loader, val_loader)
                    
                    # Evaluate
                    val_loss = trainer.best_val_loss
                    
                    results.append({
                        'config': config.copy(),
                        'val_loss': val_loss
                    })
                    
                    print(f"Val Loss: {val_loss:.4f}")
    
    # Find best
    best = min(results, key=lambda x: x['val_loss'])
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION")
    print("="*60)
    print(f"Val Loss: {best['val_loss']:.4f}")
    print(f"Config: {best['config']}")
    
    return results

# Optuna for automatic tuning
def optuna_tuning():
    """
    Optuna for hyperparameter optimization
    """
    import optuna
    
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 768])
        n_action_steps = trial.suggest_int('n_action_steps', 5, 15)
        
        # Update config
        config.update({
            'learning_rate': lr,
            'batch_size': batch_size,
            'hidden_dim': hidden_dim,
            'n_action_steps': n_action_steps
        })
        
        # Train
        trainer = VLATrainer(config)
        trainer.train(train_loader, val_loader)
        
        # Return validation loss
        return trainer.best_val_loss
    
    # Create study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    # Best parameters
    print("\n" + "="*60)
    print("BEST PARAMETERS (Optuna)")
    print("="*60)
    print(f"Best Val Loss: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    
    return study
```

---

#### 학습 모니터링
```python
# monitoring.py

class TrainingMonitor:
    """
    학습 모니터링 및 진단
    """
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': []
        }
    
    def log_metrics(self, epoch, metrics):
        """
        Log metrics
        """
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def plot_training_curves(self):
        """
        Plot training curves
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.metrics['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.metrics['learning_rate'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Gradient norm
        axes[1, 0].plot(self.metrics['gradient_norm'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].grid(True)
        
        # Overfitting detection
        train_loss = np.array(self.metrics['train_loss'])
        val_loss = np.array(self.metrics['val_loss'])
        gap = val_loss - train_loss
        
        axes[1, 1].plot(gap)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val - Train Loss')
        axes[1, 1].set_title('Overfitting Detection')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/training_curves.png')
        plt.show()
    
    def check_overfitting(self):
        """
        Overfitting 감지
        """
        train_loss = np.array(self.metrics['train_loss'])
        val_loss = np.array(self.metrics['val_loss'])
        
        # Recent trend
        recent_window = 10
        if len(val_loss) >= recent_window:
            recent_train = train_loss[-recent_window:]
            recent_val = val_loss[-recent_window:]
            
            # Check if val loss is increasing while train loss decreases
            train_trend = np.polyfit(range(recent_window), recent_train, 1)[0]
            val_trend = np.polyfit(range(recent_window), recent_val, 1)[0]
            
            if train_trend < 0 and val_trend > 0:
                print("⚠️ Overfitting detected!")
                print(f"   Train trend: {train_trend:.6f}")
                print(f"   Val trend: {val_trend:.6f}")
                
                print("\n해결책:")
                print("1. 더 많은 데이터 수집")
                print("2. Data augmentation 강화")
                print("3. Dropout 증가")
                print("4. Weight decay 증가")
                print("5. Early stopping")
                
                return True
        
        return False

# 학습 중 체크리스트
monitoring_checklist = """
매 10 epoch마다 체크:
- [ ] Training loss 감소 중?
- [ ] Validation loss overfitting?
- [ ] Action 분포 정상? (시각화)
- [ ] Gradient norm 안정? (< 10)
- [ ] Learning rate 적절?
- [ ] GPU 메모리 사용률?

문제 발생 시:
┌─────────────────────────────────────────┐
│ Loss 폭발                               │
│ → LR 낮추기 (1/10)                      │
│ → Gradient clipping 확인                │
├─────────────────────────────────────────┤
│ Loss 정체                               │
│ → LR 높이기 or schedule 조정            │
│ → Augmentation 강화                     │
├─────────────────────────────────────────┤
│ Overfitting                             │
│ → Dropout 추가                          │
│ → Weight decay 증가                     │
│ → 데이터 추가 수집                      │
├─────────────────────────────────────────┤
│ Action 이상 (모두 비슷)                 │
│ → Action normalization 확인             │
│ → 데이터 불균형 체크                    │
└─────────────────────────────────────────┘
"""
```

**시간: 주 8-10시간 (학습 시간 포함)**

---

### Week 5-6: 평가 및 디버깅

#### 종합 평가 시스템
```python
# evaluation.py

class VLAEvaluator:
    """
    다차원 VLA 평가
    
    Metrics:
    1. Success Rate (가장 중요!)
    2. Completion Time
    3. Trajectory Smoothness
    4. Safety (collisions)
    5. Efficiency
    """
    
    def __init__(self, model, env, device='cuda'):
        self.model = model
        self.env = env
        self.device = device
        
        self.model.eval()
    
    def evaluate_episode(self, episode_data):
        """
        단일 에피소드 평가
        
        Args:
            episode_data: dict with trajectory info
        
        Returns:
            metrics: dict
        """
        metrics = {}
        
        # 1. Success
        metrics['success'] = episode_data['success']
        
        # 2. Completion Time
        metrics['time'] = episode_data['num_steps'] * 0.1  # seconds
        
        # 3. Trajectory Smoothness (jerk)
        actions = np.array(episode_data['actions'])
        velocities = np.diff(actions, axis=0)
        jerks = np.diff(velocities, axis=0)
        metrics['smoothness'] = -np.mean(np.abs(jerks))  # 낮을수록 부드러움
        metrics['max_jerk'] = np.max(np.abs(jerks))
        
        # 4. Safety
        metrics['num_collisions'] = episode_data['collision_count']
        metrics['max_joint_velocity'] = np.max(np.abs(velocities))
        
        # 5. Efficiency
        if episode_data['success']:
            optimal_time = 5.0  # seconds (baseline)
            metrics['efficiency'] = min(optimal_time / metrics['time'], 1.0)
        else:
            metrics['efficiency'] = 0.0
        
        # 6. Distance to goal (final)
        metrics['final_distance'] = episode_data.get('final_distance', 1.0)
        
        # 7. Action magnitude
        metrics['avg_action_magnitude'] = np.mean(np.abs(actions))
        
        return metrics
    
    def run_evaluation(self, num_episodes=20):
        """
        전체 평가 실행
        
        Args:
            num_episodes: number of test episodes
        
        Returns:
            summary: aggregated metrics
        """
        all_metrics = []
        
        print(f"\n{'='*60}")
        print(f"Running Evaluation ({num_episodes} episodes)")
        print(f"{'='*60}")
        
        for ep in range(num_episodes):
            print(f"\nEpisode {ep + 1}/{num_episodes}")
            
            # Reset environment
            obs = self.env.reset()
            done = False
            step = 0
            max_steps = 200
            
            episode_data = {
                'actions': [],
                'observations': [],
                'collision_count': 0,
                'success': False,
                'num_steps': 0
            }
            
            # Rollout
            action_buffer = []  # For action chunking
            
            while not done and step < max_steps:
                # Get action (with chunking)
                if len(action_buffer) == 0:
                    # Predict action chunk
                    obs_tensor = self.preprocess_observation(obs)
                    
                    with torch.no_grad():
                        action_chunk = self.model.predict(obs_tensor)
                    
                    action_buffer = list(action_chunk[0].cpu().numpy())
                
                # Pop next action
                action = action_buffer.pop(0)
                
                # Execute
                obs, reward, done, info = self.env.step(action)
                
                # Record
                episode_data['actions'].append(action)
                episode_data['observations'].append(obs)
                
                if info.get('collision'):
                    episode_data['collision_count'] += 1
                
                if info.get('success'):
                    episode_data['success'] = True
                    done = True
                
                step += 1
            
            episode_data['num_steps'] = step
            episode_data['final_distance'] = info.get('distance_to_goal', 1.0)
            
            # Evaluate episode
            metrics = self.evaluate_episode(episode_data)
            all_metrics.append(metrics)
            
            # Print
            status = "✅" if metrics['success'] else "❌"
            print(f"{status} Success: {metrics['success']}, "
                  f"Time: {metrics['time']:.2f}s, "
                  f"Collisions: {metrics['num_collisions']}, "
                  f"Smoothness: {metrics['smoothness']:.4f}")
        
        # Aggregate
        summary = self.aggregate_results(all_metrics)
        
        return summary, all_metrics
    
    def aggregate_results(self, all_metrics):
        """
        결과 집계 및 출력
        """
        summary = {
            'success_rate': np.mean([m['success'] for m in all_metrics]),
            'avg_time': np.mean([m['time'] for m in all_metrics]),
            'std_time': np.std([m['time'] for m in all_metrics]),
            'avg_smoothness': np.mean([m['smoothness'] for m in all_metrics]),
            'total_collisions': np.sum([m['num_collisions'] for m in all_metrics]),
            'avg_efficiency': np.mean([m['efficiency'] for m in all_metrics if m['efficiency'] > 0]),
            'avg_final_distance': np.mean([m['final_distance'] for m in all_metrics]),
            'max_jerk': np.max([m['max_jerk'] for m in all_metrics])
        }
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Success Rate:       {summary['success_rate']*100:5.1f}%")
        print(f"Avg Time:           {summary['avg_time']:5.2f}s (±{summary['std_time']:.2f})")
        print(f"Smoothness:         {summary['avg_smoothness']:5.4f}")
        print(f"Total Collisions:   {summary['total_collisions']:5.0f}")
        print(f"Avg Efficiency:     {summary['avg_efficiency']*100:5.1f}%")
        print(f"Avg Final Distance: {summary['avg_final_distance']:5.3f}m")
        print(f"Max Jerk:           {summary['max_jerk']:5.2f}")
        print("="*60)
        
        return summary
    
    def preprocess_observation(self, obs):
        """
        Preprocess observation for model
        """
        # Convert to tensor
        rgb = torch.FloatTensor(obs['rgb']).unsqueeze(0).to(self.device)
        proprio = torch.FloatTensor(obs['proprio']).unsqueeze(0).to(self.device)
        
        return {'rgb': rgb, 'proprio': proprio}
    
    def compare_checkpoints(self, checkpoint_paths):
        """
        여러 checkpoint 비교
        """
        results = {}
        
        for path in checkpoint_paths:
            print(f"\n{'='*60}")
            print(f"Evaluating: {path}")
            print(f"{'='*60}")
            
            # Load checkpoint
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate
            summary, _ = self.run_evaluation(num_episodes=10)
            results[path] = summary
        
        # Visualize comparison
        self.plot_comparison(results)
        
        return results
    
    def plot_comparison(self, results):
        """
        결과 비교 시각화
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        checkpoints = list(results.keys())
        
        metrics = [
            ('success_rate', 'Success Rate'),
            ('avg_time', 'Avg Time (s)'),
            ('avg_smoothness', 'Smoothness'),
            ('total_collisions', 'Total Collisions'),
            ('avg_efficiency', 'Efficiency'),
            ('avg_final_distance', 'Final Distance (m)')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            values = [results[c][metric] for c in checkpoints]
            
            ax.bar(range(len(checkpoints)), values)
            ax.set_title(title)
            ax.set_xticks(range(len(checkpoints)))
            ax.set_xticklabels([f'CP{i+1}' for i in range(len(checkpoints))], rotation=45)
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('checkpoint_comparison.png')
        plt.show()

# Generalization Test
def test_generalization(model, env):
    """
    학습 조건과 다른 환경에서 테스트
    """
    test_conditions = [
        {
            'name': 'baseline',
            'lighting': 'normal',
            'box_size': 'medium',
            'clutter': 'none'
        },
        {
            'name': 'bright_light',
            'lighting': 'bright',
            'box_size': 'medium',
            'clutter': 'none'
        },
        {
            'name': 'dim_light',
            'lighting': 'dim',
            'box_size': 'medium',
            'clutter': 'none'
        },
        {
            'name': 'large_box',
            'lighting': 'normal',
            'box_size': 'large',
            'clutter': 'none'
        },
        {
            'name': 'small_box',
            'lighting': 'normal',
            'box_size': 'small',
            'clutter': 'none'
        },
        {
            'name': 'high_clutter',
            'lighting': 'normal',
            'box_size': 'medium',
            'clutter': 'high'
        },
        {
            'name': 'combined_hard',
            'lighting': 'dim',
            'box_size': 'small',
            'clutter': 'medium'
        }
    ]
    
    results = {}
    evaluator = VLAEvaluator(model, env)
    
    for condition in test_conditions:
        print(f"\n{'='*60}")
        print(f"Testing: {condition['name']}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")
        
        # Setup environment
        env.configure(condition)
        
        # Evaluate
        summary, _ = evaluator.run_evaluation(num_episodes=10)
        results[condition['name']] = summary
    
    # Plot generalization
    plot_generalization_results(results)
    
    return results

def plot_generalization_results(results):
    """
    Generalization 결과 시각화
    """
    import matplotlib.pyplot as plt
    
    conditions = list(results.keys())
    success_rates = [results[c]['success_rate'] for c in conditions]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(conditions)), success_rates)
    
    # Color code by performance
    for i, bar in enumerate(bars):
        if success_rates[i] > 0.7:
            bar.set_color('green')
        elif success_rates[i] > 0.5:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0.7, color='r', linestyle='--', label='Target (70%)')
    plt.xticks(range(len(conditions)), conditions, rotation=45, ha='right')
    plt.ylabel('Success Rate')
    plt.title('Generalization Performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('generalization_results.png')
    plt.show()
    
    # Analysis
    baseline = results['baseline']['success_rate']
    
    print("\n" + "="*60)
    print("GENERALIZATION ANALYSIS")
    print("="*60)
    print(f"Baseline: {baseline*100:.1f}%")
    print(f"\nPerformance Drops:")
    for name, result in results.items():
        if name != 'baseline':
            drop = (baseline - result['success_rate']) * 100
            print(f"  {name:20s}: {drop:+5.1f}%")
```

---

#### 디버깅 전략
```python
# debugging.py

class VLADebugger:
    """
    VLA 디버깅 도구
    """
    
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
    
    def debug_training(self):
        """
        학습 디버깅
        
        문제 1: Loss가 안 떨어짐
        """
        print("="*60)
        print("DEBUG: Training Issues")
        print("="*60)
        
        # Step 1: Overfit single batch
        print("\nStep 1: Overfitting single batch...")
        
        one_batch = next(iter(self.dataloader))
        obs, actions = one_batch
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        losses = []
        for i in range(1000):
            pred = self.model(obs)
            loss = criterion(pred, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if i % 100 == 0:
                print(f"  Step {i:4d}: Loss = {loss.item():.6f}")
        
        if losses[-1] < 0.01:
            print("✅ Model CAN learn (loss → 0)")
            print("   Problem is likely in data or training setup")
        else:
            print("❌ Model CANNOT learn from this batch")
            print("   Check:")
            print("   - Model architecture")
            print("   - Loss function")
            print("   - Learning rate (try 10x higher)")
        
        # Step 2: Data sanity check
        print("\nStep 2: Data sanity check...")
        
        print(f"  Observation keys: {obs.keys()}")
        print(f"  RGB shape: {obs['rgb'].shape}")
        print(f"  Action shape: {actions.shape}")
        print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"  Action mean: {actions.mean():.3f}")
        print(f"  Action std: {actions.std():.3f}")
        
        if actions.std() < 0.01:
            print("⚠️ Actions have very low variance!")
            print("   Data collection might be too uniform")
        
        # Step 3: Gradient check
        print("\nStep 3: Gradient check...")
        
        self.model.zero_grad()
        pred = self.model(obs)
        loss = criterion(pred, actions)
        loss.backward()
        
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                
                if grad_norm > 100:
                    print(f"⚠️ Large gradient: {name:40s} = {grad_norm:.2f}")
                elif grad_norm < 1e-7:
                    print(f"⚠️ Tiny gradient: {name:40s} = {grad_norm:.2e}")
        
        # Step 4: Visualize predictions
        print("\nStep 4: Visualizing predictions...")
        self.visualize_predictions()
    
    def visualize_predictions(self):
        """
        예측 시각화로 문제 파악
        """
        import matplotlib.pyplot as plt
        
        self.model.eval()
        
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        
        with torch.no_grad():
            for idx, (obs, gt_actions) in enumerate(self.dataloader):
                if idx >= 5:
                    break
                
                # Predict
                pred_actions = self.model(obs)
                
                # First timestep only (for action chunking)
                gt_action = gt_actions[0, 0].cpu().numpy()
                pred_action = pred_actions[0, 0].cpu().numpy()
                
                # Row 1: Observation
                axes[0, idx].imshow(obs['rgb'][0].permute(1, 2, 0).cpu())
                axes[0, idx].set_title(f'Observation {idx+1}')
                axes[0, idx].axis('off')
                
                # Row 2: Ground truth action
                axes[1, idx].bar(range(7), gt_action)
                axes[1, idx].set_title('GT Action')
                axes[1, idx].set_ylim([-1, 1])
                axes[1, idx].set_xticks(range(7))
                axes[1, idx].set_xticklabels([f'J{i+1}' for i in range(7)])
                
                # Row 3: Predicted action
                axes[2, idx].bar(range(7), pred_action)
                axes[2, idx].set_title('Pred Action')
                axes[2, idx].set_ylim([-1, 1])
                axes[2, idx].set_xticks(range(7))
                axes[2, idx].set_xticklabels([f'J{i+1}' for i in range(7)])
        
        plt.tight_layout()
        plt.savefig('prediction_visualization.png')
        plt.show()
        
        # 패턴 분석
        print("\n관찰 사항:")
        print("- 모든 action이 비슷? → 모델이 학습 안 됨")
        print("- 특정 joint만 움직임? → 데이터 불균형")
        print("- 값이 너무 큼/작음? → Normalization 문제")
        print("- GT와 완전히 다름? → 모델 capacity 부족")
    
    def detect_overfitting(self, train_losses, val_losses):
        """
        Overfitting 감지 및 해결
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_curves.png')
        plt.show()
        
        # Overfitting detection
        if len(val_losses) > 10:
            recent_val = val_losses[-10:]
            if recent_val[-1] > recent_val[0]:
                print("⚠️ Overfitting detected!")
                
                solutions = """
                해결책:
                1. 더 많은 데이터 수집 (가장 효과적!)
                2. Data augmentation 강화
                3. Dropout 추가/증가
                   - 현재 dropout=0.1 → 0.2로
                4. Weight decay 증가
                   - 현재 weight_decay=1e-4 → 1e-3로
                5. 모델 크기 줄이기
                   - hidden_dim: 512 → 256
                6. Early stopping 적용
                   - patience=20
                7. Ensemble (여러 checkpoint 평균)
                """
                print(solutions)
                
                # Early stopping checkpoint
                best_epoch = np.argmin(val_losses)
                print(f"\n최적 checkpoint: Epoch {best_epoch+1}")
                print(f"Val Loss: {val_losses[best_epoch]:.4f}")
    
    def analyze_failure_cases(self, failed_episodes):
        """
        실패 케이스 분석
        """
        print("\n" + "="*60)
        print("FAILURE CASE ANALYSIS")
        print("="*60)
        
        # Classify failures
        failure_types = {
            'grasp_failure': 0,
            'collision': 0,
            'trajectory_deviation': 0,
            'timeout': 0,
            'other': 0
        }
        
        for episode in failed_episodes:
            failure_type = self.classify_failure(episode)
            failure_types[failure_type] += 1
        
        # Report
        total = len(failed_episodes)
        for failure_type, count in failure_types.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{failure_type:20s}: {count:3d} ({percentage:5.1f}%)")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if failure_types['grasp_failure'] > total * 0.3:
            print("Main issue: Grasp Failure")
            print("Solutions:")
            print("- Collect more grasp demonstrations")
            print("- Add grasp-specific augmentation")
            print("- Check gripper control")
        
        if failure_types['collision'] > total * 0.3:
            print("Main issue: Collisions")
            print("Solutions:")
            print("- Add safety constraints")
            print("- Reduce action magnitude limits")
            print("- Improve obstacle avoidance data")
        
        if failure_types['trajectory_deviation'] > total * 0.3:
            print("Main issue: Trajectory Deviation")
            print("Solutions:")
            print("- Increase action chunking length")
            print("- Add trajectory tracking loss")
            print("- Collect more diverse trajectories")
    
    def classify_failure(self, episode):
        """
        실패 유형 분류
        """
        # 간단한 휴리스틱 기반 분류
        if episode.get('gripper_empty') and episode.get('attempted_grasp'):
            return 'grasp_failure'
        
        if episode.get('collision_count', 0) > 0:
            return 'collision'
        
        if episode.get('max_deviation', 0) > 0.5:
            return 'trajectory_deviation'
        
        if episode.get('timed_out'):
            return 'timeout'
        
        return 'other'
```

**기대 결과 및 개선 가이드**
```
첫 시도 목표:
┌────────────────────────────────┐
│ Success Rate:  40-60%          │
│ Completion Time: < 10s         │
│ Smooth trajectory              │
│ Zero collisions (safety)       │
└────────────────────────────────┘

개선 목표 (데이터/모델 튜닝 후):
┌────────────────────────────────┐
│ Success Rate:  70%+            │
│ Completion Time: 5-7s          │
│ Smooth & efficient             │
│ Robust to variations           │
└────────────────────────────────┘

60% 미만이면:
→ 더 많은 데이터 (100+ episodes)
→ Hyperparameter tuning
→ Action space 재정의
→ Observation 개선
```

**시간: 주 8-10시간**

---

## Month 11-12: 고도화 및 ROS2 통합

### Week 1-2: 실패 복구 시스템
```python
# failure_recovery.py

class FailureRecovery:
    """
    실패 감지 및 복구
    
    목적:
    - 실패 상황 자동 감지
    - 복구 action 생성
    - Retry logic
    - Safety monitoring
    """
    
    def __init__(self, policy, robot):
        self.policy = policy
        self.robot = robot
        self.max_retries = 3
        
        # Failure detectors
        self.collision_detector = CollisionDetector(robot)
        self.stuck_detector = StuckDetector(robot)
        self.grasp_detector = GraspDetector(robot)
    
    def detect_failure(self, obs, action, robot_state):
        """
        실패 상황 감지
        
        Returns:
            failure_type: str or None
            confidence: float (0-1)
        """
        failures = []
        
        # 1. Collision
        if self.collision_detector.check(robot_state):
            failures.append(('collision', 0.9))
        
        # 2. Grasp failure
        if self.grasp_detector.check_failure(obs, robot_state):
            failures.append(('grasp_failure', 0.8))
        
        # 3. Stuck
        if self.stuck_detector.check(robot_state):
            failures.append(('stuck', 0.7))
        
        # 4. Out of workspace
        if not self.is_in_workspace(robot_state['ee_position']):
            failures.append(('out_of_bounds', 0.9))
        
        # 5. Trajectory deviation (if tracking)
        if hasattr(self, 'expected_trajectory'):
            deviation = self.compute_deviation(robot_state)
            if deviation > 0.2:  # 20cm
                failures.append(('trajectory_deviation', 0.6))
        
        # Return highest confidence failure
        if failures:
            failures.sort(key=lambda x: x[1], reverse=True)
            return failures[0]
        
        return None, 0.0
    
    def recover(self, failure_type, robot_state):
        """
        실패 유형별 복구 전략
        
        Args:
            failure_type: str
            robot_state: dict
        
        Returns:
            recovery_actions: list of actions
        """
        recovery_strategies = {
            'collision': self.recover_from_collision,
            'grasp_failure': self.retry_grasp,
            'stuck': self.jiggle,
            'out_of_bounds': self.return_to_safe_zone,
            'trajectory_deviation': self.replan
        }
        
        recovery_fn = recovery_strategies.get(failure_type)
        
        if recovery_fn:
            return recovery_fn(robot_state)
        
        return None
    
    def recover_from_collision(self, robot_state):
        """
        충돌 복구: 뒤로 후진
        """
        # 마지막 action의 반대 방향
        last_action = robot_state.get('last_action', np.zeros(7))
        
        # Retreat action (50% reverse)
        recovery_action = -0.5 * last_action
        
        # Execute for a few steps
        recovery_actions = [recovery_action] * 5
        
        return recovery_actions
    
    def retry_grasp(self, robot_state):
        """
        Grasp 재시도: 약간 다른 각도로
        """
        # Current EE pose
        current_pose = robot_state['ee_pose']
        
        # Add small random offset
        position_offset = np.random.normal(0, 0.02, 3)  # ±2cm
        orientation_offset = np.random.normal(0, 0.1, 4)  # small rotation
        
        # Compute recovery trajectory
        recovery_actions = self.plan_to_pose(
            current_pose[:3] + position_offset,
            current_pose[3:] + orientation_offset
        )
        
        # Add closing gripper at end
        recovery_actions.append(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        
        return recovery_actions
    
    def jiggle(self, robot_state):
        """
        Stuck 해제: 작은 랜덤 움직임
        """
        jiggle_actions = []
        
        for _ in range(3):
            action = np.random.uniform(-0.1, 0.1, 7)
            jiggle_actions.append(action)
        
        return jiggle_actions
    
    def return_to_safe_zone(self, robot_state):
        """
        안전 영역으로 복귀
        """
        # Define safe pose (home position)
        safe_joint_positions = np.array([0, -0.5, 0, -1.5, 0, 1.0, 0])
        
        # Current position
        current = robot_state['joint_positions']
        
        # Plan trajectory to safe pose
        recovery_actions = self.plan_joint_trajectory(
            current,
            safe_joint_positions,
            num_steps=20
        )
        
        return recovery_actions
    
    def replan(self, robot_state):
        """
        재계획: 새로운 trajectory 생성
        """
        # Get current observation
        obs = self.get_observation()
        
        # Re-predict with policy
        with torch.no_grad():
            new_actions = self.policy.predict(obs)
        
        return list(new_actions.cpu().numpy())
    
    def execute_with_recovery(self, obs, max_attempts=3):
        """
        복구 로직 포함 실행
        
        Main execution loop with failure handling
        """
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            
            # Normal execution
            action = self.policy.predict(obs)
            robot_state = self.execute_action(action)
            
            # Check for failure
            failure_type, confidence = self.detect_failure(
                obs, action, robot_state
            )
            
            if failure_type and confidence > 0.7:
                print(f"⚠️ Failure detected: {failure_type} "
                      f"(confidence: {confidence:.2f})")
                print("   Attempting recovery...")
                
                # Recovery
                recovery_actions = self.recover(failure_type, robot_state)
                
                if recovery_actions:
                    # Execute recovery
                    for rec_action in recovery_actions:
                        self.execute_action(rec_action)
                    
                    # Wait for stabilization
                    time.sleep(0.5)
                    
                    # Get new observation
                    obs = self.get_observation()
                    
                    # Retry
                    continue
                else:
                    print("   No recovery strategy available")
                    return False
            
            # Success
            return True
        
        print("❌ All recovery attempts failed")
        return False

class CollisionDetector:
    """
    충돌 감지
    """
    def __init__(self, robot):
        self.robot = robot
    
    def check(self, robot_state):
        # Check contact forces
        contact_forces = robot_state.get('contact_forces', np.zeros(7))
        
        # Threshold
        threshold = 10.0  # Newtons
        
        return np.any(np.abs(contact_forces) > threshold)

class StuckDetector:
    """
    Stuck 감지
    """
    def __init__(self, robot, window=10):
        self.robot = robot
        self.window = window
        self.velocity_history = deque(maxlen=window)
    
    def check(self, robot_state):
        velocity = robot_state.get('joint_velocities', np.zeros(7))
        self.velocity_history.append(np.linalg.norm(velocity))
        
        if len(self.velocity_history) < self.window:
            return False
        
        # Check if velocity is consistently low
        avg_velocity = np.mean(self.velocity_history)
        
        return avg_velocity < 0.01  # Very slow

class GraspDetector:
    """
    Grasp 성공/실패 감지
    """
    def __init__(self, robot):
        self.robot = robot
    
    def check_success(self, robot_state):
        """
        Grasp 성공 여부
        """
        gripper_state = robot_state.get('gripper_state', 0)
        gripper_force = robot_state.get('gripper_force', 0)
        
        # Gripper closed and force detected
        return gripper_state > 0.8 and gripper_force > 1.0
    
    def check_failure(self, obs, robot_state):
        """
        Grasp 실패 감지
        """
        # Gripper closed but no object detected
        gripper_state = robot_state.get('gripper_state', 0)
        gripper_force = robot_state.get('gripper_force', 0)
        
        return gripper_state > 0.8 and gripper_force < 0.5
```

---

### Week 3-4: Safety Layer
```python
# safety_layer.py

class SafetyLayer:
    """
    로봇 안전 시스템
    
    기능:
    1. Action validation
    2. Joint limit enforcement
    3. Velocity/acceleration limits
    4. Workspace boundaries
    5. Collision prediction
    6. Emergency stop
    """
    
    def __init__(self, robot):
        self.robot = robot
        
        # Limits
        self.joint_limits = robot.get_joint_limits()
        self.velocity_limits = np.array([2.0] * 7)  # rad/s
        self.acceleration_limits = np.array([5.0] * 7)  # rad/s^2
        
        # Workspace boundaries
        self.workspace_bounds = {
            'x': [0.2, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.0, 1.0]
        }
        
        # History for acceleration check
        self.action_history = deque(maxlen=10)
        
        # Statistics
        self.violations = {
            'joint_limits': 0,
            'velocity': 0,
            'acceleration': 0,
            'workspace': 0,
            'singularity': 0
        }
    
    def check_action(self, action, current_state):
        """
        Action 안전성 검증
        
        Args:
            action: proposed action
            current_state: current robot state
        
        Returns:
            is_safe: bool
            warnings: list of warning strings
        """
        warnings = []
        
        # 1. Joint limits
        predicted_joints = current_state['joint_pos'] + action
        
        if np.any(predicted_joints < self.joint_limits[:, 0]) or \
           np.any(predicted_joints > self.joint_limits[:, 1]):
            warnings.append("Joint limits violated")
            self.violations['joint_limits'] += 1
        
        # 2. Velocity limits (assuming 10Hz control)
        velocity = action / 0.1
        if np.any(np.abs(velocity) > self.velocity_limits):
            warnings.append("Velocity limits violated")
            self.violations['velocity'] += 1
        
        # 3. Acceleration limits
        if len(self.action_history) > 0:
            last_velocity = self.action_history[-1] / 0.1
            acceleration = (velocity - last_velocity) / 0.1
            
            if np.any(np.abs(acceleration) > self.acceleration_limits):
                warnings.append("Acceleration limits violated")
                self.violations['acceleration'] += 1
        
        # 4. Workspace bounds
        ee_pos = self.robot.forward_kinematics(predicted_joints)
        if not self.is_in_workspace(ee_pos):
            warnings.append("Out of workspace")
            self.violations['workspace'] += 1
        
        # 5. Singularity check
        if self.is_near_singularity(predicted_joints):
            warnings.append("Near singularity")
            self.violations['singularity'] += 1
        
        # Record action
        self.action_history.append(action)
        
        is_safe = len(warnings) == 0
        
        return is_safe, warnings
    
    def clip_action(self, action, current_state):
        """
        Unsafe action을 safe하게 수정
        
        Returns:
            clipped_action: modified safe action
        """
        clipped = action.copy()
        
        # 1. Joint limits
        predicted = current_state['joint_pos'] + clipped
        
        # Clip to stay within limits
        clipped = np.clip(
            clipped,
            self.joint_limits[:, 0] - current_state['joint_pos'],
            self.joint_limits[:, 1] - current_state['joint_pos']
        )
        
        # 2. Velocity limits
        max_delta = self.velocity_limits * 0.1  # 10Hz
        clipped = np.clip(clipped, -max_delta, max_delta)
        
        # 3. Acceleration limits
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            max_change = self.acceleration_limits * 0.1 * 0.1  # 10Hz, dt^2
            
            delta_action = clipped - last_action
            delta_action = np.clip(delta_action, -max_change, max_change)
            clipped = last_action + delta_action
        
        return clipped
    
    def is_in_workspace(self, position):
        """
        Check if position is in workspace
        """
        x, y, z = position
        
        return (self.workspace_bounds['x'][0] <= x <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= y <= self.workspace_bounds['y'][1] and
                self.workspace_bounds['z'][0] <= z <= self.workspace_bounds['z'][1])
    
    def is_near_singularity(self, joint_positions):
        """
        Check if configuration is near singularity
        
        Method: Compute Jacobian and check condition number
        """
        jacobian = self.robot.compute_jacobian(joint_positions)
        
        # Compute condition number
        try:
            _, s, _ = np.linalg.svd(jacobian)
            condition_number = s[0] / s[-1] if s[-1] > 1e-10 else float('inf')
            
            # Threshold
            threshold = 100
            
            return condition_number > threshold
        except:
            return False
    
    def emergency_stop(self):
        """
        비상 정지
        """
        print("🛑 EMERGENCY STOP ACTIVATED")
        
        # Stop all motion
        self.robot.stop()
        
        # Hold position
        self.robot.hold_position()
        
        # Log
        print(f"   Violation history: {self.violations}")
    
    def get_statistics(self):
        """
        Safety statistics
        """
        total = sum(self.violations.values())
        
        print("\n" + "="*60)
        print("SAFETY STATISTICS")
        print("="*60)
        print(f"Total violations: {total}")
        for violation_type, count in self.violations.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {violation_type:20s}: {count:4d} ({percentage:5.1f}%)")
        print("="*60)

# Predictive Safety
class PredictiveSafety:
    """
    예측 기반 안전 시스템
    
    미래 trajectory를 예측하여 충돌 방지
    """
    
    def __init__(self, robot, env):
        self.robot = robot
        self.env = env
        self.prediction_horizon = 10  # steps
    
    def predict_trajectory(self, current_state, action_sequence):
        """
        Action sequence로부터 trajectory 예측
        
        Args:
            current_state: current robot state
            action_sequence: sequence of actions
        
        Returns:
            trajectory: list of predicted states
        """
        trajectory = []
        state = current_state.copy()
        
        for action in action_sequence:
            # Predict next state (using robot dynamics)
            next_state = self.robot.predict_next_state(state, action)
            trajectory.append(next_state)
            state = next_state
        
        return trajectory
    
    def check_collision_free(self, trajectory):
        """
        Trajectory가 충돌 없는지 확인
        
        Args:
            trajectory: list of states
        
        Returns:
            is_safe: bool
            first_collision_step: int or None
        """
        for step, state in enumerate(trajectory):
            # Check collision at this state
            if self.env.check_collision(state):
                return False, step
        
        return True, None
    
    def replan_if_unsafe(self, action_sequence):
        """
        Unsafe trajectory면 re-planning
        
        Args:
            action_sequence: proposed actions
        
        Returns:
            safe_actions: modified safe actions
        """
        current_state = self.robot.get_current_state()
        
        # Predict trajectory
        trajectory = self.predict_trajectory(current_state, action_sequence)
        
        # Check safety
        is_safe, collision_step = self.check_collision_free(trajectory)
        
        if not is_safe:
            print(f"⚠️ Predicted collision at step {collision_step}")
            print("   Re-planning...")
            
            # Truncate unsafe part
            safe_actions = action_sequence[:collision_step]
            
            # Add stop action
            safe_actions.append(np.zeros_like(action_sequence[0]))
            
            return safe_actions
        
        return action_sequence
```

**시간: 주 6-8시간**

### Week 5-6: ROS2 완전 통합

#### VLA ROS2 Node 구현
```python
# vla_node.py
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from diagnostic_updater import Updater, DiagnosticStatusWrapper
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

import torch
import numpy as np
from cv_bridge import CvBridge

class VLALifecycleNode(LifecycleNode):
    """
    VLA Lifecycle Node
    
    ROS2 Lifecycle 패턴 활용:
    - Unconfigured → Configuring → Inactive
    - Inactive → Activating → Active
    - Active → Deactivating → Inactive
    - Cleanup, Shutdown
    
    ROS2 경험 활용!
    """
    
    def __init__(self):
        super().__init__('vla_node')
        
        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('action_chunk_size', 10)
        
        # State
        self.model = None
        self.latest_image = None
        self.latest_joint_state = None
        self.action_buffer = []
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Diagnostics
        self.diagnostics = None
        
        self.get_logger().info('VLA Node created')
    
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Configure state: Setup resources
        """
        self.get_logger().info('Configuring VLA Node...')
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        device = self.get_parameter('device').value
        self.control_freq = self.get_parameter('control_frequency').value
        
        # Load VLA model
        try:
            self.model = self.load_model(model_path, device)
            self.get_logger().info(f'Model loaded from {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return TransitionCallbackReturn.FAILURE
        
        # Create callback group (for parallel callbacks)
        self.callback_group = ReentrantCallbackGroup()
        
        # Publishers
        self.joint_cmd_pub = self.create_lifecycle_publisher(
            JointState,
            '/joint_commands',
            10
        )
        
        self.status_pub = self.create_lifecycle_publisher(
            Bool,
            '/vla/status',
            10
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        
        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Diagnostics
        self.diagnostics = Updater(self)
        self.diagnostics.setHardwareID('VLA-Robot')
        self.diagnostics.add('VLA Status', self.diagnostic_callback)
        
        # Safety layer
        from safety_layer import SafetyLayer
        self.safety = SafetyLayer(robot=None)  # Initialize with actual robot
        
        # Failure recovery
        from failure_recovery import FailureRecovery
        self.recovery = FailureRecovery(policy=self.model, robot=None)
        
        self.get_logger().info('Configuration complete')
        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Activate state: Start operations
        """
        self.get_logger().info('Activating VLA Node...')
        
        # Activate publishers
        self.joint_cmd_pub.on_activate()
        self.status_pub.on_activate()
        
        # Start control loop
        self.control_timer = self.create_timer(
            1.0 / self.control_freq,
            self.control_loop,
            callback_group=self.callback_group
        )
        
        # Start diagnostics
        self.diag_timer = self.create_timer(
            1.0,
            lambda: self.diagnostics.update()
        )
        
        self.get_logger().info('VLA Node activated')
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Deactivate state: Stop operations
        """
        self.get_logger().info('Deactivating VLA Node...')
        
        # Stop control loop
        self.control_timer.cancel()
        self.diag_timer.cancel()
        
        # Deactivate publishers
        self.joint_cmd_pub.on_deactivate()
        self.status_pub.on_deactivate()
        
        # Clear action buffer
        self.action_buffer = []
        
        self.get_logger().info('VLA Node deactivated')
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Cleanup state: Release resources
        """
        self.get_logger().info('Cleaning up VLA Node...')
        
        # Destroy publishers/subscribers
        self.destroy_publisher(self.joint_cmd_pub)
        self.destroy_publisher(self.status_pub)
        self.destroy_subscription(self.image_sub)
        self.destroy_subscription(self.joint_state_sub)
        
        # Unload model
        self.model = None
        
        self.get_logger().info('Cleanup complete')
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Shutdown state
        """
        self.get_logger().info('Shutting down VLA Node...')
        return TransitionCallbackReturn.SUCCESS
    
    def image_callback(self, msg):
        """
        Camera image callback
        """
        try:
            # Convert ROS Image to numpy
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
    
    def joint_state_callback(self, msg):
        """
        Joint state callback
        """
        self.latest_joint_state = msg
    
    def control_loop(self):
        """
        Main control loop (10Hz)
        """
        # Check if data available
        if self.latest_image is None or self.latest_joint_state is None:
            return
        
        # Prepare observation
        obs = self.prepare_observation()
        
        # Get action (with chunking)
        if len(self.action_buffer) == 0:
            # Predict new action chunk
            with torch.no_grad():
                action_chunk = self.model.predict(obs)
                self.action_buffer = list(action_chunk.cpu().numpy())
        
        # Pop next action
        action = self.action_buffer.pop(0)
        
        # Safety check
        current_state = {
            'joint_pos': np.array(self.latest_joint_state.position),
            'joint_vel': np.array(self.latest_joint_state.velocity),
        }
        
        is_safe, warnings = self.safety.check_action(action, current_state)
        
        if not is_safe:
            self.get_logger().warn(f'Unsafe action detected: {warnings}')
            action = self.safety.clip_action(action, current_state)
        
        # Publish command
        self.publish_joint_command(action)
        
        # Update status
        status_msg = Bool()
        status_msg.data = True
        self.status_pub.publish(status_msg)
    
    def prepare_observation(self):
        """
        Prepare observation for model
        """
        # Preprocess image
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(self.latest_image).unsqueeze(0)
        
        # Proprioception
        joint_pos = torch.FloatTensor(self.latest_joint_state.position).unsqueeze(0)
        joint_vel = torch.FloatTensor(self.latest_joint_state.velocity).unsqueeze(0)
        
        # Combine
        proprio = torch.cat([joint_pos, joint_vel], dim=-1)
        
        obs = {
            'rgb': image_tensor.to(self.model.device),
            'proprio': proprio.to(self.model.device)
        }
        
        return obs
    
    def publish_joint_command(self, action):
        """
        Publish joint command
        """
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = self.latest_joint_state.name
        
        # Convert delta to absolute positions
        current_pos = np.array(self.latest_joint_state.position)
        target_pos = current_pos + action
        
        cmd_msg.position = target_pos.tolist()
        
        self.joint_cmd_pub.publish(cmd_msg)
    
    def diagnostic_callback(self, stat: DiagnosticStatusWrapper):
        """
        Diagnostics updater callback
        """
        # Overall status
        if self.latest_image is not None and self.latest_joint_state is not None:
            stat.summary(DiagnosticStatusWrapper.OK, "VLA operational")
        else:
            stat.summary(DiagnosticStatusWrapper.WARN, "Waiting for data")
        
        # Add diagnostic info
        stat.add("Image received", str(self.latest_image is not None))
        stat.add("Joint state received", str(self.latest_joint_state is not None))
        stat.add("Action buffer size", str(len(self.action_buffer)))
        stat.add("Model device", str(self.model.device if self.model else "None"))
        
        # Safety statistics
        if hasattr(self.safety, 'violations'):
            for key, val in self.safety.violations.items():
                stat.add(f"Safety/{key}", str(val))
        
        return stat
    
    def load_model(self, model_path, device):
        """
        Load VLA model
        """
        from act_model import ACTPolicy
        
        checkpoint = torch.load(model_path, map_location=device)
        
        model = ACTPolicy(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model

def main(args=None):
    rclpy.init(args=args)
    
    node = VLALifecycleNode()
    
    # Executor with multiple threads
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

#### Launch 파일
```python
# vla_bringup.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
from lifecycle_msgs.msg import Transition

def generate_launch_description():
    # Declare arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/path/to/model.pt',
        description='Path to VLA model checkpoint'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device for model inference'
    )
    
    use_sim_arg = DeclareLaunchArgument(
        'use_sim',
        default_value='true',
        description='Use simulation'
    )
    
    # VLA Node (Lifecycle)
    vla_node = LifecycleNode(
        package='vla_control',
        executable='vla_node',
        name='vla_node',
        namespace='',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'device': LaunchConfiguration('device'),
            'control_frequency': 10.0,
            'action_chunk_size': 10
        }],
        output='screen'
    )
    
    # Camera Node (if simulation)
    camera_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam',
        name='camera',
        condition=IfCondition(LaunchConfiguration('use_sim')),
        output='screen'
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': 'robot.urdf'}],
        output='screen'
    )
    
    # Lifecycle transitions
    # Configure
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(vla_node),
            transition_id=Transition.TRANSITION_CONFIGURE
        )
    )
    
    # Activate after configured
    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=vla_node,
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(vla_node),
                        transition_id=Transition.TRANSITION_ACTIVATE
                    )
                )
            ]
        )
    )
    
    return LaunchDescription([
        model_path_arg,
        device_arg,
        use_sim_arg,
        vla_node,
        camera_node,
        robot_state_publisher,
        configure_event,
        activate_event
    ])
```

---

#### Integration with Navigation (Nav2)
```python
# vla_nav_integration.py
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
import rclpy

class VLANavigationIntegration:
    """
    VLA + Nav2 통합
    
    시나리오:
    1. Nav2로 목표 위치까지 이동
    2. VLA로 manipulation 수행
    3. 다음 목표로 이동
    """
    
    def __init__(self):
        rclpy.init()
        
        # Nav2 navigator
        self.navigator = BasicNavigator()
        
        # VLA client
        self.vla_client = VLAActionClient()
        
        # Wait for navigation to be ready
        self.navigator.waitUntilNav2Active()
    
    def execute_pick_and_place_task(self, waypoints):
        """
        Complete pick and place task
        
        Args:
            waypoints: list of (nav_pose, manipulation_task)
        """
        for nav_pose, task in waypoints:
            print(f"\n{'='*60}")
            print(f"Waypoint: {task['name']}")
            print(f"{'='*60}")
            
            # 1. Navigate to pose
            print("Step 1: Navigating to target...")
            self.navigate_to_pose(nav_pose)
            
            # 2. Execute manipulation
            print("Step 2: Executing manipulation...")
            success = self.execute_manipulation(task)
            
            if not success:
                print(f"❌ Manipulation failed at {task['name']}")
                return False
            
            print(f"✅ Completed {task['name']}")
        
        print("\n🎉 All tasks completed!")
        return True
    
    def navigate_to_pose(self, pose: PoseStamped):
        """
        Navigate to target pose using Nav2
        """
        self.navigator.goToPose(pose)
        
        # Wait for navigation to complete
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            
            # Print progress
            if feedback:
                print(f"  Distance remaining: {feedback.distance_remaining:.2f}m")
            
            rclpy.spin_once(self.navigator, timeout_sec=0.1)
        
        result = self.navigator.getResult()
        
        if result == TaskResult.SUCCEEDED:
            print("  ✅ Navigation succeeded")
            return True
        else:
            print(f"  ❌ Navigation failed: {result}")
            return False
    
    def execute_manipulation(self, task):
        """
        Execute manipulation using VLA
        """
        # Send goal to VLA action server
        goal = VLAGoal()
        goal.task_type = task['type']  # 'pick' or 'place'
        goal.target_object = task.get('object', '')
        
        future = self.vla_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.vla_client, future)
        
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            print("  ❌ VLA goal rejected")
            return False
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.vla_client, result_future)
        
        result = result_future.result().result
        
        return result.success

# Usage
def main():
    integration = VLANavigationIntegration()
    
    # Define waypoints
    waypoints = [
        (create_pose(x=1.0, y=0.0), {'name': 'Shelf A', 'type': 'pick', 'object': 'box_1'}),
        (create_pose(x=3.0, y=2.0), {'name': 'Pallet B', 'type': 'place'}),
        (create_pose(x=1.0, y=0.0), {'name': 'Shelf A', 'type': 'pick', 'object': 'box_2'}),
        (create_pose(x=3.0, y=2.0), {'name': 'Pallet B', 'type': 'place'}),
    ]
    
    # Execute
    integration.execute_pick_and_place_task(waypoints)

def create_pose(x, y, theta=0.0):
    """
    Helper function to create PoseStamped
    """
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = x
    pose.pose.position.y = y
    
    # Quaternion from yaw
    from tf_transformations import quaternion_from_euler
    q = quaternion_from_euler(0, 0, theta)
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    
    return pose
```

---

#### ROS2 Action Server
```python
# vla_action_server.py
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from vla_interfaces.action import VLAManipulation

class VLAActionServer(Node):
    """
    VLA Action Server
    
    Action definition (vla_interfaces/action/VLAManipulation.action):
    
    # Goal
    string task_type  # 'pick', 'place', 'move'
    string target_object
    geometry_msgs/Pose target_pose
    ---
    # Result
    bool success
    string message
    ---
    # Feedback
    float32 progress
    string current_phase
    """
    
    def __init__(self, vla_model):
        super().__init__('vla_action_server')
        
        self.vla_model = vla_model
        
        self._action_server = ActionServer(
            self,
            VLAManipulation,
            'vla_manipulation',
            self.execute_callback
        )
        
        self.get_logger().info('VLA Action Server started')
    
    def execute_callback(self, goal_handle):
        """
        Execute action callback
        """
        self.get_logger().info('Executing VLA manipulation...')
        
        # Get goal
        goal = goal_handle.request
        
        # Feedback
        feedback_msg = VLAManipulation.Feedback()
        
        # Execute task
        try:
            if goal.task_type == 'pick':
                success = self.execute_pick(goal, goal_handle, feedback_msg)
            elif goal.task_type == 'place':
                success = self.execute_place(goal, goal_handle, feedback_msg)
            else:
                success = False
                self.get_logger().error(f'Unknown task type: {goal.task_type}')
        
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
            success = False
        
        # Result
        goal_handle.succeed()
        
        result = VLAManipulation.Result()
        result.success = success
        result.message = 'Success' if success else 'Failed'
        
        return result
    
    def execute_pick(self, goal, goal_handle, feedback_msg):
        """
        Execute pick task
        """
        phases = ['approaching', 'grasping', 'lifting', 'retracting']
        
        for i, phase in enumerate(phases):
            # Update feedback
            feedback_msg.current_phase = phase
            feedback_msg.progress = (i + 1) / len(phases)
            goal_handle.publish_feedback(feedback_msg)
            
            self.get_logger().info(f'Phase: {phase}')
            
            # Execute phase with VLA
            success = self.execute_phase(phase)
            
            if not success:
                return False
        
        return True
    
    def execute_phase(self, phase):
        """
        Execute single phase using VLA
        """
        # Get observations
        obs = self.get_observation()
        
        # VLA inference
        with torch.no_grad():
            actions = self.vla_model.predict(obs)
        
        # Execute actions
        for action in actions:
            self.execute_action(action)
            
            # Check for failure
            if self.check_failure():
                return False
        
        return True
```

**시간: 주 8-10시간**

---

### Week 7-8: Sim-to-Real Transfer 준비

#### Domain Randomization 강화
```python
# advanced_domain_randomization.py

class AdvancedDomainRandomizer:
    """
    고급 Domain Randomization
    
    목적: Sim-to-Real gap 최소화
    
    전략:
    1. Physics randomization
    2. Visual randomization
    3. Sensor noise
    4. Actuation noise
    5. Dynamics randomization
    """
    
    def __init__(self, world, config):
        self.world = world
        self.config = config
        
        # Randomization ranges
        self.ranges = {
            'gravity': [-10.5, -9.5],
            'friction': [0.3, 1.5],
            'mass': [0.8, 1.2],  # multiplier
            'lighting_intensity': [2000, 8000],
            'color_temperature': [3000, 7000],
            'camera_noise': [0, 0.05],
            'actuator_noise': [0, 0.02],
        }
    
    def randomize_all(self):
        """
        Randomize all aspects
        """
        self.randomize_physics()
        self.randomize_visuals()
        self.randomize_sensors()
        self.randomize_dynamics()
    
    def randomize_physics(self):
        """
        물리 파라미터 랜덤화
        """
        # Gravity
        gravity_z = np.random.uniform(*self.ranges['gravity'])
        self.world.set_gravity([0, 0, gravity_z])
        
        # Global friction multiplier
        friction_mult = np.random.uniform(*self.ranges['friction'])
        
        # Apply to all objects
        for obj in self.world.scene.get_all_objects():
            if hasattr(obj, 'get_applied_physics_material'):
                material = obj.get_applied_physics_material()
                
                if material:
                    base_static = material.get_static_friction()
                    base_dynamic = material.get_dynamic_friction()
                    
                    material.set_static_friction(base_static * friction_mult)
                    material.set_dynamic_friction(base_dynamic * friction_mult)
        
        # Mass variation
        mass_mult = np.random.uniform(*self.ranges['mass'])
        
        for obj in self.world.scene.get_all_objects():
            if hasattr(obj, 'get_mass'):
                base_mass = obj.get_mass()
                obj.set_mass(base_mass * mass_mult)
        
        # Restitution (bounciness)
        for obj in self.world.scene.get_all_objects():
            if hasattr(obj, 'get_applied_physics_material'):
                material = obj.get_applied_physics_material()
                
                if material:
                    restitution = np.random.uniform(0, 0.5)
                    material.set_restitution(restitution)
    
    def randomize_visuals(self):
        """
        시각적 요소 랜덤화
        """
        # Lighting
        from pxr import UsdLux
        stage = omni.usd.get_context().get_stage()
        
        for i in range(4):
            light_path = f"/World/Light_{i}"
            light_prim = stage.GetPrimAtPath(light_path)
            
            if light_prim:
                light = UsdLux.RectLight(light_prim)
                
                # Intensity
                intensity = np.random.uniform(*self.ranges['lighting_intensity'])
                light.GetIntensityAttr().Set(intensity)
                
                # Color temperature
                temp = np.random.uniform(*self.ranges['color_temperature'])
                light.GetColorTemperatureAttr().Set(temp)
                
                # Position variation
                current_pos = light.GetPrim().GetAttribute('xformOp:translate').Get()
                pos_noise = np.random.uniform(-0.5, 0.5, 3)
                new_pos = tuple(np.array(current_pos) + pos_noise)
                light.GetPrim().GetAttribute('xformOp:translate').Set(new_pos)
        
        # Textures and colors
        for obj in self.world.scene.get_all_objects():
            if hasattr(obj, 'set_color'):
                # Random color in HSV space
                hue = np.random.uniform(0, 1)
                saturation = np.random.uniform(0.3, 1.0)
                value = np.random.uniform(0.4, 1.0)
                
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                obj.set_color(rgb)
        
        # Background
        # Add random patterns or textures to floor/walls
        self.randomize_background()
    
    def randomize_sensors(self):
        """
        센서 노이즈 랜덤화
        """
        # Camera noise will be added during observation
        self.camera_noise_std = np.random.uniform(*self.ranges['camera_noise'])
        
        # Camera parameters
        if hasattr(self, 'camera'):
            # Exposure
            exposure = np.random.uniform(0.8, 1.2)
            
            # Gain
            gain = np.random.uniform(0.9, 1.1)
            
            # Apply (implementation dependent on camera API)
    
    def randomize_dynamics(self):
        """
        동역학 랜덤화
        """
        # Joint damping
        for joint_idx in range(7):
            damping = np.random.uniform(0.5, 2.0)
            # Set joint damping (implementation dependent)
        
        # Motor characteristics
        # Backlash, delay, etc.
    
    def add_sensor_noise(self, observation):
        """
        관측에 노이즈 추가 (실시간)
        """
        obs = observation.copy()
        
        # Image noise
        if 'rgb' in obs:
            # Gaussian noise
            noise = np.random.normal(0, self.camera_noise_std * 255, obs['rgb'].shape)
            obs['rgb'] = np.clip(obs['rgb'] + noise, 0, 255).astype(np.uint8)
            
            # Salt and pepper noise
            if np.random.random() < 0.1:
                mask = np.random.random(obs['rgb'].shape[:2]) < 0.01
                obs['rgb'][mask] = np.random.choice([0, 255])
        
        # Depth noise
        if 'depth' in obs:
            depth_noise = np.random.normal(0, 0.01, obs['depth'].shape)
            obs['depth'] = np.clip(obs['depth'] + depth_noise, 0, 5.0)
        
        # Proprioception noise
        if 'joint_pos' in obs:
            joint_noise = np.random.normal(0, 0.01, obs['joint_pos'].shape)
            obs['joint_pos'] += joint_noise
        
        if 'joint_vel' in obs:
            vel_noise = np.random.normal(0, 0.05, obs['joint_vel'].shape)
            obs['joint_vel'] += vel_noise
        
        return obs
    
    def add_actuation_noise(self, action):
        """
        액츄에이션 노이즈 추가
        """
        noise_std = np.random.uniform(*self.ranges['actuator_noise'])
        noise = np.random.normal(0, noise_std, action.shape)
        
        noisy_action = action + noise
        
        # Add delay (random)
        if np.random.random() < 0.1:
            # 10% chance of 1-step delay
            # Store action for next timestep
            pass
        
        return noisy_action
    
    def randomize_background(self):
        """
        배경 랜덤화
        """
        # Add random objects to background
        num_objects = np.random.randint(0, 5)
        
        for i in range(num_objects):
            # Random position (background)
            position = [
                np.random.uniform(-5, 5),
                np.random.uniform(5, 10),  # Far from robot
                np.random.uniform(0, 2)
            ]
            
            # Random shape
            shape_type = np.random.choice(['cube', 'sphere', 'cylinder'])
            
            # Add to scene
            # Implementation depends on Isaac Sim API

# Curriculum learning for domain randomization
class CurriculumDomainRandomization:
    """
    점진적 난이도 증가
    
    초기: 약한 randomization (학습 용이)
    후반: 강한 randomization (robust)
    """
    
    def __init__(self, randomizer, curriculum_steps=10):
        self.randomizer = randomizer
        self.curriculum_steps = curriculum_steps
        self.current_step = 0
    
    def get_randomization_strength(self):
        """
        현재 curriculum step에 따른 randomization 강도
        """
        progress = self.current_step / self.curriculum_steps
        
        # Exponential increase
        strength = progress ** 2
        
        return min(strength, 1.0)
    
    def randomize(self):
        """
        Curriculum에 따른 randomization
        """
        strength = self.get_randomization_strength()
        
        # Scale randomization ranges
        for key in self.randomizer.ranges:
            base_range = self.randomizer.ranges[key]
            
            # Center value
            center = (base_range[0] + base_range[1]) / 2
            
            # Scaled range
            half_width = (base_range[1] - base_range[0]) / 2 * strength
            
            scaled_range = [center - half_width, center + half_width]
            
            # Temporarily update
            original_range = self.randomizer.ranges[key]
            self.randomizer.ranges[key] = scaled_range
        
        # Apply randomization
        self.randomizer.randomize_all()
        
        # Restore original ranges
        # (or keep for next time)
    
    def step_curriculum(self):
        """
        Advance curriculum
        """
        self.current_step += 1
        
        if self.current_step > self.curriculum_steps:
            self.current_step = self.curriculum_steps
        
        print(f"Curriculum step: {self.current_step}/{self.curriculum_steps} "
              f"(strength: {self.get_randomization_strength():.2f})")
```

---

#### Reality Gap 분석
```python
# reality_gap_analysis.py

class RealityGapAnalyzer:
    """
    Sim-Real Gap 분석
    
    비교 항목:
    1. Physics (dynamics, friction, contact)
    2. Perception (lighting, camera, colors)
    3. Actuation (delays, backlash, errors)
    4. Environment (objects, layout)
    """
    
    def __init__(self):
        self.sim_data = []
        self.real_data = []
    
    def collect_sim_data(self, num_episodes=50):
        """
        Simulation 데이터 수집
        """
        print("Collecting simulation data...")
        
        for ep in range(num_episodes):
            episode_data = self.run_episode_in_sim()
            self.sim_data.append(episode_data)
        
        print(f"Collected {len(self.sim_data)} simulation episodes")
    
    def collect_real_data(self, num_episodes=50):
        """
        Real robot 데이터 수집
        """
        print("Collecting real robot data...")
        
        for ep in range(num_episodes):
            episode_data = self.run_episode_on_real()
            self.real_data.append(episode_data)
        
        print(f"Collected {len(self.real_data)} real episodes")
    
    def analyze_gap(self):
        """
        Gap 분석 및 시각화
        """
        print("\n" + "="*60)
        print("REALITY GAP ANALYSIS")
        print("="*60)
        
        # 1. Success rate gap
        sim_success = np.mean([ep['success'] for ep in self.sim_data])
        real_success = np.mean([ep['success'] for ep in self.real_data])
        
        print(f"\nSuccess Rate:")
        print(f"  Simulation: {sim_success*100:.1f}%")
        print(f"  Real:       {real_success*100:.1f}%")
        print(f"  Gap:        {(sim_success - real_success)*100:+.1f}%")
        
        # 2. Trajectory comparison
        self.analyze_trajectory_gap()
        
        # 3. Timing comparison
        self.analyze_timing_gap()
        
        # 4. Perception comparison
        self.analyze_perception_gap()
        
        # 5. Recommendations
        self.generate_recommendations()
    
    def analyze_trajectory_gap(self):
        """
        Trajectory 차이 분석
        """
        print(f"\nTrajectory Gap:")
        
        # Extract trajectories
        sim_trajs = [ep['trajectory'] for ep in self.sim_data if ep['success']]
        real_trajs = [ep['trajectory'] for ep in self.real_data if ep['success']]
        
        # Compare smoothness
        sim_smoothness = [self.compute_smoothness(t) for t in sim_trajs]
        real_smoothness = [self.compute_smoothness(t) for t in real_trajs]
        
        print(f"  Sim smoothness:  {np.mean(sim_smoothness):.4f}")
        print(f"  Real smoothness: {np.mean(real_smoothness):.4f}")
        
        # Compare path length
        sim_lengths = [self.compute_path_length(t) for t in sim_trajs]
        real_lengths = [self.compute_path_length(t) for t in real_trajs]
        
        print(f"  Sim path length:  {np.mean(sim_lengths):.2f}m")
        print(f"  Real path length: {np.mean(real_lengths):.2f}m")
    
    def analyze_timing_gap(self):
        """
        Timing 차이 분석
        """
        print(f"\nTiming Gap:")
        
        sim_times = [ep['completion_time'] for ep in self.sim_data if ep['success']]
        real_times = [ep['completion_time'] for ep in self.real_data if ep['success']]
        
        print(f"  Sim time:  {np.mean(sim_times):.2f}s (±{np.std(sim_times):.2f})")
        print(f"  Real time: {np.mean(real_times):.2f}s (±{np.std(real_times):.2f})")
        
        # Time distribution comparison
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(sim_times, bins=20, alpha=0.5, label='Simulation')
        plt.hist(real_times, bins=20, alpha=0.5, label='Real')
        plt.xlabel('Completion Time (s)')
        plt.ylabel('Frequency')
        plt.title('Timing Distribution: Sim vs Real')
        plt.legend()
        plt.savefig('timing_gap.png')
        plt.show()
    
    def analyze_perception_gap(self):
        """
        Perception 차이 분석
        """
        print(f"\nPerception Gap:")
        
        # Compare image statistics
        sim_images = [ep['images'] for ep in self.sim_data]
        real_images = [ep['images'] for ep in self.real_data]
        
        # Brightness
        sim_brightness = [np.mean(img) for imgs in sim_images for img in imgs]
        real_brightness = [np.mean(img) for imgs in real_images for img in imgs]
        
        print(f"  Sim brightness:  {np.mean(sim_brightness):.1f}")
        print(f"  Real brightness: {np.mean(real_brightness):.1f}")
        
        # Contrast
        sim_contrast = [np.std(img) for imgs in sim_images for img in imgs]
        real_contrast = [np.std(img) for imgs in real_images for img in imgs]
        
        print(f"  Sim contrast:  {np.mean(sim_contrast):.1f}")
        print(f"  Real contrast: {np.mean(real_contrast):.1f}")
    
    def generate_recommendations(self):
        """
        개선 권장사항
        """
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        sim_success = np.mean([ep['success'] for ep in self.sim_data])
        real_success = np.mean([ep['success'] for ep in self.real_data])
        
        gap = sim_success - real_success
        
        if gap > 0.2:  # 20% gap
            print("\n⚠️ Large sim-real gap detected!")
            print("\nPriority actions:")
            print("1. Increase domain randomization strength")
            print("2. Collect more diverse simulation data")
            print("3. Fine-tune on real robot data")
            print("4. Check calibration (camera, robot)")
        
        elif gap > 0.1:  # 10% gap
            print("\n⚠️ Moderate sim-real gap")
            print("\nSuggested actions:")
            print("1. Add more visual randomization")
            print("2. Tune physics parameters to match real")
            print("3. Add sensor noise modeling")
        
        else:
            print("\n✅ Small sim-real gap")
            print("\nMaintain current approach:")
            print("1. Continue domain randomization")
            print("2. Monitor performance over time")
            print("3. Occasional real data collection")
    
    def compute_smoothness(self, trajectory):
        """
        Trajectory smoothness (jerk)
        """
        positions = np.array([state['joint_pos'] for state in trajectory])
        velocities = np.diff(positions, axis=0)
        jerks = np.diff(velocities, axis=0)
        
        return -np.mean(np.abs(jerks))
    
    def compute_path_length(self, trajectory):
        """
        Total path length
        """
        positions = np.array([state['ee_pos'] for state in trajectory])
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        
        return np.sum(distances)
```

---

#### 실제 로봇 배포 체크리스트
```markdown
## Real Robot Deployment Checklist

### Pre-deployment (Simulation)

#### Model Validation
- [ ] Model achieves >70% success in simulation
- [ ] Model robust to domain randomization
- [ ] Action smoothness acceptable
- [ ] No safety violations in 100+ episodes
- [ ] Tested with various box sizes/positions

#### Safety Verification
- [ ] Safety layer tested and verified
- [ ] Emergency stop functional
- [ ] Joint limits enforced
- [ ] Velocity limits enforced
- [ ] Workspace boundaries defined
- [ ] Collision detection working

#### Code Quality
- [ ] Code reviewed and tested
- [ ] No hardcoded paths or parameters
- [ ] Proper error handling
- [ ] Logging implemented
- [ ] ROS2 integration tested

---

### Hardware Setup

#### Robot Calibration
- [ ] Robot zeroed and homed
- [ ] Joint encoders calibrated
- [ ] Tool center point (TCP) calibrated
- [ ] Gripper calibrated
- [ ] Force/torque sensor calibrated (if applicable)

#### Camera Setup
- [ ] Camera mounted securely
- [ ] Camera calibrated (intrinsics)
- [ ] Camera-robot calibration (extrinsics)
- [ ] Lighting consistent with training
- [ ] Frame rate stable (20Hz)
- [ ] Image quality verified

#### Workspace Preparation
- [ ] Workspace clear of obstacles
- [ ] Safety barriers in place
- [ ] Emergency stop accessible
- [ ] Lighting controlled
- [ ] Floor markers for repeatability

---

### Initial Testing

#### Sanity Checks
- [ ] Camera feed visible in ROS2
- [ ] Joint states publishing correctly
- [ ] Commands being received
- [ ] TF tree correct
- [ ] No network delays

#### Manual Control
- [ ] Manually move robot through workspace
- [ ] Test gripper open/close
- [ ] Verify safety stops work
- [ ] Check for mechanical issues
- [ ] Confirm smooth motion

#### Dry Run (No Objects)
- [ ] Run VLA with empty workspace
- [ ] Verify reasonable motions
- [ ] No erratic behavior
- [ ] Actions within expected range
- [ ] Monitor for 10+ minutes

---

### Gradual Deployment

#### Phase 1: Single Object, Easy Position
- [ ] Place object in known good position
- [ ] Run 10 episodes
- [ ] Monitor closely
- [ ] Success rate >50%
- [ ] No safety incidents

#### Phase 3: Single Object, Varied Positions
- [ ] Test 5-7 different positions
- [ ] Run 5 episodes per position
- [ ] Success rate >60%
- [ ] Consistent behavior

#### Phase 3: Multiple Objects
- [ ] Test with 2-3 objects
- [ ] Various sizes
- [ ] Run 20 episodes
- [ ] Success rate >70%

#### Phase 3: Full Deployment
- [ ] Realistic scenarios
- [ ] Extended operation (1+ hour)
- [ ] Success rate >70%
- [ ] Failure recovery working

---

### Monitoring & Maintenance

#### Continuous Monitoring
- [ ] Log all episodes
- [ ] Track success rate over time
- [ ] Monitor failure modes
- [ ] Check for degradation
- [ ] Review safety incidents

#### Regular Maintenance
- [ ] Daily: Visual inspection
- [ ] Weekly: Calibration check
- [ ] Monthly: Full recalibration
- [ ] Quarterly: Performance review

#### Data Collection
- [ ] Collect failure cases
- [ ] Periodically collect success cases
- [ ] Label and store for retraining
- [ ] Analyze trends

---

### Troubleshooting Guide

#### Low Success Rate (<50%)
1. Check camera calibration
2. Verify lighting conditions
3. Review domain randomization
4. Collect real data for fine-tuning

#### Erratic Behavior
1. Check action normalization
2. Verify safety layer active
3. Review recent changes
4. Test in simulation first

#### Gripper Failures
1. Calibrate gripper force
2. Adjust grasp positions
3. Check object properties
4. Review grasp detection logic

#### Collisions
1. Reduce action magnitude
2. Strengthen safety constraints
3. Add more collision training data
4. Review workspace setup
```

**시간: 주 8-10시간**

---

### 성능 최적화

#### 추론 속도 최적화
```python
# optimization.py

class ModelOptimizer:
    """
    VLA 모델 최적화
    
    목표:
    - 추론 속도 향상 (< 100ms)
    - 메모리 사용량 감소
    - Throughput 증가
    """
    
    def __init__(self, model):
        self.model = model
    
    def optimize_all(self):
        """
        전체 최적화 파이프라인
        """
        # 1. TorchScript compilation
        print("Step 1: TorchScript compilation...")
        scripted_model = self.to_torchscript()
        
        # 2. Quantization
        print("Step 2: Quantization...")
        quantized_model = self.quantize(scripted_model)
        
        # 3. ONNX export (optional)
        print("Step 3: ONNX export...")
        self.export_onnx()
        
        # 4. TensorRT (NVIDIA)
        print("Step 4: TensorRT optimization...")
        trt_model = self.to_tensorrt()
        
        return trt_model
    
    def to_torchscript(self):
        """
        TorchScript로 변환
        
        장점:
        - Python overhead 제거
        - 최적화된 실행
        - 배포 용이
        """
        self.model.eval()
        
        # Example input
        dummy_obs = {
            'rgb': torch.randn(1, 3, 224, 224).cuda(),
            'proprio': torch.randn(1, 15).cuda()
        }
        
        # Trace model
        with torch.no_grad():
            scripted = torch.jit.trace(self.model, dummy_obs)
        
        # Save
        scripted.save('model_scripted.pt')
        
        print("✅ TorchScript model saved")
        
        return scripted
    
    def quantize(self, model):
        """
        모델 양자화 (FP32 → INT8)
        
        장점:
        - 모델 크기 1/4
        - 추론 속도 2-4배
        - 약간의 정확도 손실 (<2%)
        """
        from torch.quantization import quantize_dynamic
        
        quantized = quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        # Save
        torch.save(quantized.state_dict(), 'model_quantized.pt')
        
        print("✅ Quantized model saved")
        
        return quantized
    
    def export_onnx(self):
        """
        ONNX로 export
        
        장점:
        - 다양한 runtime 지원
        - 추가 최적화 가능
        - 플랫폼 독립적
        """
        self.model.eval()
        
        dummy_obs = {
            'rgb': torch.randn(1, 3, 224, 224).cuda(),
            'proprio': torch.randn(1, 15).cuda()
        }
        
        torch.onnx.export(
            self.model,
            dummy_obs,
            'model.onnx',
            input_names=['rgb', 'proprio'],
            output_names=['actions'],
            dynamic_axes={
                'rgb': {0: 'batch'},
                'proprio': {0: 'batch'},
                'actions': {0: 'batch'}
            },
            opset_version=14
        )
        
        print("✅ ONNX model exported")
    
    def to_tensorrt(self):
        """
        TensorRT로 변환 (NVIDIA GPU)
        
        장점:
        - 최대 추론 속도
        - GPU 최적화
        - Mixed precision
        """
        import tensorrt as trt
        from torch2trt import torch2trt
        
        self.model.eval()
        
        dummy_obs = {
            'rgb': torch.randn(1, 3, 224, 224).cuda(),
            'proprio': torch.randn(1, 15).cuda()
        }
        
        # Convert
        model_trt = torch2trt(
            self.model,
            [dummy_obs],
            fp16_mode=True,  # FP16 precision
            max_workspace_size=1 << 30  # 1GB
        )
        
        # Save
        torch.save(model_trt.state_dict(), 'model_trt.pth')
        
        print("✅ TensorRT model saved")
        
        return model_trt
    
    def benchmark(self, model, num_iterations=100):
        """
        추론 속도 벤치마크
        """
        import time
        
        model.eval()
        
        dummy_obs = {
            'rgb': torch.randn(1, 3, 224, 224).cuda(),
            'proprio': torch.randn(1, 15).cuda()
        }
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_obs)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_obs)
        
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_iterations * 1000  # ms
        
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Throughput: {1000/avg_time:.1f} FPS")
        
        return avg_time

# Compare optimizations
def compare_optimizations():
    """
    최적화 효과 비교
    """
    # Original model
    model = ACTPolicy(config).cuda()
    model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
    
    optimizer = ModelOptimizer(model)
    
    print("\n" + "="*60)
    print("OPTIMIZATION BENCHMARK")
    print("="*60)
    
    # Original
    print("\n1. Original Model (FP32)")
    time_original = optimizer.benchmark(model)
    
    # TorchScript
    print("\n2. TorchScript")
    scripted = optimizer.to_torchscript()
    time_scripted = optimizer.benchmark(scripted)
    
    # Quantized
    print("\n3. Quantized (INT8)")
    quantized = optimizer.quantize(model)
    time_quantized = optimizer.benchmark(quantized)
    
    # TensorRT
    print("\n4. TensorRT (FP16)")
    trt = optimizer.to_tensorrt()
    time_trt = optimizer.benchmark(trt)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<20s} {'Time (ms)':<12s} {'Speedup':<10s}")
    print("-"*60)
    print(f"{'Original':<20s} {time_original:>10.2f} ms {1.0:>8.1f}x")
    print(f"{'TorchScript':<20s} {time_scripted:>10.2f} ms {time_original/time_scripted:>8.1f}x")
    print(f"{'Quantized':<20s} {time_quantized:>10.2f} ms {time_original/time_quantized:>8.1f}x")
    print(f"{'TensorRT':<20s} {time_trt:>10.2f} ms {time_original/time_trt:>8.1f}x")
    print("="*60)

"""
예상 결과:

Method              Time (ms)    Speedup   
------------------------------------------------------------
Original                80.00 ms      1.0x
TorchScript             50.00 ms      1.6x
Quantized               40.00 ms      2.0x
TensorRT                25.00 ms      3.2x
============================================================

→ TensorRT 사용 시 실시간 제어 가능 (10Hz)
"""
```

---

#### Action Smoothing
```python
# action_smoothing.py

class ActionSmoother:
    """
    Action smoothing for jerk reduction
    
    방법:
    1. Moving average
    2. Exponential smoothing
    3. Savitzky-Golay filter
    """
    
    def __init__(self, method='exponential', window=5, alpha=0.3):
        self.method = method
        self.window = window
        self.alpha = alpha
        
        self.action_history = deque(maxlen=window)
    
    def smooth(self, action):
        """
        Smooth action
        """
        self.action_history.append(action)
        
        if self.method == 'moving_average':
            return self.moving_average()
        
        elif self.method == 'exponential':
            return self.exponential_smoothing(action)
        
        elif self.method == 'savgol':
            return self.savitzky_golay()
        
        else:
            return action
    
    def moving_average(self):
        """
        Moving average smoothing
        """
        if len(self.action_history) == 0:
            return np.zeros(7)
        
        return np.mean(list(self.action_history), axis=0)
    
    def exponential_smoothing(self, action):
        """
        Exponential smoothing
        
        smoothed = alpha * current + (1-alpha) * previous
        """
        if len(self.action_history) < 2:
            return action
        
        previous_smoothed = self.action_history[-2]
        smoothed = self.alpha * action + (1 - self.alpha) * previous_smoothed
        
        return smoothed
    
    def savitzky_golay(self):
        """
        Savitzky-Golay filter
        """
        from scipy.signal import savgol_filter
        
        if len(self.action_history) < self.window:
            return self.action_history[-1]
        
        # Convert to array
        history_array = np.array(list(self.action_history))
        
        # Apply filter
        smoothed = savgol_filter(
            history_array,
            window_length=self.window,
            polyorder=2,
            axis=0
        )
        
        return smoothed[-1]
```

**시간: 주 4-6시간**

---

## Phase 3 완료 체크
```
✅ Isaac Sim 환경 마스터
✅ Action/Observation Space 설계
✅ 물류 VLA 개발 완료
✅ 데이터 수집 및 품질 관리
✅ VLA 학습 및 평가
✅ Hyperparameter tuning
✅ 실패 복구 시스템
✅ Safety layer 구현
✅ ROS2 완전 통합
✅ Domain Randomization
✅ Sim-to-Real 준비
✅ 성능 최적화

성과:
- Success Rate: 70%+
- 실시간 제어 가능 (< 100ms)
- ROS2 Lifecycle 패턴 적용
- 안전 시스템 완비
- 실제 로봇 배포 준비 완료
```
