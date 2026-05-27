# Xbox Control - FR5 机器人控制系统


这是一个基于FR5（法奥机器人）的综合控制系统，集成了机器人控制、视觉处理、夹爪驱动和数据记录功能。该项目支持实时机器人操作、策略推理、轨迹回放和数据采集等多种应用场景。


### 核心模块

#### 根目录文件

| 文件 | 功能描述 |
|------|--------|
| `fr5_env.py` | 机器人环境类，管理机器人、摄像头和夹爪的集成 |
| `fr5_controller.py` | FR5机器人伺服联合控制器（支持ServoJ指令） |
| `camera_controller.py` | 摄像头控制与图像采集模块 |
| `gripper_controller.py` | Robotiq夹爪控制模块 |
| `action_node.py` | 动作节点处理模块 |
| `fr5_inference_util.py` | 机器人推理工具函数 |
| `fr5_local_policy_controller_node.py` | 本地策略控制节点（ROS集成） |
| `serialization_utils.py` | 序列化工具函数 |
| `test_*.py` | 单元测试文件（action、camera、env、gripper） |

#### `control/` 目录 - 控制与数据处理

| 文件 | 功能描述 |
|------|--------|
| `control.py` | 主控制脚本，包含ROS JointState发布器 |
| `data_rlds_recording.py` | RLDS格式数据记录脚本 |
| `replay_joint_pos.py` | 关节位置回放脚本 |
| `action_delta_check.py` | 动作增量检查与验证 |
| `action_delta_rebuild.py` | 动作增量重建工具 |
| `noise_check.py` | 噪声检测与分析 |
| `data_check.py` | 数据完整性检查 |
| `orbbec_calibrate.py` | Orbbec摄像头标定脚本 |
| `camera_intrinsics.yaml` | 摄像头内参配置文件 |

#### `fairino/` 目录 - 法奥机器人SDK

包含法奥机器人的Python SDK库，版本支持：
- 法奥机器人 SDK V2.0.8+
- Python版本：3.8, 3.9, 3.10, 3.11, 3.12（支持Windows和Linux）
- 主要接口：`Robot.py` 类提供机器人通信和控制

#### `mujoco_fr5/` 目录 - 仿真资源

包含Mujoco仿真模型和资源：
- **assets/fr5/** - FR5机器人模型
  - `mjmodel.xml` - Mujoco模型文件
  - `meshes/` - 机器人和夹爪的三维网格
  - `3dgs_mesh/` - 3D高斯溅射网格
  - `textures/` - 纹理资源


### 1. 实时机器人控制
- **ServoJ控制**：高频关节伺服控制（支持自定义频率）
- **关节状态反馈**：实时读取机器人关节角度和速度
- **多进程架构**：使用multiprocessing确保实时性

### 2. 视觉系统
- 摄像头采集与管理
- 深度图像处理
- 摄像头标定与参数管理
- 时间戳同步

### 3. 夹爪控制
- Robotiq 85夹爪驱动
- 开度控制与反馈
- 与ROS集成的关节状态发布

### 4. 数据采集与处理
- RLDS格式数据记录
- 轨迹回放与复现
- 动作增量检查与修复
- 数据验证与质量检查

### 5. ROS集成
- 关节状态发布（/joint_states话题）
- 与ROS导航和规划的集成
- 标准ROS消息格式


### Python库
```
numpy
pygame (Xbox控制支持)
rclpy (ROS 2集成)
pyrobotiqgripper (夹爪驱动)
sensor_msgs (ROS消息)
```

### 硬件驱动
- Fairino FR5机器人SDK
- Orbbec摄像头驱动
- Robotiq夹爪驱动


### 1. 环境配置

```bash
# 安装Python依赖
pip install numpy pygame rclpy pyrobotiqgripper

# 安装fairino SDK（已包含在fairino/目录）
cd fairino
python setup.py install
```

### 2. 摄像头标定

```bash
python control/orbbec_calibrate.py
```

### 3. 运行机器人控制

```bash
# 启动FR5控制
python control/control.py

# 或运行ROS节点
python fr5_local_policy_controller_node.py
```

### 4. 数据采集

```bash
# 记录RLDS格式数据
python control/data_rlds_recording.py
```

### 5. 轨迹回放

```bash
# 回放之前记录的轨迹
python control/replay_joint_pos.py
```

### camera_intrinsics.yaml
存储摄像头的内参矩阵和畸变系数：
```yaml
camera_matrix: [fx, fy, cx, cy]
distortion_coefficients: [k1, k2, p1, p2]
```

- `control/data_check.py` - 数据完整性验证
- `control/noise_check.py` - 噪声检测
- `control/action_delta_check.py` - 动作增量验证
- `control/action_delta_rebuild.py` - 修复动作增量


```bash
# 运行单元测试
python test_action.py
python test_camera.py
python test_env.py
python test_gripper.py
```

