import torch
import torch.nn as nn
import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import os

DEVICE = torch.device("cpu") # 테스트는 CPU로 충분합니다.

# =====================================================================
# 1. 상수 및 로봇 리밋 (trainer.py에서 가져와 하드코딩)
# =====================================================================
URDF_PATH = r"/home/whatdojdo/Desktop/coop_controlMLP/m0609.urdf"
MAX_DELTA_Q = 0.05 

# 터미널 로그에 찍혔던 q_min, q_max 값을 그대로 하드코딩 (파싱 로직 불필요)
q_min = torch.tensor([-6.2832, -6.2832, -2.6180, -6.2832, -6.2832, -6.2832], device=DEVICE)
q_max = torch.tensor([ 6.2832,  6.2832,  2.6180,  6.2832,  6.2832,  6.2832], device=DEVICE)

# 🌟 테스트할 가중치 파일 경로
WEIGHTS_PATH = "/home/whatdojdo/Desktop/coop_controlMLP/trainer_v3/complete_train/trained_mlp_weights_10k_epoch.pth"

# =====================================================================
# 2. 모델 클래스 (trainer.py에 의존하지 않도록 직접 선언)
# =====================================================================
class controlMLP(nn.Module):
    def __init__(self, q_min, q_max, max_delta):
        super(controlMLP, self).__init__()
        self.q_min = q_min
        self.q_max = q_max
        self.max_delta = max_delta

        self.net = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Tanh()
        )

    def forward(self, q_curr, P_tgt, A_tgt, A_up_tgt):
        x = torch.cat([q_curr, P_tgt, A_tgt, A_up_tgt], dim=-1)
        a = self.net(x) 
        
        delta_upper = torch.clamp(self.q_max - q_curr, max=self.max_delta)
        delta_lower = torch.clamp(self.q_min - q_curr, min=-self.max_delta)
        
        a_normalized = (a + 1.0) / 2.0
        delta_q = delta_lower + a_normalized * (delta_upper - delta_lower)
        
        return delta_q

# =====================================================================
# 3. 가중치 로드 및 PyBullet 환경 세팅
# =====================================================================
model = controlMLP(q_min, q_max, MAX_DELTA_Q).to(DEVICE)

if os.path.exists(WEIGHTS_PATH):
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('_orig_mod.', '')
        clean_state_dict[clean_key] = v
    model.load_state_dict(clean_state_dict)
    print(f"✅ 학습된 가중치 로드 완료!")
else:
    print(f"⚠️ 삐빅! 가중치 파일이 없습니다. 랜덤으로 움직입니다.")

model.eval() 

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0], useFixedBase=True)

num_joints = p.getNumJoints(robot_id)
active_joints = []
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    if info[2] != p.JOINT_FIXED:
        active_joints.append(i)

# =====================================================================
# 4. GUI 슬라이더 생성
# =====================================================================
target_x = p.addUserDebugParameter("Target X", -1.0, 1.0, 0.4)
target_y = p.addUserDebugParameter("Target Y", -1.0, 1.0, 0.0)
target_z = p.addUserDebugParameter("Target Z", 0.0, 1.5, 0.5)

target_roll = p.addUserDebugParameter("Target Roll", -math.pi, math.pi, math.pi)
target_pitch = p.addUserDebugParameter("Target Pitch", -math.pi, math.pi, 0.0)
target_yaw = p.addUserDebugParameter("Target Yaw", -math.pi, math.pi, 0.0)

print("✅ PyBullet 시각화 시작! 슬라이더를 움직여보세요.")

q_curr = torch.zeros(1, 6, device=DEVICE)

while True:
    tx = p.readUserDebugParameter(target_x)
    ty = p.readUserDebugParameter(target_y)
    tz = p.readUserDebugParameter(target_z)
    
    roll = p.readUserDebugParameter(target_roll)
    pitch = p.readUserDebugParameter(target_pitch)
    yaw = p.readUserDebugParameter(target_yaw)
    
    rot_matrix = p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw]))
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    P_tgt = torch.tensor([[tx, ty, tz]], dtype=torch.float32, device=DEVICE)
    A_up_tgt = torch.tensor([rot_matrix[:, 0]], dtype=torch.float32, device=DEVICE) 
    A_tgt = torch.tensor([rot_matrix[:, 2]], dtype=torch.float32, device=DEVICE)    

    with torch.no_grad():
        delta_q = model(q_curr, P_tgt, A_tgt, A_up_tgt)
        q_curr = q_curr + delta_q
        q_curr = torch.clamp(q_curr, q_min, q_max)

    q_list = q_curr[0].cpu().numpy().tolist()
    for i, joint_idx in enumerate(active_joints):
        p.setJointMotorControl2(bodyIndex=robot_id, 
                                jointIndex=joint_idx, 
                                controlMode=p.POSITION_CONTROL, 
                                targetPosition=q_list[i])
        
    p.removeAllUserDebugItems() 
    p.addUserDebugLine([tx, ty, tz], [tx + A_tgt[0][0].item()*0.2, ty + A_tgt[0][1].item()*0.2, tz + A_tgt[0][2].item()*0.2], [1, 0, 0], 3) 
    p.addUserDebugLine([tx, ty, tz], [tx + A_up_tgt[0][0].item()*0.2, ty + A_up_tgt[0][1].item()*0.2, tz + A_up_tgt[0][2].item()*0.2], [0, 1, 0], 3) 

    p.stepSimulation()
    time.sleep(0.02)