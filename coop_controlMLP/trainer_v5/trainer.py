#---------------------------------------------------------------------------------------
# changelog

# 26.5.4
# 첫 학습 진행
# v2. 특이점 텀 약화, 새로운 스텝와이즈 로스텀

# 26.5.5
# v3. 특이점 텀을 더 약화. (0.5 -> 0.1) => DL-based control에서는 특이점이 상관이 없다. 과감하게
# 불러와서 학습할 수 있도록 수정

# v4. 
# model size 증대 (2층 -> 3층)
# adaptive MAXSTEP적용 => 진짜 딱 필요한 만큼으로 max_step을 정의
# LR 스케줄러 도입

# v5.
# get_kinematics() 분석 완료
# 실제 그리퍼 오프셋 0.24m 확인완료
# norm_q를 입력으로 변환
# step-wise cos loss 추가 => 매 스텝마다 cos에러도 보정하도록 유도

#---------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_kinematics as pk
import math

import time

#=====================================
# robot parameters

URDF_PATH = r"/home/ssu/Desktop/coop_controlMLP/m0609.urdf" # 실제 경로로 수정
EE_LINK_NAME = "link_6"  # 엔드 이펙터 링크 이름 (예시)
Z_OFFSET = 0.24   #in meters 그리퍼가 link_6로부터 가진 오프셋

MAX_DELTA_Q = 0.05 # 1스텝당 관절 최대 변화량 (Radian)

DIST_ARRIVED_THRES = 0.001 #목표 좌표와 거리가 이정도 차이 나는건 그냥 도착으로 간주하겟다 (1mm)
VECTOR_ORI_ARRIVED_THRES = 0.998 #목표 접근 벡터와 각도 차이가 이정도 나면 도착으로 간주하겟다

#=====================================

# =====================================================================
# 1. 하이퍼파라미터 및 환경 설정
# =====================================================================

# 연장 학습 희망시, 기존 가중치 파일 로드
# 초기부터 진행시 None
LOAD_WEIGHTS_PATH = None

#DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096
MAX_STEPS = int((math.pi * 2) / MAX_DELTA_Q) # 가상 롤아웃 스텝 수 (N) [적응형]
LR = 3e-3
EPOCHS = 100000

# 정규화를 위한 상수
MAX_WORKSPACE_DIST = 1.5 # 로봇 최대 도달 거리 (예: 1.5m) #L_normed_reach
SMOOTH_DENOM = (MAX_STEPS - 1) * (4 * (MAX_DELTA_Q ** 2)) #L_normed_smooth

#조연 가중치 텀 스케일링 팩터
SMOOTH_LOSS_SCALE = 0.2
SINGULAR_LOSS_SCALE = 0.1
STEP_REACH_LOSS_SCALE = 0.1
STEP_COS_LOSS_SCALE = 0.1

#loss range 계산
LOSS_RANGE = (1.0 + 1.0) + (SMOOTH_LOSS_SCALE + SINGULAR_LOSS_SCALE + STEP_REACH_LOSS_SCALE + STEP_COS_LOSS_SCALE) #메인로스 + 서브로스

# =======================================================================================
# urdf read + joint limit parse
import xml.etree.ElementTree as ET

# 1. URDF 파일 읽기 및 체인 생성
urdf_str = open(URDF_PATH).read()
# [수정 후] 🌟 SerialChain으로 만들면서 EE_LINK_NAME을 한 번에 지정
chain = pk.build_serial_chain_from_urdf(urdf_str, EE_LINK_NAME).to(device=DEVICE)

# =====================================================================
# 2. URDF에서 동적(Active) 조인트의 Limit 자동 추출
# =====================================================================
# pytorch_kinematics가 학습(q 텐서)에 실제로 사용하는 조인트 이름들만 가져오기
joint_names = chain.get_joint_parameter_names()

root = ET.fromstring(urdf_str)
q_min_list = []
q_max_list = []

for name in joint_names:
    # 해당 이름을 가진 joint 태그 찾기
    joint_node = root.find(f".//joint[@name='{name}']")
    
    if joint_node is not None:
        limit_node = joint_node.find('limit')
        if limit_node is not None:
            # lower, upper 속성값을 읽어서 float로 변환
            # (만약 속성이 누락되었다면 기본값으로 -pi, pi를 줌)
            lower = float(limit_node.get('lower', -3.14159))
            upper = float(limit_node.get('upper', 3.14159))
        else:
            # limit 태그가 없는 연속 회전(continuous) 조인트의 경우
            lower, upper = -3.14159, 3.14159
    else:
        lower, upper = -3.14159, 3.14159
        
    q_min_list.append(lower)
    q_max_list.append(upper)

# 파이토치 텐서로 최종 변환
q_min = torch.tensor(q_min_list, device=DEVICE)
q_max = torch.tensor(q_max_list, device=DEVICE)

print(f"✅ 활성화된 조인트: {joint_names}")
print(f"✅ q_min: {q_min}")
print(f"✅ q_max: {q_max}")

# =======================================================================================

# =====================================================================
# 2. 네트워크 모델 정의 (ResNet 스타일의 상태 갱신을 위한 MLP)
# =====================================================================
import torch
import torch.nn as nn

class controlMLP(nn.Module):
    def __init__(self, q_min, q_max, max_delta):
        super(controlMLP, self).__init__()
        
        # 외부에서 파싱해 온 조인트 리밋을 모델 내부 변수로 저장
        self.q_min = q_min
        self.q_max = q_max
        self.max_delta = max_delta

        # 네트워크 구조 (입력 15 -> 출력 6)
        self.net = nn.Sequential(
            nn.Linear(15, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 6), #출력층
            nn.Tanh() # 노트의 tanh (-1 ~ 1) 출력
        )

    def normalize_q_curr(self, q_curr):
        """
        raw q_curr을 각 joint의 URDF limit 기준으로 정규화한다.

        q_curr == q_min -> -1
        q_curr == q_max -> +1

        q_curr shape: (B, 6)
        q_min/q_max shape: (6,)
        PyTorch broadcasting으로 joint별 정규화가 적용된다.
        """
        q_norm = 2.0 * (q_curr - self.q_min) / (self.q_max - self.q_min) - 1.0
        q_norm = torch.clamp(q_norm, -1.0, 1.0)

        return q_norm

    def forward(self, q_curr, P_tgt, A_tgt, A_up_tgt):

        # 0. q_curr => q_norm 으로 정규화
        q_norm = self.normalize_q_curr(q_curr)

        # 1. 상태 결합 및 네트워크 추론
        x = torch.cat([q_norm, P_tgt, A_tgt, A_up_tgt], dim=-1)
        a = self.net(x) # a는 [-1, 1] 범위를 가지는 shape: (Batch, 6)
        
        # ==========================================================
        # Action Scaling 로직 (노트의 공식 완벽 구현)
        # ==========================================================
        
        # 2. 이번 스텝의 허용 한계(Upper/Lower) 계산
        # torch.clamp를 응용하면 min/max 연산을 배치 텐서에 대해 초고속으로 수행할 수 있습니다.
        
        # delta_upper = min(delta_max, q_max - q_curr)
        delta_upper = torch.clamp(self.q_max - q_curr, max=self.max_delta)
        
        # delta_lower = max(-delta_max, q_min - q_curr)
        delta_lower = torch.clamp(self.q_min - q_curr, min=-self.max_delta)
        
        # 3. tanh 출력(-1 ~ 1)을 0 ~ 1 범위로 정규화
        a_normalized = (a + 1.0) / 2.0
        
        # 4. 동적 범위에 선형 매핑
        delta_q = delta_lower + a_normalized * (delta_upper - delta_lower)
        
        return delta_q

# =====================================================================
# 3. 유틸리티 함수: FK 기반 타겟 데이터 무작위 생성 (순기구학 샘플링)
# =====================================================================
def generate_valid_targets(batch_size):
    """
    무작위 관절 각도를 생성하여, 그리퍼 끝단 기준의
    도달 가능한 완벽한 P(위치), A(접근 벡터), A_up(업 벡터) 타겟을 뽑아냄.
    정답지
    """
    # 1. 랜덤 조인트 생성 (앞서 파싱한 q_min, q_max 범위 내에서 생성)
    rand_noise = torch.rand(batch_size, 6, device=DEVICE)
    q_target_rand = q_min + rand_noise * (q_max - q_min)
    
    # 2. FK를 통해 베이스 좌표계 기준 'link_6'의 변환 행렬 도출
    ret = chain.forward_kinematics(q_target_rand)
    T_link6 = ret.get_matrix() # Shape: (batch_size, 4, 4)
    
    # 3. 로컬 Z축 오프셋 행렬 생성 및 적용 (그리퍼 길이 반영)
    # 대각선이 1인 4x4 단위 행렬(Identity Matrix)을 배치 사이즈만큼 복사
    T_local = torch.eye(4, device=DEVICE).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 로컬 좌표계의 Z축 위치(row 2, col 3)에 오프셋 값 삽입
    T_local[:, 2, 3] = Z_OFFSET 
    
    # link_6 변환 행렬의 '오른쪽'에 곱하여 최종 그리퍼 월드 좌표계 도출 (T_tcp = T_link6 * T_local)
    T_tcp = torch.matmul(T_link6, T_local) # Shape: (batch_size, 4, 4)
    
    # 4. 최종 TCP 변환 행렬에서 위치(P) 및 방향(A) 추출 [월드]
    # 4x4 행렬의 3열(인덱스 3)의 x, y, z 값이 위치(Translation) 벡터
    P_target = T_tcp[:, :3, 3]
    
    # 4x4 행렬의 2열(인덱스 2)이 로컬 Z축이 바라보는 방향 (접근 벡터, Approach)
    A_target = T_tcp[:, :3, 2]    
    
    # 4x4 행렬의 0열(인덱스 0)이 로컬 X축이 바라보는 방향 (업 벡터, Up)
    A_up_target = T_tcp[:, :3, 0] 
    
    return P_target, A_target, A_up_target

def get_kinematics(q):
    """
    주어진 q에 대한 link_6의 P, A, w(조작성 지수) 계산
    => w만 사용하긴 함
    """
    ret = chain.forward_kinematics(q)
    T = ret.get_matrix()
    
    # link_6 로컬 좌표계 원점이 월드 기준으로 어디에 있나
    P_curr = T[:, :3, 3]
    A_curr = T[:, :3, 2]
    A_up_curr = T[:, :3, 0]
    
    # 자코비안 및 요시카와 지수(w) 계산
    # [수정 후] 🌟 자코비안에서도 파라미터 삭제 (SerialChain은 자동으로 끝단 자코비안을 구해줌)
    J = chain.jacobian(q) 
    J_J_T = torch.matmul(J, J.transpose(-2, -1))
    
    # 수치적 안정성을 위해 abs 및 아주 작은 값(1e-6) 더함
    det = torch.abs(torch.linalg.det(J_J_T)) + 1e-6 
    w = torch.sqrt(det)
    
    return P_curr, A_curr, A_up_curr, w

def get_kinematics_of_GRIPPER(q):
    """
    주어진 q에 대해 link_6의 좌표를 구하고, (world)
    로컬 Z축으로 Z_OFFSET만큼 이동한 최종 TCP의 gripper, g, g_up 반환
    LOSS CALCULATION의 핵심
    """
    B = q.shape[0] # 배치 사이즈
    
    # 1. Base 기준 link_6의 변환 행렬 계산
    ret = chain.forward_kinematics(q)
    T_link6 = ret.get_matrix() # Shape: (B, 4, 4)
    
    # 2. 로컬 오프셋 행렬 생성 (배치 사이즈만큼 복사)
    # 대각선이 1인 4x4 단위 행렬 생성
    T_local = torch.eye(4, device=DEVICE).unsqueeze(0).repeat(B, 1, 1)
    # Z축 이동 거리에 offset 삽입
    T_local[:, 2, 3] = Z_OFFSET 
    
    # 3. 행렬 곱셈으로 최종 TCP 좌표 도출 (Right-Multiply)
    # T_tcp = T_link6 @ T_local => 월드 기준!
    T_tcp = torch.matmul(T_link6, T_local)
    
    # 4. 위치(gripper) 및 방향(g, g_up) 추출
    gripper_curr = T_tcp[:, :3, 3]    # 최종 손끝 위치
    g_curr = T_tcp[:, :3, 2]    # link6의 로컬 Z축이 향하는 방향 (접근 벡터)
    g_up_curr = T_tcp[:, :3, 0] # link6의 로컬 X축이 향하는 방향 (Up 벡터)
    
    return gripper_curr, g_curr, g_up_curr

# =====================================================================
# 4. 메인 학습 루프 (BPTT)
# =====================================================================

model = controlMLP(q_min, q_max, MAX_DELTA_Q).to(DEVICE)

# 🌟 [NEW] 가중치 불러오기 로직
if LOAD_WEIGHTS_PATH:
    import os
    if os.path.exists(LOAD_WEIGHTS_PATH):
        # 1. weights_only=True 추가로 보안 경고(FutureWarning) 해결
        state_dict = torch.load(LOAD_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        
        # 2. torch.compile이 남긴 '_orig_mod.' 껍데기 이름표 떼어내기
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace('_orig_mod.', '')
            clean_state_dict[clean_key] = v
            
        # 3. 세척된 가중치를 쌩얼 모델에 덮어씌우기
        model.load_state_dict(clean_state_dict)
        print(f"✅ 기존 학습 가중치 '{LOAD_WEIGHTS_PATH}'를 성공적으로 불러왔습니다! 이어서 학습합니다.")
    else:
        print(f"⚠️ 경고: '{LOAD_WEIGHTS_PATH}' 파일이 없습니다. 처음부터 새로 학습합니다.")
print()

# 모델 컴파일 (가중치를 먼저 덮어씌운 후에 컴파일하는 것이 가장 안전합니다)
model = torch.compile(model)

optimizer = optim.Adam(model.parameters(), lr=LR)
# 🌟 [NEW] 학습률 스케줄러 추가: 3000 에폭마다 보폭(LR)을 절반(0.5)으로 줄여줍니다!
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

#학습 시작 시간 기록
time_start = time.time()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # 1. 학습 환경 세팅 (랜덤 초기 자세 & 도달 가능한 랜덤 타겟 생성)
    P_target, A_target, A_up_target = generate_valid_targets(BATCH_SIZE)
    
    # 초기 로봇 자세 생성 (실제로는 특정 Home 자세에서 출발하게 해도 됨)
    q_curr = q_min + torch.rand(BATCH_SIZE, 6, device=DEVICE) * (q_max - q_min)
    
    # 누적 로스 변수 초기화
    loss_smooth = 0.0
    loss_singular = 0.0
    loss_step_reach = 0.0
    loss_step_cos = 0.0
    
    delta_q_prev = torch.zeros(BATCH_SIZE, 6, device=DEVICE)
    
    # 🌟 [NEW] 활성화 마스크 (1.0 = 이동 중, 0.0 = 도달 완료)
    # Shape: (BATCH_SIZE, 1) -> 나중에 델타q(BATCH_SIZE, 6)와 곱하기 위해 차원을 맞춤
    active_mask = torch.ones(BATCH_SIZE, 1, device=DEVICE)

    # 2. 가상 롤아웃 (BPTT Unfolding)
    for step in range(MAX_STEPS):
        # 1) 추론
        delta_q = model(q_curr, P_target, A_target, A_up_target)
        
        # 🌟 [NEW] 마스크 적용: 이미 도달한 로봇은 delta_q를 강제로 0으로 만듦
        delta_q = delta_q * active_mask
        
        # 2) 상태 갱신
        q_curr = q_curr + delta_q
        
        #---------------------------------------------------------------------------
        # ROBOT MOVE
        #=> FK를 사용하여, 저 각도면 로봇이 어떤 자세일지를 계산

        # 3) 현재 뼈대 각도에 대한 FK 정보 추출
        _, _, _, w_curr = get_kinematics(q_curr)
        # 🌟 진짜 그리퍼 손끝 좌표 추출
        gripper_curr, g_curr, g_up_curr = get_kinematics_of_GRIPPER(q_curr)
        #---------------------------------------------------------------------------
        
        # ============도달 판정 및 마스크 업데이트=========================================

        dist = torch.norm(gripper_curr - P_target, dim=1) #목표와의 오차는?
        cos_A = torch.sum(g_curr * A_target, dim=1).clamp(-1.0, 1.0) #목표와의 벡터 오차
        cos_A_up = torch.sum(g_up_curr * A_up_target, dim=1).clamp(-1.0, 1.0) #목표와의 벡터 오차
        
        # 도달 판정 (Boolean Tensor, Shape: (BATCH_SIZE,))
        dist_reached = dist < DIST_ARRIVED_THRES 
        ori_reached = (cos_A > VECTOR_ORI_ARRIVED_THRES) & (cos_A_up > VECTOR_ORI_ARRIVED_THRES)
        
        is_reached = dist_reached & ori_reached
        
        # 도달한 로봇의 마스크를 0.0으로 깎아버림
        # (~is_reached)는 True를 False로 뒤집어 줌. float()를 씌우면 0.0이 됨.
        # 이미 0이 된 녀석은 계속 0을 유지하도록 누적 곱셈(*) 사용
        active_mask = active_mask * (~is_reached).float().unsqueeze(1)

        #=============================================================================
        
        # 5) [Step-wise 로스 누적]

        # 부드러움 로스
        if step > 0:
            diff_sq = torch.norm(delta_q - delta_q_prev, dim=1)**2
            loss_smooth += diff_sq.mean()

        # 특이점 로스
        loss_singular += torch.exp(-20.0 * w_curr).mean() 
        
        # step-wise reach loss
        loss_step_reach += (dist / MAX_WORKSPACE_DIST).mean()

        # step-wise cos loss
        loss_step_cos += ((1.0 - cos_A) + (1.0 - cos_A_up)).mean() / 4.0 # 최대 4.0이므로 정규화

        #================================================
            
        delta_q_prev = delta_q

    #=============================================================================
    # 3. [Terminal 로스 연산] N번의 루프가 끝난 최종 상태의 벡터 정렬 상태 확인

    # [MAIN LOSS TERM]
    # REACH LOSS
    loss_reach = (dist / MAX_WORKSPACE_DIST).mean()

    # COS LOSS
    loss_cosine = ((1.0 - cos_A) + (1.0 - cos_A_up)).mean() / 4.0 # 최대 4.0이므로 정규화

    #-----------------------------------
    # [SUB LOSS TERM]

    # 4. 로스 스케일링(Auto-Normalization) 및 합산
    loss_smooth = loss_smooth / SMOOTH_DENOM
    loss_singular = loss_singular / MAX_STEPS
    loss_step_reach = loss_step_reach / MAX_STEPS
    loss_step_cos = loss_step_cos / MAX_STEPS

    # sub loss scaling
    loss_smooth = (SMOOTH_LOSS_SCALE * loss_smooth)
    loss_singular = (SINGULAR_LOSS_SCALE * loss_singular)
    loss_step_reach = (STEP_REACH_LOSS_SCALE * loss_step_reach)
    loss_step_cos = (STEP_COS_LOSS_SCALE * loss_step_cos)

    #-----------------------------------
    
    #TOTAL LOSS#
    main_loss = (loss_reach + loss_cosine)
    sub_loss = loss_smooth + loss_singular + loss_step_reach + loss_step_cos
    
    total_loss = main_loss + sub_loss

    #-----------------------------------------------------------------------------

    # 5. 역전파 및 가중치 업데이트
    total_loss.backward()
    
    # 🌟 Gradient Clipping: 기울기 폭발 완벽 차단
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    # 🌟 [NEW] 에폭이 끝날 때마다 스케줄러 한 걸음 진행
    scheduler.step()

    #-----------------------------------------------------------------------------
    # 로그 출력
    if epoch % 50 == 0:
        print(f"Epoch {epoch:04d} | Total: {total_loss.item():.4f}"
            f"[{((1.0 - (total_loss.item() / LOSS_RANGE)) * 100):.4f}% perfection] "
            f"(Reach: {loss_reach.item():.4f}, Cos: {loss_cosine.item():.4f}, "
            f"Smth: {loss_smooth.item():.4f}, Sing: {loss_singular.item():.4f}, "
            f"StepReach: {loss_step_reach.item():.4f}, StepCos: {loss_step_cos.item():.4f})")
        
    # 중간 체크포인트 저장
    if epoch > 0 and epoch % 1000 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
        print(f"💾 중간 저장 완료: checkpoint_epoch_{epoch}.pth")

# =====================================================================
# 🌟 [NEW] 5. 학습 종료 후 최종 가중치 저장
# =====================================================================
print(f"전체 학습이 모두 완료되었습니다!")
print(f"총 훈련 시간 wall-time (hrs) : {(time.time() - time_start) / 3600:.2f}")

torch.save(model.state_dict(), "trained_mlp_weights.pth")

print("✅ 'trained_mlp_weights.pth' 파일로 가중치가 안전하게 저장되었습니다.")