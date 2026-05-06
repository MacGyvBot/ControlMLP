# ============================================================
# debug_gripper_frame.py
#
# 목적:
# 1) URDF를 읽는다.
# 2) 특정 joint angle q에서 FK를 수행한다.
# 3) world 좌표계 기준으로
#    - link_6 원점
#    - link_6 local axes
#    - gripper_point (= link_6 원점에서 local +z로 Z_OFFSET 이동)
#    - g (= local +z)
#    - g_up (= local +x)
#    를 3D로 시각화한다.
# 4) 추가로 tool0 frame도 같이 그려서 비교한다.
#
# 필요 패키지:
# pip install matplotlib torch pytorch-kinematics
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_kinematics as pk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# 사용자 설정
# ============================================================
URDF_PATH = r"/home/ssu/Desktop/coop_controlMLP/m0609.urdf"   # <- 네 환경에 맞게 바꿔도 됨
Z_OFFSET = 0.22
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 디버그할 관절각 (deg)
# 일단 0자세부터 보고, 그 다음 네가 원하는 자세로 바꿔가며 확인해.
Q_DEG = [0, 0, 0, 0, 0, 0]

# 축 길이 표시용
WORLD_AXIS_LEN = 0.15
LOCAL_AXIS_LEN = 0.10
GRIPPER_VEC_LEN = 0.12

SHOW_TOOL0 = True
SHOW_LINK_CHAIN = True
SHOW_TEXT_LABEL = True


# ============================================================
# 유틸리티
# ============================================================
def to_np(x):
    return x.detach().cpu().numpy()

def normalize(v, eps=1e-9):
    return v / (torch.norm(v) + eps)

def set_axes_equal(ax):
    """3D plot 축 비율을 동일하게 맞춤"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

def draw_frame(ax, origin, R, axis_len=0.1, label_prefix=None, alpha=1.0, linewidth=2.5):
    """
    origin: (3,)
    R: (3,3) rotation matrix
       R[:,0] = x-axis
       R[:,1] = y-axis
       R[:,2] = z-axis
    """
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    # x: red, y: green, z: blue
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              length=axis_len, color='r', linewidth=linewidth, alpha=alpha)

    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              length=axis_len, color='g', linewidth=linewidth, alpha=alpha)

    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              length=axis_len, color='b', linewidth=linewidth, alpha=alpha)

    if label_prefix is not None and SHOW_TEXT_LABEL:
        ax.text(origin[0], origin[1], origin[2], label_prefix, fontsize=10, color='k')

def draw_vector(ax, origin, vec, length=0.1, color='orange', label=None, linewidth=3.0):
    v = vec / (np.linalg.norm(vec) + 1e-9)
    ax.quiver(origin[0], origin[1], origin[2],
              v[0], v[1], v[2],
              length=length, color=color, linewidth=linewidth)
    if label is not None and SHOW_TEXT_LABEL:
        tip = origin + v * length
        ax.text(tip[0], tip[1], tip[2], label, color=color, fontsize=10)

def print_matrix(name, T):
    np.set_printoptions(precision=4, suppress=True)
    print(f"\n{name} =")
    print(T)

# ============================================================
# 메인
# ============================================================
def main():
    # -----------------------------
    # 1) URDF 읽기
    # -----------------------------
    with open(URDF_PATH, "r", encoding="utf-8") as f:
        urdf_str = f.read()

    # link_6용 체인
    chain_link6 = pk.build_serial_chain_from_urdf(urdf_str, "link_6").to(device=DEVICE)

    # tool0 비교용 체인
    if SHOW_TOOL0:
        chain_tool0 = pk.build_serial_chain_from_urdf(urdf_str, "tool0").to(device=DEVICE)

    # q 준비
    q_deg = torch.tensor(Q_DEG, dtype=torch.float32, device=DEVICE)
    q = q_deg * torch.pi / 180.0
    q = q.unsqueeze(0)   # (1, 6)

    print("=========================================")
    print("Joint names used by chain_link6:")
    print(chain_link6.get_joint_parameter_names())
    print("q_deg =", Q_DEG)
    print("q_rad =", to_np(q[0]))
    print("=========================================")

    # -----------------------------
    # 2) 모든 링크 FK (link_6 체인)
    # end_only=False 가 핵심
    # -----------------------------
    fk_all = chain_link6.forward_kinematics(q, end_only=False)

    # 확인용
    print("\nAvailable frame keys from fk_all:")
    print(list(fk_all.keys()))

    # 필요한 프레임들
    T_link1 = fk_all["link_1"].get_matrix()[0]
    T_link2 = fk_all["link_2"].get_matrix()[0]
    T_link3 = fk_all["link_3"].get_matrix()[0]
    T_link4 = fk_all["link_4"].get_matrix()[0]
    T_link5 = fk_all["link_5"].get_matrix()[0]
    T_link6 = fk_all["link_6"].get_matrix()[0]

    # base_link는 월드 원점으로 간주
    T_base = torch.eye(4, device=DEVICE)

    # -----------------------------
    # 3) link_6 기반 gripper point / g / g_up 계산
    # -----------------------------
    p_link6 = T_link6[:3, 3]
    R_link6 = T_link6[:3, :3]

    x_link6 = normalize(R_link6[:, 0])   # local +x
    y_link6 = normalize(R_link6[:, 1])   # local +y
    z_link6 = normalize(R_link6[:, 2])   # local +z

    gripper_point = p_link6 + z_link6 * Z_OFFSET
    g = z_link6
    g_up = x_link6

    # 직교성 체크
    dot_g_gup = torch.dot(g, g_up).item()

    # -----------------------------
    # 4) tool0도 비교
    # -----------------------------
    if SHOW_TOOL0:
        T_tool0 = chain_tool0.forward_kinematics(q).get_matrix()[0]
        p_tool0 = T_tool0[:3, 3]
        R_tool0 = T_tool0[:3, :3]

        x_tool0 = normalize(R_tool0[:, 0])
        y_tool0 = normalize(R_tool0[:, 1])
        z_tool0 = normalize(R_tool0[:, 2])

        # link_6와 tool0 축 비교
        dot_g_tool0z = torch.dot(g, z_tool0).item()
        dot_gup_tool0x = torch.dot(g_up, x_tool0).item()
    else:
        T_tool0 = None

    # -----------------------------
    # 5) 수치 출력
    # -----------------------------
    print_matrix("T_link6", to_np(T_link6))
    print("\n[link_6 origin(world)] =", to_np(p_link6))
    print("[link_6 x_axis(world)] =", to_np(x_link6))
    print("[link_6 y_axis(world)] =", to_np(y_link6))
    print("[link_6 z_axis(world)] =", to_np(z_link6))

    print("\n[gripper_point(world)] =", to_np(gripper_point))
    print("[g (= link_6 local +z in world)] =", to_np(g))
    print("[g_up (= link_6 local +x in world)] =", to_np(g_up))
    print(f"[dot(g, g_up)] = {dot_g_gup:.8f}   (0에 매우 가까우면 수직)")

    if SHOW_TOOL0:
        print_matrix("T_tool0", to_np(T_tool0))
        print("\n[tool0 origin(world)] =", to_np(p_tool0))
        print("[tool0 x_axis(world)] =", to_np(x_tool0))
        print("[tool0 y_axis(world)] =", to_np(y_tool0))
        print("[tool0 z_axis(world)] =", to_np(z_tool0))
        print(f"[dot(g, tool0_z)] = {dot_g_tool0z:.8f}")
        print(f"[dot(g_up, tool0_x)] = {dot_gup_tool0x:.8f}")

    # -----------------------------
    # 6) 3D Plot
    # -----------------------------
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 월드 프레임
    world_origin = np.array([0.0, 0.0, 0.0])
    world_R = np.eye(3)
    draw_frame(ax, world_origin, world_R, axis_len=WORLD_AXIS_LEN, label_prefix="world")

    # 링크 원점들
    p_base = to_np(T_base[:3, 3])
    p1 = to_np(T_link1[:3, 3])
    p2 = to_np(T_link2[:3, 3])
    p3 = to_np(T_link3[:3, 3])
    p4 = to_np(T_link4[:3, 3])
    p5 = to_np(T_link5[:3, 3])
    p6 = to_np(p_link6)

    # 체인 시각화
    if SHOW_LINK_CHAIN:
        chain_pts = np.stack([p_base, p1, p2, p3, p4, p5, p6], axis=0)
        ax.plot(chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2],
                '-o', color='k', linewidth=2.0, markersize=5, label='robot chain')

    # link_6 frame
    draw_frame(ax, p6, to_np(R_link6), axis_len=LOCAL_AXIS_LEN, label_prefix="link_6")

    # gripper point
    gp = to_np(gripper_point)
    ax.scatter(gp[0], gp[1], gp[2], s=80, c='orange', label='gripper_point')
    if SHOW_TEXT_LABEL:
        ax.text(gp[0], gp[1], gp[2], "gripper_point", color='orange', fontsize=10)

    # link_6 origin도 표시
    ax.scatter(p6[0], p6[1], p6[2], s=60, c='purple', label='link_6 origin')

    # link_6 origin -> gripper_point 선분
    ax.plot([p6[0], gp[0]], [p6[1], gp[1]], [p6[2], gp[2]],
            '--', color='orange', linewidth=2.0, label='Z_OFFSET')

    # gripper point 기준 g, g_up
    draw_vector(ax, gp, to_np(g),    length=GRIPPER_VEC_LEN, color='darkorange', label='g (= +z of link_6)')
    draw_vector(ax, gp, to_np(g_up), length=GRIPPER_VEC_LEN, color='magenta',   label='g_up (= +x of link_6)')

    # 필요하면 세 번째 축도 같이 표시 가능
    g_side = np.cross(to_np(g), to_np(g_up))
    draw_vector(ax, gp, g_side, length=GRIPPER_VEC_LEN, color='cyan', label='g_side')

    # tool0 frame 비교
    if SHOW_TOOL0:
        pt = to_np(p_tool0)
        draw_frame(ax, pt, to_np(R_tool0), axis_len=LOCAL_AXIS_LEN * 0.9, label_prefix="tool0", alpha=0.7, linewidth=2.0)
        ax.scatter(pt[0], pt[1], pt[2], s=60, c='brown', label='tool0 origin')

    # 꾸미기
    ax.set_title("Debug visualization: link_6 / gripper_point / g / g_up", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")

    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()