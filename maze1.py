import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------
# 초기 정책 파라미터 설정
# ---------------------------
theta_0 = np.array([
    [np.nan, 1, 1, np.nan],  # s0
    [np.nan, 1, np.nan, 1],  # s1
    [np.nan, np.nan, 1, 1],  # s2
    [1, 1, 1, np.nan],       # s3
    [np.nan, np.nan, 1, 1],  # s4
    [1, np.nan, np.nan, np.nan],  # s5
    [1, np.nan, np.nan, np.nan],  # s6
    [1, 1, np.nan, np.nan]   # s7
])

# ---------------------------
# 정책 확률 계산 함수
# ---------------------------
def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

# ---------------------------
# 다음 상태 계산 함수
# ---------------------------
def get_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p=pi[s, :])
    if next_direction == "up":
        s_next = s - 3
    elif next_direction == "right":
        s_next = s + 1
    elif next_direction == "down":
        s_next = s + 3
    elif next_direction == "left":
        s_next = s - 1
    return s_next

# ---------------------------
# 목표 지점 도달까지 시뮬레이션
# ---------------------------
def goal_maze(pi):
    s = 0
    state_history = [0]
    while True:
        next_s = get_next_s(pi, s)
        state_history.append(next_s)
        if next_s == 8:
            break
        else:
            s = next_s
    return state_history

# 정책 생성
pi_0 = simple_convert_into_pi_from_theta(theta_0)
state_history = goal_maze(pi_0)
print(state_history)
print("목표 지점에 이르기까지 걸린 단계 수는 " + str(len(state_history) - 1) + "단계입니다")

# ---------------------------
# 미로 환경 그리기
# ---------------------------
fig = plt.figure(figsize=(5,5))
plt.plot([1,1], [0,1], color='red', linewidth=2)
plt.plot([1,2], [2,2], color='red', linewidth=2)
plt.plot([2,2], [2,1], color='red', linewidth=2)
plt.plot([2,3], [1,1], color='red', linewidth=2)

# 상태를 의미하는 문자열(S0~S8) 표시
plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')

# 축 범위 및 눈금 제거
ax = plt.gca()
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False)

# 애니메이션 점 생성
line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)

# ---------------------------
# Green Ball History Visualization 이동 궤적 시각화
# ---------------------------

def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    state = state_history[i]  # ✅ 이건 OK
    x = (state % 3) + 0.5
    y = 2.5 - int(state / 3)
    line.set_data([x], [y])
    return (line,)


# 애니메이션 생성
anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(state_history), interval=300, repeat=False)

plt.show()