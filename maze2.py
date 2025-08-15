import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 미로 환경 정의
maze = np.array([
    ['.', '.', '.', '.', '.'],
    ['.', 'W', 'W', 'W', '.'],
    ['.', 'W', 'W', 'W', 'G'],
    ['O', '.', '.', '.', '.'],
    ['C', 'C', 'C', 'C', 'C']
], dtype=object)

n_states = 25
n_actions = 4
start_state = 15  # (3,0)
goal_state = 14   # (2,4)

# 좌표 변환
def state_to_pos(state):
    return divmod(state, 5)

def pos_to_state(pos):
    return pos[0] * 5 + pos[1]

# 이동 가능 여부
def is_valid(pos):
    r, c = pos
    return 0 <= r < 5 and 0 <= c < 5 and maze[r][c] != 'W'

# 다음 상태 계산
def get_next_state(state, action):
    r, c = state_to_pos(state)
    drc = [(-1,0),(0,1),(1,0),(0,-1)]
    nr, nc = r + drc[action][0], c + drc[action][1]
    return pos_to_state((nr,nc)) if is_valid((nr,nc)) else state

# 보상 함수
def get_reward(state):
    r, c = state_to_pos(state)
    val = maze[r][c]
    if val == 'G': return 10
    if val == 'C': return -10
    return 0

# Q-learning 관련
def get_action(state, Q, epsilon):
    return np.random.choice(n_actions) if np.random.rand() < epsilon else np.nanargmax(Q[state])

def Q_learning(s, a, r, s_next, Q, eta, gamma):
    Q[s,a] += eta * (r + gamma * np.nanmax(Q[s_next]) - Q[s,a])
    return Q

# Q 초기화
Q = np.random.rand(n_states, n_actions)
for s in range(n_states):
    for a in range(n_actions):
        if s == get_next_state(s, a):
            Q[s, a] = np.nan

# 학습 후 경로 저장
s = start_state
path = [s]
epsilon = 0.1
eta, gamma = 0.1, 0.9

# 1번 에피소드로 경로 학습
while True:
    a = get_action(s, Q, epsilon)
    s_next = get_next_state(s, a)
    r = get_reward(s_next)
    Q = Q_learning(s, a, r, s_next, Q, eta, gamma)
    path.append(s_next)
    if s_next == goal_state:
        break
    s = s_next




"----------------시각화-----------------"
# 색상 매핑
color_map = {'O': 'orange', 'G': 'green', 'C': 'red', '.': 'blue', 'W': 'gray'}

# 시각화
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
plt.xticks([]); plt.yticks([]); ax.set_aspect('equal')

# 초기 미로 그리기
cell_patches = {}
for r in range(5):
    for c in range(5):
        val = maze[r][c]
        color = color_map[val]
        patch = plt.Rectangle((c, 4 - r), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(patch)
        cell_patches[(r, c)] = patch
        if val == 'G':
            ax.text(c + 0.5, 4 - r + 0.4, '10', ha='center', fontsize=10)
        elif val == 'C':
            ax.text(c + 0.5, 4 - r + 0.4, '-10', ha='center', fontsize=10)

def animate(i):
    if i > 0:
        # 이전 위치 원상복구
        prev_r, prev_c = state_to_pos(path[i - 1])
        prev_val = maze[prev_r][prev_c]
        cell_patches[(prev_r, prev_c)].set_facecolor(color_map[prev_val])
    # 현재 위치 노란색으로
    r, c = state_to_pos(path[i])
    cell_patches[(r, c)].set_facecolor('yellow')
    return list(cell_patches.values())

ani = animation.FuncAnimation(fig, animate, frames=len(path), interval=400, repeat=False)
plt.show()


