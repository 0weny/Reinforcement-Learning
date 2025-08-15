import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


fig = plt.figure(figsize = (5,5))


#붉은 벽 그리기
plt.plot([1,1],[0,1], color = 'red', linewidth =2)
plt.plot([1,2],[2,2], color = 'red', linewidth =2)
plt.plot([2,2],[2,1], color = 'red', linewidth =2)
plt.plot([2,3],[1,1], color = 'red', linewidth =2)


# 상태표시

plt.text(0.5,2.5,'S0',size = 14, ha = 'center')
plt.text(1.5,2.5,'S1',size = 14, ha = 'center')
plt.text(2.5,2.5,'S2',size = 14, ha = 'center')
plt.text(0.5,1.5,'S3',size = 14, ha = 'center')
plt.text(1.5,1.5,'S4',size = 14, ha = 'center')
plt.text(2.5,1.5,'S5',size = 14, ha = 'center')
plt.text(0.5,0.5,'S6',size = 14, ha = 'center')
plt.text(1.5,0.5,'S7',size = 14, ha = 'center')
plt.text(2.5,0.5,'S8',size = 14, ha = 'center')
plt.text(0.5,2.3,'START', ha = 'center')
plt.text(2.5,0.3,'GOAL', ha = 'center')


#그림 그릴 범위 및 눈금 제거
ax = plt.gca();
ax.set_xlim(0,3)
ax.set_ylim(0,3)
plt.tick_params(axis='both', which='both', bottom =False, top =False, labelbottom =False, right =False, left = False, labelleft=False)

line, = ax.plot([0.5,],[2.5], marker = "o", color ="g", markersize = 60)


#강화학습 파라미터 초기값 theta_0 : 정책을 결정
#nan => 0대신 nan으로 설정
#행: 상테, 열: 행동방향(상우하좌)

theta_0 = np.array(
[[np.nan, 1, 1, np.nan ], # s0
[np.nan, 1, np.nan, 1 ],  # s1
[np.nan, np.nan, 1, 1 ],  # s2
[1, 1, 1, np.nan ],       # s3
[np.nan, np.nan, 1, 1 ],  # s4
[1, np.nan, np.nan, np.nan ], # s5
[1, np.nan, np.nan, np.nan ], # s6
[1, 1, np.nan, np.nan ],      # s7
]) 

# 강화학습 파라메터 초기값 𝜃0 --> 확률로 변환

def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape # theta의 행렬 크기를 구함
    pi = np.zeros((m, n)) # m x n 행렬을 0으로 채움
    for i in range(0, m): # m개의 행에 대한 반복
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :]) # 비율 계산 
                                                        # nansum: nan 제외하고 합산
    pi = np.nan_to_num(pi) # nan을 0으로 변환
    return pi

# 무작위 행동정책 pi_0을 계산
pi_0 = simple_convert_into_pi_from_theta(theta_0)
print(pi_0)

" ε-greedy 알고리즘 구현"
# -------------------------------------
def get_action(s, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    # 행동을 결정
    if np.random.rand() < epsilon:
        # 확률 ε로 무작위 행동을 선택함
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
# Q값이 최대가 되는 행동을 선택함
        next_direction = direction[np.nanargmax(Q[s, :])]
# 행동을 인덱스로 변환
    
    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3
    return action


def get_s_next(s, a, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a] # 행동 a의 방향
      #행동으로 다음 상태를 결정
    if next_direction == "up":
        s_next = s - 3 # 위로 이동하면 상태값이 3 줄어든다
    elif next_direction == "right":
        s_next = s + 1 # 오른쪽으로 이동하면 상태값이 1 늘어난다
    elif next_direction == "down":
        s_next = s + 3 # 아래로 이동하면 상태값이 3 늘어난다
    elif next_direction == "left":
        s_next = s - 1 # 왼쪽으로 이동하면 상태값이 1 줄어든다
    return s_next

"Q-learning 사용하여 미로 빠져 나오기"
"-------------------------------------"
# Q러닝 알고리즘으로 미로를 빠져나오는 함수, 상태 및 행동 그리고 Q값의 히스토리를 출력







# 1단계 이동한 후의 상태 s를 계산하는 함수
def get_next_s(pi, s): # 현재 상태 s에서 정책 pi를 따라 행동한 후 next state 계산
    direction = ["up", "right", "down", "left"]
    
    next_direction = np.random.choice(direction, p=pi[s, :])
    # pi[s,:]의 확률에 따라, direction 값이 선택된다
    
    if next_direction == "up":
        s_next = s - 3 # 위로 이동하면 상태값이 3 줄어든다
    elif next_direction == "right":
        s_next = s + 1 # 오른쪽으로 이동하면 상태값이 1 늘어난다
    elif next_direction == "down":
        s_next = s + 3 # 아래로 이동하면 상태값이 3 늘어난다
    elif next_direction == "left":
        s_next = s - 1 # 왼쪽으로 이동하면 상태값이 1 줄어든다
    
    return s_next



# Keep moving towards the goal
def goal_maze(pi):
    s = 0 # 시작 지점 : S0에서 시작
    state_history = [0] # 에이전트의 경로를 기록하는 리스트 초기화
    
    while (1): # 목표 지점에 이를 때까지 반복
        next_s = get_next_s(pi, s)
        state_history.append(next_s) # 경로 리스트에 다음 상태(위치)를 추가
        # [0, 1, …]
        if next_s == 8: # 목표 지점에 이르면 종료
            break
        else:
            s = next_s
            
    return state_history


pi_0 = simple_convert_into_pi_from_theta(theta_0)
print(pi_0)

# 목표 지점에 이를 때까지 미로 안을 이동
state_history = goal_maze(pi_0)  # goal을 향하여 pi_0 정책을 따라 계속 진행

print(state_history)
print("목표 지점에 이르기까지 걸린 단계 수는 " + str(len(state_history) - 1) + "단계입니다")

# 애니메이션을 그리기 위한 설정
state = state_history[i] # 현재 위치
x = (state % 3) + 0.5 # 열 --> x좌표 : (state % 3) 계산 결과 0, 1, 2
y = 2.5 - int(state / 3) # 행 --> y좌표 : (state / 3) 계산 결과 0, 1, 2


def init():
#배경 이미지 초기화
    line.set_data([], [])
    return (line,)

def animate(i):
    """프레임 단위로 이미지 생성"""
    state = state_history[i] # 현재 위치
    x = (state % 3) + 0.5 # 열 --> x좌표 : (state % 3) 계산 결과 0, 1, 2
    y = 2.5 - int(state / 3) # 행 --> y좌표 : (state / 3) 계산 결과 0, 1, 2
    line.set_data(x, y)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history), interval=200, repeat=False)

plt.show()










