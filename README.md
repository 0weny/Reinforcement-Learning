# Reinforcement-Learning Mazes — Policy Simulation & Q-learning
기본적인 강화학습 알고리즘을 구현하였습니다.


강화학습 미로 코드:
- **maze1.py**: 확률 정책(Policy)으로 3×3 미로를 시뮬레이션하고 애니메이션으로 경로를 시각화
- **maze2.py**: Q-learning으로 5×5 미로에서 목표 보상을 최대화하는 경로를 학습하고 애니메이션으로 시각화


## 1. maze1.py — 정책 기반 확률 시뮬레이션 (3×3)
- 상태별 허용 행동을 1/NaN으로 둔 행렬 `theta_0`로 정의 → 정규화해서 정책 π 생성 → π에 따라 확률적으로 다음 상태 이동
- π에 따라 무작위 전이, 목표 도달 시 종료


![Maze1 Demo](https://github.com/0weny/Reinforcement-Learning/blob/main/static/maze1.gif?raw=true)


<br/>
<br/>

## 2. maze2.py — Q-learning (5×5)
- Q-learning으로 상태-행동 가치Q를 업데이트(ε-greedy)
  -> 보상 기반 학습으로 Q(s,a) 업데이트
  -> ε-greedy로 탐험/활용 균형

![Maze2 Demo](https://github.com/0weny/Reinforcement-Learning/blob/main/static/maze2.gif?raw=true)

<br/><br/>
## 🔧 환경 요구사항
- Python 3.9+ 
- 패키지: `numpy`, `matplotlib`




