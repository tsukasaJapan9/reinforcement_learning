import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -----------------------------------
# 迷路の描画
# -----------------------------------
fig =  plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1)

# 壁
ax.plot([1, 1], [0, 1], color='red', linewidth=2)
ax.plot([1, 2], [2, 2], color='red', linewidth=2)
ax.plot([2, 2], [2, 1], color='red', linewidth=2)
ax.plot([2, 3], [1, 1], color='red', linewidth=2)

# 状態
ax.text(0.5, 2.5, 'S0', size=14, ha='center')
ax.text(1.5, 2.5, 'S1', size=14, ha='center')
ax.text(2.5, 2.5, 'S2', size=14, ha='center')
ax.text(0.5, 1.5, 'S3', size=14, ha='center')
ax.text(1.5, 1.5, 'S4', size=14, ha='center')
ax.text(2.5, 1.5, 'S5', size=14, ha='center')
ax.text(0.5, 0.5, 'S6', size=14, ha='center')
ax.text(1.5, 0.5, 'S7', size=14, ha='center')
ax.text(2.5, 0.5, 'S8', size=14, ha='center')
ax.text(0.5, 2.3, 'START', ha='center')
ax.text(2.5, 0.3, 'GOAL', ha='center')

# 描画範囲の設定と目盛りを消す設定
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

# 現在地S0に緑丸を描画
line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)

# -----------------------------------
# 学習コード
# -----------------------------------
# 方策を決定するパラメータ
# 行は状態0~7、列は移動方向で↑、→、↓、←を表す
# 状態8はゴールなので、方策は無し
theta_0 = np.array([
    [np.nan, 1, 1, np.nan],  # s0
    [np.nan, 1, np.nan, 1],  # s1
    [np.nan, np.nan, 1, 1],  # s2
    [1, 1, 1, np.nan],  # s3
    [np.nan, np.nan, 1, 1],  # s4
    [1, np.nan, np.nan, np.nan],  # s5
    [1, np.nan, np.nan, np.nan],  # s6
    [1, 1, np.nan, np.nan],  # s7
])

def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

def softmax_convert_into_pi_from_theta(theta):
    beta = 1.0
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    exp_theta = np.exp(beta * theta)
    for i in range(0, m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

def get_next_s(pi, s):
    direction = ['up', 'right', 'down', 'left']
    next_direction = np.random.choice(direction, p=pi[s, :])
    if next_direction == 'up':
        s_next = s - 3
    elif next_direction == 'right':
        s_next = s + 1
    elif next_direction == 'down':
        s_next = s + 3
    elif next_direction == 'left':
        s_next = s - 1
    return s_next

ACTION = ['up', 'right', 'down', 'left']

def get_action_and_next_s(pi, s):
    direction = ['up', 'right', 'down', 'left']
    next_direction = np.random.choice(direction, p=pi[s, :])
    if next_direction == 'up':
        action = 0
        s_next = s - 3
    elif next_direction == 'right':
        action = 1
        s_next = s + 1
    elif next_direction == 'down':
        action = 2
        s_next = s + 3
    elif next_direction == 'left':
        action = 3
        s_next = s - 1
    return [action, s_next]

def goal_maze(pi):
    s = 0
    state_history = [0]
    while (1):
        next_s = get_next_s(pi, s)
        state_history.append(next_s)
        if next_s == 8:
            break
        else:
            s = next_s
    return state_history

def update_theta(theta, pi, s_a_history):
    eta = 0.1
    T = len(s_a_history) - 1
    [m, n] = theta.shape
    delta_theta = theta.copy()
    # stateのループ
    for i in range(0, m):
        # actionのループ
        for j in range(0, n):
            if not(np.isnan(theta[i, j])):
                # s_a_historyから状態iのものを取り出す
                SA_i = [SA for SA in s_a_history if SA[0] == i]

                # s_a_historyから状態iで行動jをしたものを取り出す
                SA_ij = [SA for SA in s_a_history if SA == [i, j]]
                N_i = len(SA_i)
                N_ij = len(SA_ij)
                delta_theta[i, j] = (N_ij + pi[i, j] * N_i) / T
    new_theta = theta + eta * delta_theta
    return new_theta

def goal_maze_ret_s_a(pi):
    s = 0
    # state/actionの履歴
    # 最初の状態でstateは0、まだ行動してないのでnan
    # stateの状態の時に次にどの行動をとったのかを記録
    s_a_history = [[0, np.nan]]
    while (1):
        [action, next_s] = get_action_and_next_s(pi, s)
        # リストの最後の要素(-1)のindex:1にactionを代入してる
        # つまり最初の状態であればnp.nanに採用したactionを入れている
        s_a_history[-1][1] = action
        # ここでstateがnext_sに更新されるが、actionは次のループのget_action_and_next_sで決まるため、nanを入れている
        s_a_history.append([next_s, np.nan])
        if next_s == 8:
            break
        else:
            s = next_s
    return s_a_history

# -----------------------------------
# 学習
# -----------------------------------
stop_epsilon = 10**-8
theta = theta_0
pi_0 = softmax_convert_into_pi_from_theta(theta)
pi = pi_0

count = 1
while True:
    # ゴールまで進めて[状態, 行動]の履歴を取得
    s_a_history = goal_maze_ret_s_a(pi)
    # thetaの更新
    new_theta = update_theta(theta, pi, s_a_history)
    new_pi = softmax_convert_into_pi_from_theta(new_theta)
    print(np.sum(np.abs(new_pi - pi)))
    print('迷路を解くのにかかったステップ数は' + str(len(s_a_history) - 1) + 'です')
    if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
        break
    else:
        theta = new_theta
        pi = new_pi

np.set_printoptions(precision=3, suppress=True)
print(pi)

def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    state = s_a_history[i][0]
    x = (state % 3) + 0.5
    y = 2.5 - int(state / 3)
    line.set_data([x], [y])
    return (line,)

#　初期化関数とフレームごとの描画関数を用いて動画を作成
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(s_a_history), interval=200, repeat=False)

plt.show()
