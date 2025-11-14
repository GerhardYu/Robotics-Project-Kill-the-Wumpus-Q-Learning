import json
import random
import matplotlib.pyplot as plt

from env import WumpusEnv, CAVE


def q_learn(env,
            episodes=10000,
            alpha=0.1,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05):
    """
    Tabular Q-learning.
    Q[(state, action)] -> value

      state = (room, arrows, wumpus_alive, smell, rustle, breeze)
      action in [0..5]
    """
    Q = {}
    actions = list(range(6))

    def get_q(s, a):
        return Q.get((s, a), 0.0)

    def best_action(s):
        qs = [get_q(s, a) for a in actions]
        max_q = max(qs)
        # Break ties randomly
        candidates = [a for a, q in zip(actions, qs) if q == max_q]
        return random.choice(candidates)

    episode_rewards = []
    episode_wins = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        # Linear epsilon decay
        frac = ep / max(1, episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        while not done:
            # epsilon-greedy
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = best_action(state)

            next_state, reward, done, _info = env.step(action)

            old_q = get_q(state, action)
            max_next = max(get_q(next_state, a) for a in actions)
            target = reward + (0.0 if done else gamma * max_next)
            new_q = old_q + alpha * (target - old_q)
            Q[(state, action)] = new_q

            state = next_state
            total_reward += reward

        # Episode finished
        episode_rewards.append(total_reward)
        episode_wins.append(1 if env.win else 0)

        # Console progress
        if (ep + 1) % 500 == 0:
            recent = 500
            r_slice = episode_rewards[-recent:]
            w_slice = episode_wins[-recent:]
            avg_r = sum(r_slice) / len(r_slice)
            avg_w = sum(w_slice) / len(w_slice)
            print(
                f"Episode {ep+1}/{episodes} | "
                f"avg reward(last {recent}): {avg_r:.3f} | "
                f"win rate(last {recent}): {avg_w*100:.1f}%"
            )

    return Q, episode_rewards, episode_wins


def save_q_table(Q, path="q_table.json"):
    """
    Convert Q[(state, action)] -> JSON-friendly dict:

      {
        "room,arrows,wAlive,smell,rustle,breeze": {
          "action": q_value,
          ...
        },
        ...
      }
    """
    store = {}
    for (state, action), q in Q.items():
        room, arrows, w_alive, smell, rustle, breeze = state
        key_state = f"{room},{arrows},{w_alive},{smell},{rustle},{breeze}"
        if key_state not in store:
            store[key_state] = {}
        store[key_state][str(action)] = q

    with open(path, "w") as f:
        json.dump(store, f, indent=2)
    print(f"Saved Q-table to {path}")


def moving_average(data, window):
    if window <= 1:
        return data[:]
    out = []
    cumsum = 0.0
    for i, v in enumerate(data):
        cumsum += v
        if i >= window:
            cumsum -= data[i - window]
            out.append(cumsum / window)
        else:
            out.append(cumsum / (i + 1))
    return out


def plot_training(rewards, wins, window=100, out_prefix="training"):
    episodes = list(range(1, len(rewards) + 1))
    ma_rewards = moving_average(rewards, window)
    ma_wins = moving_average(wins, window)

    # ----- Reward plot -----
    plt.figure()
    plt.plot(episodes, rewards, alpha=0.3)
    plt.plot(episodes, ma_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title(f"Episode Reward (moving avg window={window})")
    plt.tight_layout()
    reward_path = f"{out_prefix}_reward.png"
    plt.savefig(reward_path)
    plt.close()
    print(f"Saved reward plot to {reward_path}")

    # ----- Win-rate plot -----
    plt.figure()
    plt.plot(episodes, ma_wins)
    plt.xlabel("Episode")
    plt.ylabel("Win rate (moving avg)")
    plt.title(f"Win Rate (moving avg window={window})")
    plt.tight_layout()
    win_path = f"{out_prefix}_winrate.png"
    plt.savefig(win_path)
    plt.close()
    print(f"Saved win-rate plot to {win_path}")


if __name__ == "__main__":
    env = WumpusEnv(CAVE, seed=None)

    print("Training Q-learning agent...")
    Q, rewards, wins = q_learn(
        env,
        episodes=5000,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
    )

    save_q_table(Q, "q_table.json")
    plot_training(rewards, wins, window=100, out_prefix="training")
