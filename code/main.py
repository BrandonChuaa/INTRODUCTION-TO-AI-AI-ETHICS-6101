import os
import numpy as np
from model import Agent
from utils import plot_learning_curve, make_env
LOAD = False
VERSION = '0'
if __name__ == '__main__':
    env = make_env('SpaceInvaders-v0')
    best_score = -np.inf
    n_games = 1000
    algo = 'DQN'
    agent = Agent(input_space_dim=(env.observation_space.shape),
                  memory_size=50000, batch_size=32, action_num=6, gamma=0.99, epsilon_min=0.1,
                  epsilon_decay=5e-6, replace_count=5000, lr=1e-4, batch_norm=True, save=True, algo=algo)
    if LOAD:
        agent.load_models()

    fname = algo + '_SpaceInvaders-v0_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games' + VERSION
    fname = fname + '.png'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    figure_file = os.path.join(cur_dir, '.\\plots\\', fname)

    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    # Training loop
    for i in range(n_games):
        final = False
        observation = env.reset()

        score = 0
        while not final:
            action = agent.choose_action(observation)
            next_obs, reward, final, info = env.step(action)
            score += reward

            if not LOAD:
                agent.store_transition(state=observation, action=action,
                                       reward=reward, next_state=next_obs, final=bool(final))
                agent.learn()
            observation = next_obs
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            # if not LOAD:
            # agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if LOAD and n_steps >= 18000:
            break
    agent.save_models()
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
