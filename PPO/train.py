import numpy as np
import random
import torch
import gym
import laserhockey.hockey_env as lh
from Agent import Agent
from AgentRegister import AgentRegister


def train(training_episodes,training_length,eval_length,save_episodes,episode_max_steps=1000, update_timesteps=20000,filename="", seed=None,load_filename=None, load_info="best"):

    max_episodes = int(np.round(training_episodes * (1 + eval_length / training_length), 0))
    frame_skip_frequency = 1
    frame_skipping_length = 1
    env = lh.HockeyEnv()
    if seed is not None:
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    ar = AgentRegister(True, True, 3)
    agent = Agent(ar)

    if load_filename is not None:
        print("Loaded")
        agent.load(load_filename, load_info)
    else:
        agent.configure(episode_max_steps, filename, save_episodes,update_timesteps)

    agent.opp_type = 3
    agent.filename = filename
    agent.frame_skipping_activated = False
    agent.frame_skipping_length = frame_skipping_length

    print(agent.config, "\n")

    current_frame_skip_frequency = frame_skip_frequency

    for i_episode in range(1, max_episodes+1):
        while True:
            player2, basic_agent, weak = ar.sample_agent(agent.mode)
            agent.opp_weak = weak
            if player2 is not None:
                break

        state = env.reset()
        state2 = env.obs_agent_two()

        if frame_skip_frequency is not None and agent.mode == 1:
            if current_frame_skip_frequency == 1:
                agent.frame_skipping_activated = True
            elif current_frame_skip_frequency == 0:
                current_frame_skip_frequency = frame_skip_frequency
                agent.frame_skipping_activated = False
            current_frame_skip_frequency -= 1

        for _ in range(episode_max_steps):
            a1 = agent.act(state)
            a2 = player2.act(state2)
            state, reward, done, info = env.step(np.hstack((a1, a2)))
            agent.feedback(reward, info, done, state)
            state2 = env.obs_agent_two()
            env.render()
            if done:
                break

        if (i_episode) % (training_length + eval_length) == training_length:
            agent.change_mode(False)
        elif (i_episode) % (training_length + eval_length) == 0:
            agent.change_mode(True)