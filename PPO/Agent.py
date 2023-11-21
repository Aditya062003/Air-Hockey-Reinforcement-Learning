import numpy as np
import torch
import pickle
from enum import Enum
from PPO import PPO

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.final_states = []
        self.lengths = []
        self.winners = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.final_states[:]
        del self.lengths[:]
        del self.winners[:]

class Mode(Enum):
    TRAINING = 1
    EVALUATION = 2


class OpponentType(Enum):
    DEFENDING = 1
    SHOOTING = 2
    NORMAL = 3


class Agent:
    def __init__(self, agent_register=None):
        self._mode = Mode.TRAINING
        self._current_frame_skip_pos = 0
        self._current_frame_skip_action = None
        self._current_frame_skip_reward_true = 0
        self._current_frame_skip_reward_calc = 0
        self._current_ep_timestep_counter = 0
        self._current_reward_calc = 0
        self._current_reward_true = 0
        self._current_opp_type = 1
        self._current_opp_weak = False
        self._current_frame_skip_activated = False
        self._current_closeness_puck = 0
        self._current_touch_puck = 0
        self._current_puck_direction = 0

        self._timesteps_since_update = 0

        self._memory = Memory()
        self._memory.lengths.append(0)

        self._agent_register = agent_register

    def configure(self, episode_max_steps, filename, save_episodes,update_timesteps):
        
        self._config = {
            'episode_max_steps': episode_max_steps,
            'filename': filename,
            'save_episodes': save_episodes,
            'update_timesteps': update_timesteps
        }

        self._stats = {
            'global_timestep_counter': 0,
            'ep_counter': 1,
            'learning_ep_counter': 1,

            'ep_last_training': 1,
            'ep_last_switch_to_evaluation': 0,

            'episode_mode': [],
            'ep_rewards_calc': [],
            'ep_rewards_true': [],
            'ep_closeness_puck': [],
            'ep_touch_puck': [],
            'ep_puck_direction': [],
            'ep_length':  [],
            'ep_wins':  [],
            'ep_eval_results': [[], [], []],
            'ep_opp_weak': [],
            'ep_opp_type': [],
            'ep_frame_skip': []
        }

        self._configure_ppo()

    def _configure_ppo(self):
        self.ppo = PPO()

    def act(self, state):
        if self._mode == Mode.TRAINING:
            if self._current_frame_skip_pos == 0:
                self._current_frame_skip_pos = (self._config["frame_skipping_length"] if self._current_frame_skip_activated else 1)
                action = self.ppo.act(state, self._memory)
                self._current_frame_skip_action = action
            else:
                action = self._current_frame_skip_action

            self._current_frame_skip_pos -= 1
        else:
            action = self.ppo.act(state)

        return action

    def feedback(self, reward, info, done, state):
        reward_calc = self._calculate_reward(reward, info, done)

        if (self._current_frame_skip_pos > 0 and not done and not self._mode == Mode.EVALUATION):
            self._current_frame_skip_reward_calc += reward
            self._current_frame_skip_reward_true += reward_calc
            return

        elif ((self._current_frame_skip_pos == 0 or done) and not self._mode == Mode.EVALUATION):
            reward_calc = self._current_frame_skip_reward_calc + reward_calc
            reward = self._current_frame_skip_reward_true + reward

            self._current_frame_skip_reward_calc = 0
            self._current_frame_skip_reward_true = 0

        if (self._mode == Mode.TRAINING and (self._current_frame_skip_pos == 0 or done)):
            self._memory.rewards.append(reward_calc)
            self._memory.is_terminals.append(done)
            self._timesteps_since_update += 1

        self._current_ep_timestep_counter += 1
        self._stats["global_timestep_counter"] += 1
        self._current_reward_calc += reward_calc
        self._current_reward_true += reward
        self._current_closeness_puck += info["reward_closeness_to_puck"]
        self._current_touch_puck += info["reward_touch_puck"]
        self._current_puck_direction += info["reward_puck_direction"]
        self._memory.lengths[-1] += 1

        if done or (self._current_ep_timestep_counter % self._config["episode_max_steps"] == 0):
            self._stats["ep_counter"] += 1
            if self._mode == Mode.TRAINING:
                self._stats["learning_ep_counter"] += 1
            self._stats["episode_mode"].append(self._mode.value)
            self._stats["ep_rewards_calc"].append(self._current_reward_calc)
            self._stats["ep_rewards_true"].append(self._current_reward_true)
            self._stats["ep_closeness_puck"].append(self._current_closeness_puck)
            self._stats["ep_touch_puck"].append(self._current_touch_puck)
            self._stats["ep_puck_direction"].append(self._current_puck_direction)
            self._stats["ep_length"].append(self._current_ep_timestep_counter)
            self._stats["ep_wins"].append(info["winner"])
            self._stats["ep_opp_type"].append(self._current_opp_type)
            self._stats["ep_opp_weak"].append(self._current_opp_weak)

            doc_frame_skip = 1
            if (self._mode == Mode.EVALUATION and self._current_frame_skip_activated):
                doc_frame_skip = self._stats["frame_skipping_length"]
            self._stats["ep_frame_skip"].append(doc_frame_skip)

            self._memory.winners.append(info["winner"])
            self._memory.final_states.append(state.reshape(-1))

            # Prepare next expisode: Reset temporary statistics
            self._current_frame_skip_pos = 0
            self._current_reward_calc = 0
            self._current_reward_true = 0
            self._current_ep_timestep_counter = 0
            self.current_closeness_to_puck = 0
            self._current_touch_puck = 0
            self._current_puck_direction = 0

            # Prepare next episode: Set next params
            self._memory.lengths.append(0)

            if self._stats["ep_counter"] % self._config["save_episodes"] == 0:
                self.save()

            if (self._mode == Mode.TRAINING and self._timesteps_since_update >= self._config["update_timesteps"]):
                self._update()

    def _calculate_reward(self, reward, info, done):
        if self._mode == Mode.EVALUATION:
            return reward
        elif done:
            value = reward
            return value
        else:
            return 0

    def change_mode(self):
        if self._current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")

        if self._mode == Mode.TRAINING:
            if len(self._memory.actions) > 0:
                self._update()

            self.frame_skipping_activated = False
            self._mode = Mode.EVALUATION
            self._stats["ep_last_switch_to_evaluation"] = self._stats["ep_counter"]

        elif self._mode == Mode.EVALUATION:
            self._compute_statistics()

            self._mode = Mode.TRAINING
            self._stats["ep_last_training"] = self._stats["ep_counter"]

    def _compute_statistics(self):

        train_start = self._stats["ep_last_training"]-1
        train_end = self._stats["ep_last_switch_to_evaluation"]-1
        eval_start = self._stats["ep_last_switch_to_evaluation"]-1
        eval_end = len(self._stats["ep_rewards_calc"]) 

        train_rewards_arr = np.asarray(self._stats["ep_rewards_true"])[
            train_start:train_end]
        eval_rewards_arr = np.asarray(self._stats["ep_rewards_true"])[
            eval_start:eval_end]

        training_matches = np.asarray(self._stats["ep_wins"])[
            train_start:train_end]
        eval_matches = np.asarray(self._stats["ep_wins"])[
            eval_start:eval_end]

        avg_train_rewards = (np.sum(train_rewards_arr) / len(train_rewards_arr))
        avg_eval_rewards = (np.sum(eval_rewards_arr) / len(eval_rewards_arr))

        train_wins, train_lost, train_draws = self._calculateWLDRates(training_matches, train_start, train_end)

        eval_wins, eval_lost, eval_draws = self._calculateWLDRates(eval_matches, eval_start, eval_end)

        hist_index = self._current_opp_type.value-1

        save_note = ""
        if len(self._stats["ep_eval_results"][hist_index]) == 0:
            past_max_values = [-1000000]
        else:
            past_max_values = np.max(np.asarray(self._stats["ep_eval_results"][hist_index]))

        if past_max_values < avg_eval_rewards:
            save_note = " - Checkpoint saved"
            self.save("best")
            if (self._agent_register is not None and len(self._stats["ep_eval_results"][hist_index]) > 0 and avg_eval_rewards >= 4):
                self._agent_register.add_agent(self._config["filename"], "best")
                save_note = save_note + " - added"

            self._stats["ep_eval_results"][hist_index].append(avg_eval_rewards)

        print(("{}: ## Learn(R;W,D,L in %): {:.1f}, {:.0f}, {:.0f}, {:.0f} " + "Eval(R;W,D,L in %): {:.1f}, {:.0f}, {:.0f}, {:.0f} {}").format(self._stats["ep_counter"]-1,avg_train_rewards, train_wins, train_draws, train_lost,avg_eval_rewards, eval_wins, eval_draws, eval_lost,save_note))

    def _calculateWLDRates(cls, matches, start, end):
        count = float(end-start)

        wins = float(np.sum(np.where(matches > 0, 1, 0)))
        lost = float(np.sum(np.where(matches < 0, 1, 0)))
        draws = float(np.sum(np.where(matches == 0, 1, 0)))

        return (wins / count * 100.0, lost / count * 100.0, draws / count * 100.0)

    def _update(self):
        if self._memory.lengths[-1] == 0:
            del self._memory.lengths[-1]
        self.ppo.update(self._memory)
        self._memory.clear_memory()
        self._memory.lengths.append(0)
        self._timesteps_since_update = 0

    def save(self, info=""):
        filename = "checkpoints/Hockey-v0_{}_{}.pth".format(self._config["filename"], info)

        torch.save({
            'policy': self.ppo.policy.state_dict(),
            'policy_old': self.ppo.policy_old.state_dict(),
            'optimizer': self.ppo.optimizer.state_dict(),
            'configuration': self._config,
            'statistics': self._stats,
            'input_normalizer': pickle.dumps(self.ppo.input_normalizer),
            'memory': self._memory,
        }, filename)

    def load(self, filename, info=""):
        filename = "checkpoints/Hockey-v0_{}_{}.pth".format(
            filename, info)
        checkpoint = torch.load(filename)

        self._config = checkpoint["configuration"]
        self._stats = checkpoint["statistics"]
        self._stats = checkpoint["statistics"]

        self._configure_ppo()
        self._memory = checkpoint["memory"]

        self.ppo.policy.load_state_dict(checkpoint["policy"])
        self.ppo.policy_old.load_state_dict(checkpoint["policy_old"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer"])
        self.ppo.input_normalizer = pickle.loads(
            checkpoint["input_normalizer"])

    @property
    def config(self):
        return self._config

    @property
    def stats(self):
        return self._stats

    @property
    def mode(self):
        return self._mode

    @property
    def opp_type(self):
        return self._current_opp_type

    @opp_type.setter
    def opp_type(self, value):
        if self._current_ep_timestep_counter != 0:
            raise Exception("Can't switch opponent_type during episode.")
        self._current_opp_type = value

    @property
    def opp_weak(self):
        return self._current_opp_weak

    @opp_weak.setter
    def opp_weak(self, value):
        if self._current_ep_timestep_counter != 0:
            raise Exception(("Can't switch opponent weakness " + "during a running episode."))
        self._current_opp_weak = value

    @property
    def filename(self):
        return self._config["filename"]

    @filename.setter
    def filename(self, value):
        self._config["filename"]

    @property
    def frame_skipping_activated(self):
        return self._current_frame_skip_activated

    @frame_skipping_activated.setter
    def frame_skipping_activated(self, value):
        if self._mode == Mode.EVALUATION:
            raise Exception("Can't be activated during evaluation")
        if self._current_ep_timestep_counter != 0:
            raise Exception(("Can't switch frame skipping skipping activation " + "during a running episode."))
        self._current_frame_skip_activated = value

    @property
    def frame_skipping_length(self):
        return self._current_frame_skip_activated

    @frame_skipping_length.setter
    def frame_skipping_length(self, value):
        if self._mode == Mode.EVALUATION:
            raise Exception("Can't be activated during evaluation")
        if self._current_ep_timestep_counter != 0:
            raise Exception(("Can't switch frame skipping length during " + "a running episode."))
        self._config["frame_skipping_length"] = value
