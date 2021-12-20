from torch.nn.functional import one_hot, log_softmax, softmax, normalize
import pandas as pd
import numpy as np
import torch
import wandb
from hydra.utils import instantiate

from snake_reinforced.game import init_ple_env, ACTION2PLE_CODE
from snake_reinforced.agents import to_action_name
from snake_reinforced.vis import save_snake_video


def get_discounted_rewards(rewards: np.array, gamma: float) -> np.array:
    """
        Calculates the sequence of discounted rewards-to-go.
        Args:
            rewards: the sequence of observed rewards
            gamma: the discount factor
        Returns:
            discounted_rewards: the sequence of the rewards-to-go
    """
    discounted_rewards = np.empty_like(rewards, dtype=np.float32)
    for i in range(rewards.shape[0]):
        gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
        discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
        discounted_reward = np.sum(rewards[i:] * discounted_gammas)
        discounted_rewards[i] = discounted_reward
    return discounted_rewards


def play_episode(ple_env, agent, cfg):
    ple_env.reset_game()

    # initialize the episode arrays
    episode_actions = []
    episode_logits = []
    episode_rewards = []
    episode_state_values = []

    # episode loop
    while True:
        observation = ple_env.getScreenRGB()
        act_result = agent.act(observation)
        action_code = ACTION2PLE_CODE[to_action_name(act_result['action'])]

        reward = ple_env.act(action_code)

        episode_actions.append(act_result['action'])
        episode_logits.append(act_result['action_logits'])
        episode_rewards.append(reward)
        episode_state_values.append(act_result['value'])

        done = ple_env.game_over()
        # done = True
        # the episode is over
        if done:
            episode_rewards = np.array(episode_rewards)
            episode_state_values = torch.tensor(
                episode_state_values, dtype=torch.float32, device=cfg.device)
            episode_logits = torch.cat(episode_logits).to(
                torch.float32).to(cfg.device)
            episode_actions = torch.tensor(
                episode_actions, dtype=torch.long, device=cfg.device)

            # turn the rewards we accumulated during the episode into the rewards-to-go:
            # earlier actions are responsible for more rewards than the later taken actions
            discounted_rewards_to_go = get_discounted_rewards(rewards=episode_rewards,
                                                              gamma=cfg.training.gamma_discount)

            discounted_rewards_to_go = torch.tensor(
                discounted_rewards_to_go).float().to(cfg.device)

            discounted_rewards_to_go -= episode_state_values

            # calculate the sum of the rewards for the running average metric
            sum_of_rewards = np.sum(episode_rewards)

            # set the mask for the actions taken in the episode
            mask = one_hot(episode_actions, num_classes=len(ACTION2PLE_CODE))

            # calculate the log-probabilities of the taken actions
            # mask is needed to filter out log-probabilities of not related logits
            episode_log_probs = torch.sum(
                mask.float() * log_softmax(episode_logits, dim=1), dim=1)

            # weight the episode log-probabilities by the rewards-to-go
            episode_weighted_log_probs = episode_log_probs * \
                discounted_rewards_to_go

            # calculate the sum over trajectory of the weighted log-probabilities
            sum_weighted_log_probs = torch.sum(
                episode_weighted_log_probs).unsqueeze(dim=0)
            return {
                'sum_weighted_log_probs': sum_weighted_log_probs,
                'episode_logits': episode_logits,
                'sum_of_rewards': sum_of_rewards,
                'discounted_rewards_to_go': discounted_rewards_to_go,
            }


def calculate_loss(epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor, epoch_advantages: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
        Calculates the policy "loss" and the entropy bonus
        Args:
            epoch_logits: logits of the policy network we have collected over the epoch
            weighted_log_probs: loP * W of the actions taken
        Returns:
            policy loss
            entropy
    """
    policy_loss = -1 * torch.mean(weighted_log_probs)

    value_loss = torch.mean(torch.pow(epoch_advantages, 2))

    # add the entropy bonus
    p = softmax(epoch_logits, dim=1)
    log_p = log_softmax(epoch_logits, dim=1)
    entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
    entropy_bonus = entropy
    return policy_loss, entropy, value_loss


class PolicyGradientTrainer:
    def __init__(self, cfg, use_wandb=True):
        self.cfg = cfg

        self.ple_env = init_ple_env()
        self.agent = instantiate(cfg.agent)
        self.use_wandb = use_wandb

    def log(self, *args, **kwargs):
        if self.use_wandb:
            wandb.log(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def log_loss(self, loss_name, loss):
        self.log({loss_name: loss.detach().cpu().item()})

    def save_snake_video(self, epoch_i=None):
        filename = self.cfg.logging.video_filename.format(epoch_i=epoch_i)
        save_snake_video(self.ple_env, self.agent, filename,
                         n_last_frames=self.cfg.logging.n_last_frames)
        return filename

    def _call_epoch_callbacks(self, epoch_i):
        if (epoch_i + 1) % self.cfg.logging.log_video_every_n_epoch == 0:
            print('epoch_i={epoch_i}. Saving video..')
            video_path = self.save_snake_video(epoch_i)
            self.log({'video': video_path})

            if self.use_wandb:
                wandb.log({"video": wandb.Video(video_path, fps=30)})

    def train_epoch_step(self, epoch_logits, epoch_weighted_log_probs, epoch_advantages):
        policy_loss, entropy, value_loss = calculate_loss(epoch_logits=epoch_logits,
                                                          weighted_log_probs=epoch_weighted_log_probs,
                                                          epoch_advantages=epoch_advantages)

        total_loss = policy_loss + self.cfg.loss.entropy_weight * \
            entropy + self.cfg.loss.value_weight * value_loss

        self.log_loss("policy_loss", policy_loss)
        self.log_loss("entropy", entropy)
        self.log_loss("value_loss", value_loss)
        self.log_loss("total_loss", total_loss)

        return total_loss

    def train(self):
        if self.use_wandb:
            wandb.init(project=self.cfg.logging.project_name,
                       name=self.cfg.logging.run_name)

        optimizer = instantiate(self.cfg.optimizer, self.agent.parameters())

        epoch_i = 0
        while True:
            episodes = [play_episode(self.ple_env, self.agent, self.cfg)
                        for i in range(self.cfg.training.batch_size)]
            episodes = pd.DataFrame(episodes)

            epoch_logits = torch.cat(
                episodes['episode_logits'].values.tolist())
            epoch_weighted_log_probs = torch.cat(
                episodes['sum_weighted_log_probs'].values.tolist())
            epoch_advantages = torch.cat(
                episodes['discounted_rewards_to_go'].values.tolist())

            loss = self.train_epoch_step(
                epoch_logits, epoch_weighted_log_probs, epoch_advantages)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.log({'mean_reward': episodes['sum_of_rewards'].mean()})

            self._call_epoch_callbacks(epoch_i)
            epoch_i += 1
