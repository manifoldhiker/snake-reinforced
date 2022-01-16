from torch.nn.functional import one_hot, log_softmax, softmax, normalize
import pandas as pd
import numpy as np
import torch
import wandb
from datetime import datetime
from hydra.utils import instantiate

from snake_reinforced.game import init_ple_env, ACTION2PLE_CODE, get_snake_len
from snake_reinforced.agents import to_action_name
from snake_reinforced.vis import save_snake_video
from snake_reinforced.infrastructure import seed_all, save_state_dict
from snake_reinforced.utilities.grad import grad_norm, main_params


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

    max_episode_len = cfg.training.get('max_episode_len', 10_000)

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

        reward_config = cfg.get('reward', False)
        if reward_config:
            if reward > 0:
                reward *= reward_config.get('food_multiplier', 1)
            reward /= reward_config.get('divide_by', 1)

        episode_actions.append(act_result['action'])
        episode_logits.append(act_result['action_logits'])
        episode_rewards.append(reward)
        episode_state_values.append(act_result['value'])

        done = ple_env.game_over() or len(episode_actions) >= max_episode_len
        # done = True
        # the episode is over
        if done:
            episode_rewards = np.array(episode_rewards)

            episode_state_values = torch.cat(episode_state_values).flatten()
            assert episode_state_values.requires_grad

            episode_logits = torch.cat(episode_logits)

            episode_actions = torch.tensor(
                episode_actions, dtype=torch.long, device=cfg.device)

            # turn the rewards we accumulated during the episode into the rewards-to-go:
            # earlier actions are responsible for more rewards than the later taken actions
            discounted_rewards_to_go = get_discounted_rewards(rewards=episode_rewards,
                                                              gamma=cfg.training.gamma_discount)

            discounted_rewards_to_go = torch.tensor(
                discounted_rewards_to_go).float().to(cfg.device)

            baseline = episode_state_values

            discounted_rewards_to_go -= baseline

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
                discounted_rewards_to_go.detach()

            # calculate the sum over trajectory of the weighted log-probabilities
            sum_weighted_log_probs = torch.sum(
                episode_weighted_log_probs).unsqueeze(dim=0)
            return {
                'sum_weighted_log_probs': sum_weighted_log_probs,
                'episode_logits': episode_logits,
                'sum_of_rewards': sum_of_rewards,
                'discounted_rewards_to_go': discounted_rewards_to_go,
                'snake_len': get_snake_len(ple_env)
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

    assert epoch_logits.requires_grad
    assert weighted_log_probs.requires_grad
    assert epoch_advantages.requires_grad

    policy_loss = -1 * torch.mean(weighted_log_probs)

    value_loss = torch.mean(torch.pow(epoch_advantages, 2))

    # add the entropy bonus
    p = softmax(epoch_logits, dim=1)
    log_p = log_softmax(epoch_logits, dim=1)
    entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
    return policy_loss, entropy, value_loss


class PolicyGradientTrainer:
    def __init__(self, cfg, use_wandb=True, console_verbose=False):
        seed_all(cfg.seed)
        self.cfg = cfg

        self.ple_env = init_ple_env(
            init_length=cfg.get('init_snake_length', 3))
        self.agent = instantiate(cfg.agent)
        self.use_wandb = use_wandb
        self.console_verbose = console_verbose

    def log(self, *args, **kwargs):
        if self.use_wandb:
            wandb.log(*args, **kwargs)
            if self.console_verbose:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def save_snake_video(self, epoch_i=None):
        filename = self.cfg.logging.video_filename.format(epoch_i=epoch_i)
        save_snake_video(self.ple_env, self.agent, filename,
                         n_last_frames=self.cfg.logging.n_last_frames)
        return filename

    def save_agent(self, epoch_i=None):
        filename = self.cfg.logging.checkpoint_filename.format(epoch_i=epoch_i)
        save_state_dict(self.agent, filename)
        return filename

    def _call_epoch_callbacks(self, epoch_i):
        # print(self.agent.critic[0].weight[:5])
        if (epoch_i + 1) % self.cfg.logging.log_video_every_n_epoch == 0:
            print({'epoch': epoch_i, 'message': 'Saving video..'})
            video_path = self.save_snake_video(epoch_i)
            self.log({'video_path': video_path, 'epoch': epoch_i})

            if self.use_wandb:
                wandb.log({"video": wandb.Video(video_path, fps=30)})

        if (epoch_i + 1) % self.cfg.logging.checkpoint_every_n_epoch == 0:
            checkpoint_path = self.save_agent(epoch_i)
            print('Saved model to ',  checkpoint_path)

    def train_epoch_step(self,
                         epoch_logits,
                         epoch_weighted_log_probs,
                         epoch_advantages,
                         epoch_i):
        policy_loss, entropy, value_loss = calculate_loss(epoch_logits=epoch_logits,
                                                          weighted_log_probs=epoch_weighted_log_probs,
                                                          epoch_advantages=epoch_advantages)

        total_loss = policy_loss + self.cfg.loss.entropy_weight * \
            entropy + self.cfg.loss.value_weight * value_loss

        def log_loss(loss_name, loss):
            self.log({loss_name: loss.detach().cpu().item(), 'epoch': epoch_i})

        log_loss("policy_loss", policy_loss)
        log_loss("entropy", entropy)
        log_loss("value_loss", value_loss)
        log_loss("total_loss", total_loss)

        return total_loss

    def train(self):
        seed_all(self.cfg.seed)
        if self.use_wandb:
            wandb.init(project=self.cfg.logging.project_name,
                       name=self.cfg.logging.run_name,
                       config=self.cfg)

        optimizer = instantiate(self.cfg.optimizer, self.agent.parameters())

        epoch_i = 0
        while True:
            optimizer.zero_grad()
            episodes = []
            total_frames = 0
            print(
                f"{datetime.utcnow().strftime('%m_%d_%Y__%H_%M_%S')} Sampling episodes")
            for i in range(self.cfg.training.batch_size):
                frames = play_episode(self.ple_env, self.agent, self.cfg)
                episodes.append(frames)

                n_frames = len(frames["episode_logits"])
                total_frames += n_frames

                if total_frames > self.cfg.training.sampling_frame_limit:
                    print(f'Sampled {total_frames}, stopping for this epoch.')
                    break

            episodes = pd.DataFrame(episodes)

            epoch_logits = torch.cat(
                episodes['episode_logits'].values.tolist())
            epoch_weighted_log_probs = torch.cat(
                episodes['sum_weighted_log_probs'].values.tolist())
            epoch_advantages = torch.cat(
                episodes['discounted_rewards_to_go'].values.tolist())

            loss = self.train_epoch_step(
                epoch_logits, epoch_weighted_log_probs, epoch_advantages, epoch_i)

            loss.backward()

            if self.cfg.logging.get('track_grad_norm', True):
                grad_norm_dict = grad_norm(self.agent, norm_type=2)
            else:
                grad_norm_dict = {}

            if self.cfg.training.get('grad_clip', False):
                max_norm = self.cfg.training.get('grad_clip_max_norm', 2.0)

                params = main_params(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    params, max_norm=max_norm)

                # self.log(
                #     {'grad_clip_total_norm': total_norm.item(),
                #      'epoch': epoch_i})

            optimizer.step()

            self.log(
                {'mean_reward': episodes['sum_of_rewards'].mean(),
                 'mean_snake_len': episodes['snake_len'].mean(),
                 **grad_norm_dict,
                 'epoch': epoch_i})

            self._call_epoch_callbacks(epoch_i)
            epoch_i += 1
