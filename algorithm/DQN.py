from copy import deepcopy
import torch
import numpy as np

class DQNPolicy(object):
    """Implementation of Deep Q Network. arXiv:1312.5602.
    Implementation of Double Q-Learning. arXiv:1509.06461.
    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).
    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double dqn. Default to True.
    :param bool clip_loss_grad: clip the gradient of the loss in accordance
        with nature14236; this amounts to using the Huber loss instead of
        the MSE loss. Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False) -> None:
        # init global parameters
        self.model = model
        self.optim = optim
        self.eps = 0.0
        self._gamma = discount_factor
        self._n_step = estimation_step
        self._target = target_update_freq > 0 
        self._freq_upate = target_update_freq
        self._iter_number = 0
        if self._target:
            self.memory_old = deepcopy(self.model)
            self.memory_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad
    