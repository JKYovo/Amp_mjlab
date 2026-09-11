"""Regression tests for transactional AMP-PPO non-finite update handling."""

import unittest

import torch

from rsl_rl.algorithms.amp_ppo import AMPPPO


class _Policy(torch.nn.Module):
    is_recurrent = False

    def __init__(self, *, nan_entropy: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.nan_entropy = nan_entropy

    def act(self, obs, **_kwargs):
        self.action_mean = obs[:, :1] * self.weight
        self.action_std = torch.ones_like(self.action_mean)
        entropy = self.weight * 0.0 + 1.0
        if self.nan_entropy:
            entropy = entropy * torch.tensor(float("nan"))
        self.entropy = entropy.expand_as(self.action_mean)
        return self.action_mean

    def get_actions_log_prob(self, actions):
        return -torch.square(actions - self.action_mean).sum(dim=-1)

    def evaluate(self, obs, **_kwargs):
        return obs[:, :1] * self.weight


class _Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(4, 2)
        self.amp_linear = torch.nn.Linear(2, 1)

    def forward(self, value):
        return self.amp_linear(torch.tanh(self.trunk(value)))

    def compute_grad_pen(self, *_args, **_kwargs):
        return self.amp_linear.weight.square().mean() * 0.0


class _Storage:
    def __init__(self, sample):
        self.sample = sample
        self.num_envs = 4
        self.num_transitions_per_env = 1
        self.cleared = False

    def mini_batch_generator(self, *_args):
        yield self.sample

    def clear(self):
        self.cleared = True


class _AmpSource:
    def __init__(self, sample):
        self.sample = sample

    def feed_forward_generator(self, *_args):
        yield self.sample


def _algorithm(*, nan_entropy: bool) -> AMPPPO:
    batch = 4
    obs = torch.zeros(batch, 1)
    critic_obs = torch.ones(batch, 1)
    actions = torch.zeros(batch, 1)
    sample = (
        obs,
        critic_obs,
        actions,
        torch.zeros(batch, 1),
        torch.ones(batch, 1),
        torch.full((batch, 1), 0.5),
        torch.zeros(batch, 1),
        torch.full((batch, 1), 0.01),
        torch.ones(batch, 1),
        (None, None),
        None,
        None,
    )
    amp_state = torch.zeros(batch, 2)

    algorithm = AMPPPO.__new__(AMPPPO)
    algorithm.policy = _Policy(nan_entropy=nan_entropy)
    algorithm.discriminator = _Discriminator()
    algorithm.storage = _Storage(sample)
    algorithm.amp_storage = _AmpSource((amp_state, amp_state))
    algorithm.amp_data = _AmpSource((amp_state, amp_state))
    algorithm.amp_normalizer = None
    algorithm.optimizer = torch.optim.Adam(
        list(algorithm.policy.parameters()) + list(algorithm.discriminator.parameters()),
        lr=1.0e-3,
    )
    algorithm.rnd = None
    algorithm.rnd_optimizer = None
    algorithm.symmetry = None
    algorithm.normalize_advantage_per_mini_batch = False
    algorithm.num_learning_epochs = 1
    algorithm.num_mini_batches = 1
    algorithm.desired_kl = 0.01
    algorithm.schedule = "adaptive"
    algorithm.learning_rate = 1.0e-3
    algorithm.is_multi_gpu = False
    algorithm.gpu_global_rank = 0
    algorithm.clip_param = 0.2
    algorithm.use_clipped_value_loss = False
    algorithm.value_loss_coef = 1.0
    algorithm.entropy_coef = 0.01
    algorithm.amploss_coef = 1.0
    algorithm.max_grad_norm = 1.0
    algorithm.min_std = None
    algorithm.device = "cpu"
    return algorithm


class NonFiniteUpdateTest(unittest.TestCase):
    def test_finite_batch_commits_candidate_learning_rate_and_weights(self):
        algorithm = _algorithm(nan_entropy=False)
        old_weight = algorithm.policy.weight.detach().clone()

        losses = algorithm.update()

        self.assertEqual(losses["skipped_non_finite_batches"], 0.0)
        self.assertAlmostEqual(algorithm.learning_rate, 1.5e-3)
        self.assertAlmostEqual(algorithm.optimizer.param_groups[0]["lr"], 1.5e-3)
        self.assertFalse(torch.equal(algorithm.policy.weight, old_weight))
        self.assertTrue(algorithm.storage.cleared)

    def test_nonfinite_batch_does_not_commit_candidate_or_parameters(self):
        algorithm = _algorithm(nan_entropy=True)
        old_parameters = [param.detach().clone() for param in algorithm.optimizer.param_groups[0]["params"]]

        losses = algorithm.update()

        self.assertEqual(losses["skipped_non_finite_batches"], 1.0)
        self.assertEqual(algorithm.learning_rate, 1.0e-3)
        self.assertEqual(algorithm.optimizer.param_groups[0]["lr"], 1.0e-3)
        self.assertEqual(algorithm.optimizer.state, {})
        for current, old in zip(algorithm.optimizer.param_groups[0]["params"], old_parameters):
            torch.testing.assert_close(current, old)
        self.assertTrue(algorithm.storage.cleared)


if __name__ == "__main__":
    unittest.main()
