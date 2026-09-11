"""Check that resuming does not reset the adaptive learning-rate scalar."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner


class ResumeTest(unittest.TestCase):
    def test_optimizer_resume_restores_scheduler_lr_only_when_requested(self):
        for resume_optimizer in (True, False):
            with self.subTest(resume_optimizer=resume_optimizer), tempfile.TemporaryDirectory() as directory:
                policy = torch.nn.Linear(2, 1)
                optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
                state = optimizer.state_dict()
                state['param_groups'][0]['lr'] = 0.00011390625
                checkpoint = Path(directory) / 'model.pt'
                torch.save({'model_state_dict': policy.state_dict(),
                            'optimizer_state_dict': state, 'iter': 79500,
                            'infos': {'test': True}}, checkpoint)
                runner = AmpOnPolicyRunner.__new__(AmpOnPolicyRunner)
                runner.alg = SimpleNamespace(policy=policy, optimizer=optimizer,
                                             learning_rate=1e-3, rnd=None)
                runner.empirical_normalization = False
                runner.current_learning_iteration = 0
                self.assertEqual(runner.load(checkpoint, load_optimizer=resume_optimizer), {'test': True})
                expected = 0.00011390625 if resume_optimizer else 1e-3
                self.assertEqual(runner.alg.learning_rate, expected)
                self.assertEqual(optimizer.param_groups[0]['lr'], expected)
                self.assertEqual(runner.current_learning_iteration, 79500)


if __name__ == '__main__':
    unittest.main()
