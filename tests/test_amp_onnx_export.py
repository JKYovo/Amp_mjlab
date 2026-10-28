"""Automatic export must not freeze normalization or mutate the PPO policy."""
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.normalizer import EmpiricalNormalization
from src.tasks.amp_loco.rl.runner import AMPOnPolicyRunner, _OnnxPolicyWrapper


class ExportIsolationTest(unittest.TestCase):
    def make_runner(self, device='cpu', normalized=True):
        torch.manual_seed(42)
        with redirect_stdout(io.StringIO()):
            policy=ActorCritic(12,15,3,actor_hidden_dims=[16,8],critic_hidden_dims=[16,8])
        policy.to(device).train()
        # The complete policy cannot safely be deepcopied once this cached
        # distribution contains non-leaf tensors. Copy only the actor MLP.
        policy.act(torch.randn(7,12,device=device))
        normalizer=EmpiricalNormalization([12],until=100000000).to(device).train()
        normalizer(torch.randn(7,12,device=device))
        critic_normalizer=EmpiricalNormalization([15],until=100000000).to(device).train()
        runner=AMPOnPolicyRunner.__new__(AMPOnPolicyRunner)
        runner.alg=SimpleNamespace(policy=policy)
        runner.obs_normalizer=normalizer
        runner.privileged_obs_normalizer=critic_normalizer
        runner.empirical_normalization=normalized
        runner.device=device
        return runner

    def snapshot(self, runner):
        modules=(runner.alg.policy,runner.obs_normalizer,runner.privileged_obs_normalizer)
        return [dict(modes={k:v.training for k,v in m.named_modules()},
                     state={k:v.detach().clone() for k,v in m.state_dict().items()},
                     devices={k:str(v.device) for k,v in m.state_dict().items()},
                     identities={k:id(v) for k,v in m.named_parameters()}) for m in modules]

    def assert_unchanged(self, runner, before):
        after=self.snapshot(runner)
        for old,new in zip(before,after):
            for key in ('modes','devices','identities'):
                self.assertEqual(old[key],new[key])
            self.assertEqual(old['state'].keys(),new['state'].keys())
            for name,value in old['state'].items():
                torch.testing.assert_close(value,new['state'][name],rtol=0.,atol=0.)

    def exercise_mocked_export(self, device='cpu', normalized=True, fail=False):
        runner=self.make_runner(device,normalized)
        # Preserve mixed child modes rather than blindly calling train() on
        # the entire live model after exporting it.
        runner.alg.policy.critic.eval()
        runner.obs_normalizer.train()
        before=self.snapshot(runner)

        def exporter(wrapper,*args,**kwargs):
            self.assertFalse(wrapper.training)
            self.assertIsNot(wrapper.actor,runner.alg.policy.actor)
            self.assertTrue(all(p.device.type=='cpu' for p in wrapper.parameters()))
            if normalized:
                self.assertIsNot(wrapper.obs_normalizer,runner.obs_normalizer)
                self.assertFalse(wrapper.obs_normalizer.training)
            else:
                self.assertIsNone(wrapper.obs_normalizer)
            self.assert_unchanged(runner,before)
            if fail:
                raise RuntimeError('injected export failure')

        with TemporaryDirectory() as directory, patch('torch.onnx.export',side_effect=exporter), \
                patch('src.tasks.amp_loco.rl.runner._inline_external_onnx_data'):
            if fail:
                with self.assertRaisesRegex(RuntimeError,'injected export failure'):
                    runner._export_policy_to_onnx(directory)
            else:
                runner._export_policy_to_onnx(directory)
                runner._export_policy_to_onnx(directory)
        self.assert_unchanged(runner,before)
        count=int(runner.obs_normalizer.count)
        runner.obs_normalizer(torch.randn(9,12,device=device)+3.)
        self.assertEqual(int(runner.obs_normalizer.count),count+9)
        # Optimizer parameter identity and gradients still work after export.
        loss=runner.alg.policy.act_inference(torch.randn(9,12,device=device)).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in runner.alg.policy.actor.parameters()))

    def test_wrapper_snapshot_has_no_shared_parameter_or_buffer_storage(self):
        runner=self.make_runner()
        wrapper=_OnnxPolicyWrapper(runner.alg.policy,runner.obs_normalizer).cpu().eval()
        for source,copy in ((runner.alg.policy.actor,wrapper.actor),
                            (runner.obs_normalizer,wrapper.obs_normalizer)):
            for name,value in source.state_dict().items():
                other=copy.state_dict()[name]
                self.assertNotEqual(value.data_ptr(),other.data_ptr())
                torch.testing.assert_close(value,other,rtol=0.,atol=0.)
        self.assertFalse(hasattr(wrapper,'critic'))
        self.assertTrue(runner.alg.policy.training)
        self.assertTrue(runner.obs_normalizer.training)

    def test_success_keeps_stats_updating_and_mixed_modes_unchanged(self):
        self.exercise_mocked_export()

    def test_export_failure_leaves_live_training_state_intact(self):
        self.exercise_mocked_export(fail=True)

    def test_no_empirical_normalization_success(self):
        self.exercise_mocked_export(normalized=False)

    def test_no_empirical_normalization_failure(self):
        self.exercise_mocked_export(normalized=False,fail=True)

    def test_existing_eval_mode_stays_eval(self):
        runner=self.make_runner()
        runner.alg.policy.eval()
        runner.obs_normalizer.eval()
        before=self.snapshot(runner)
        with TemporaryDirectory() as directory, patch('torch.onnx.export'), \
                patch('src.tasks.amp_loco.rl.runner._inline_external_onnx_data'):
            runner._export_policy_to_onnx(directory)
        self.assert_unchanged(runner,before)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_success_and_failure_preserve_devices_and_optimizer_parameters(self):
        for normalized in (False,True):
            for fail in (False,True):
                self.exercise_mocked_export('cuda:0',normalized,fail)

    def test_real_single_file_onnx_matches_live_policy_and_keeps_stats_updating(self):
        runner=self.make_runner()
        before=self.snapshot(runner)
        with TemporaryDirectory() as directory:
            runner._export_policy_to_onnx(directory)
            path=Path(directory)/'policy.onnx'
            self.assertTrue(path.is_file())
            self.assertFalse(Path(str(path)+'.data').exists())
            model=onnx.load(path,load_external_data=False)
            self.assertFalse(any(t.data_location==onnx.TensorProto.EXTERNAL for t in model.graph.initializer))
            onnx.checker.check_model(model,full_check=True)
            options=ort.SessionOptions()
            options.intra_op_num_threads=2
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            for batch in (1,7,20):
                obs=torch.randn(batch,12)
                norm=runner.obs_normalizer
                # Frozen inference formula; do not update the live statistics
                # while measuring equivalence to the exported snapshot.
                with torch.no_grad():
                    expected=runner.alg.policy.act_inference((obs-norm._mean)/(norm._std+norm.eps)).numpy()
                actual=session.run(None,{'obs':obs.numpy()})[0]
                np.testing.assert_allclose(actual,expected,rtol=1.e-5,atol=1.e-6)
        self.assert_unchanged(runner,before)
        previous=int(runner.obs_normalizer.count)
        runner.obs_normalizer(torch.randn(4,12))
        self.assertEqual(int(runner.obs_normalizer.count),previous+4)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
