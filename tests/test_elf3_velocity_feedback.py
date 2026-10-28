import math
import unittest
import torch
from src.tasks.amp_loco.mdp.velocity_feedback_model import JointVelocityFeedback, tustin_lowpass_coefficients


class FeedbackTest(unittest.TestCase):
    def test_cutoff_and_dc_gain(self):
        b0,b1,a1 = tustin_lowpass_coefficients(20.,.005)
        z = complex(math.cos(-2*math.pi*20*.005),math.sin(-2*math.pi*20*.005))
        h = (b0+b1*z)/(1+a1*z)
        self.assertAlmostEqual(abs(h),math.sqrt(.5),places=12)
        self.assertAlmostEqual(math.atan2(h.imag,h.real),-math.pi/4,places=12)
        self.assertAlmostEqual((b0+b1)/(1+a1),1.,places=12)

    def test_sequence_reset_and_reads(self):
        raw = torch.zeros(3,29)
        f = JointVelocityFeedback(raw,.005,mode='lpf20')
        b0,b1,a1 = tustin_lowpass_coefficients(20.,.005)
        old_x, old_y = raw.clone(),raw.clone()
        for i in range(40):
            x = torch.full_like(raw,float(i%5))
            expected = b0*x+b1*old_x-a1*old_y
            torch.testing.assert_close(f.update(x),expected)
            old_x,old_y=x.clone(),expected.clone()
        before=f.value.clone()
        for _ in range(5): torch.testing.assert_close(f.value,before)
        self.assertEqual(f.update_count,40)
        reset_raw=torch.full_like(raw,7.)
        f.reset(reset_raw,torch.tensor([1]))
        torch.testing.assert_close(f.value[1],reset_raw[1])
        torch.testing.assert_close(f.value[[0,2]],before[[0,2]])
        torch.testing.assert_close(reset_raw,torch.full_like(raw,7.))


if __name__ == '__main__': unittest.main()
