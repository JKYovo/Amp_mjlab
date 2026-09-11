"""CPU regression tests: python -m unittest discover -s tests -v."""
import unittest

import torch

from rsl_rl.utils.motion_loader import AMPLoader


def make_loader(lengths):
    loader = AMPLoader.__new__(AMPLoader)
    loader._body_indexes = [0]
    for name, width, offset in (
        ('_body_pos_b_list', 3, 0), ('_body_ori_b_list', 6, 10000),
        ('_body_lin_vel_b_list', 3, 20000), ('_body_ang_vel_b_list', 3, 30000),
    ):
        setattr(loader, name, [
            (torch.arange(length, dtype=torch.float32) + clip * 100 + offset)
            .view(-1, 1, 1).expand(-1, 1, width).clone()
            for clip, length in enumerate(lengths)
        ])
    loader._build_sampling_cache()
    return loader


class MotionSamplingTest(unittest.TestCase):
    def test_all_24_clips_in_repeated_20_batch_updates(self):
        torch.manual_seed(42)
        lengths = [1 + 2 * i for i in range(24)]
        loader = make_loader(lengths)
        for _ in range(3):
            counts = torch.zeros(24, dtype=torch.long)
            for current, following in loader.feed_forward_generator(20, 4096):
                self.assertEqual(current.shape, (4096, 15))
                clips = (current[:, 0] // 100).long()
                frames = (current[:, 0] % 100).long()
                counts += torch.bincount(clips, minlength=24)
                torch.testing.assert_close(following[:, 0] // 100, current[:, 0] // 100)
                expected = torch.minimum(frames + 1, torch.tensor(lengths)[clips] - 1)
                torch.testing.assert_close(following[:, 0] % 100, expected.float())
                for start, stop, offset in ((0, 3, 0), (3, 9, 10000),
                                            (9, 12, 20000), (12, 15, 30000)):
                    torch.testing.assert_close(current[:, start:stop],
                                               (current[:, :1] + offset).expand(-1, stop-start))
            # Equal clip weighting, not duration weighting (lengths vary 1..47).
            self.assertTrue(torch.all((counts.float() / counts.sum() - 1/24).abs() < .004))

    def test_single_clip_uniform_frames_and_terminal_clamp(self):
        torch.manual_seed(7)
        loader = make_loader([7])
        current, following = next(loader.feed_forward_generator(1, 28000))
        counts = torch.bincount(current[:, 0].long(), minlength=7)
        self.assertTrue(torch.all((counts.float() / counts.sum() - 1/7).abs() < .01))
        torch.testing.assert_close(following[:, 0], (current[:, 0] + 1).clamp(max=6))

    def test_single_frame_and_seed_reproducibility(self):
        loader = make_loader([1])
        current, following = next(loader.feed_forward_generator(1, 16))
        torch.testing.assert_close(current, following)
        loader = make_loader([3, 7, 9])
        torch.manual_seed(19)
        a = next(loader.feed_forward_generator(1, 100))
        torch.manual_seed(19)
        b = next(loader.feed_forward_generator(1, 100))
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second)

    def test_empty_motion_rejected(self):
        with self.assertRaises(ValueError):
            make_loader([3, 0])


if __name__ == '__main__':
    unittest.main()
