"""Guard against geometry-only DWAQ ports and re-enabled reward experiments."""
import hashlib
import json
from pathlib import Path
import re
import unittest

import numpy as np
import mujoco
from mjlab.entity import Entity
from scripts.check_elf3_dwaq_asset import verify_compiled_robot, verify_original_amp_scope
from src.assets.robots.elf3.elf3_constants import (
    ELF3_ACTION_SCALE, ELF3_ACTUATORS, ELF3_DWAQ_CONTRACT,
    ELF3_DWAQ_CONTRACT_PATH, ELF3_HOME_KEYFRAME, ELF3_XML, get_elf3_robot_cfg,
)


class DwaqAssetTest(unittest.TestCase):
    def test_contract_is_byte_identical_to_archived_dwaq_a14f9d5(self):
        self.assertEqual(hashlib.sha256(ELF3_DWAQ_CONTRACT_PATH.read_bytes()).hexdigest(),
                         '90e4b49b10ff160002c09d5c735b51f8d46e9c710cd2e2c44257fe7edb320919')

    def test_mesh_and_canonical_mjcf_provenance(self):
        provenance=json.loads((ELF3_XML.parent.parent/'dwaq_provenance.json').read_text())
        self.assertEqual(hashlib.sha256(ELF3_XML.read_bytes()).hexdigest(),provenance['local_mjcf_sha256'])
        self.assertEqual(len(provenance['meshes_sha256']),34)
        for name,expected in provenance['meshes_sha256'].items():
            self.assertEqual(hashlib.sha256((ELF3_XML.parent/'robot_meshes'/name).read_bytes()).hexdigest(),expected,name)

    def test_standalone_xml_also_uses_dwaq_training_parameters(self):
        # Direct XML consumers must not silently receive the previous robot,
        # even when they never call get_elf3_robot_cfg()/get_spec().
        model=mujoco.MjModel.from_xml_path(str(ELF3_XML))
        for name in ELF3_DWAQ_CONTRACT['joint_names']:
            j=model.joint(name).id
            a=model.actuator(name).id
            effort=ELF3_DWAQ_CONTRACT['effort_limit'][name]
            self.assertEqual(model.dof_armature[model.jnt_dofadr[j]],.01)
            np.testing.assert_allclose(model.jnt_actfrcrange[j],[-effort,effort])
            np.testing.assert_allclose(model.actuator_ctrlrange[a],[-effort,effort])

    def test_all_configured_joint_parameters_match_dwaq(self):
        contract=ELF3_DWAQ_CONTRACT
        self.assertEqual(len(ELF3_ACTUATORS),29)
        self.assertEqual(ELF3_HOME_KEYFRAME.pos,tuple(contract['initial_position']))
        for actuator in ELF3_ACTUATORS:
            name=actuator.target_names_expr[0]
            for field in ('stiffness','damping','effort_limit','armature'):
                self.assertEqual(getattr(actuator,field),contract[field][name],f'{name}.{field}')
            home=[v for regex,v in ELF3_HOME_KEYFRAME.joint_pos.items() if re.fullmatch(regex,name)]
            scale=[v for regex,v in ELF3_ACTION_SCALE.items() if re.fullmatch(regex,name)]
            self.assertEqual(home,[contract['default_joint_pos'][name]])
            self.assertEqual(scale,[contract['action_scale'][name]])
        self.assertTrue(all(v==20 for v in contract['velocity_limit'].values()))

    def test_compiled_actuator_and_joint_force_bounds_not_just_config(self):
        robot=Entity(get_elf3_robot_cfg())
        model=robot.spec.compile()
        report=verify_compiled_robot(model)
        self.assertEqual(report['verified_joint_count'],29)
        self.assertFalse(report['verified_gpu_model'])
        np.testing.assert_allclose(model.dof_armature[6:],.01,rtol=0.,atol=1.e-12)
        self.assertAlmostEqual(float(model.body_mass.sum()),43.22,delta=.01)

    def test_original_amp_scope_train_and_play(self):
        for play in (False,True):
            result=verify_original_amp_scope(play)
            self.assertTrue(result['original_rewards_equal'])
            self.assertFalse(result['additional_action_delay'])
            self.assertFalse(result['recovery'])


if __name__=='__main__':
    unittest.main()
