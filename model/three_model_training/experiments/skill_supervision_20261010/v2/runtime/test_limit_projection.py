import unittest
import numpy as np
from limit_projection import project_arm_targets


class ProjectionTest(unittest.TestCase):
    def test_unchanged_legal_commands_and_gripper(self):
        q=np.zeros((8,7),dtype=np.float32);q[:,6]=.04
        limits=np.array([[-1,1]]*6+[[0,.08]],dtype=np.float64)
        out,r=project_arm_targets(q,limits)
        np.testing.assert_array_equal(out,q)
        self.assertEqual(r['changed_count'],0)

    def test_projection_uses_original_limits_and_reports_changes(self):
        q=np.zeros((8,7),dtype=np.float32);q[:4,1]=[.01,.02,.03,.04637301];q[:,6]=.05
        limits=np.array([[-1,1]]*6+[[0,.08]],dtype=np.float64);limits[1,1]=0
        out,r=project_arm_targets(q,limits)
        np.testing.assert_array_equal(out[:4,1],np.zeros(4))
        np.testing.assert_array_equal(out[:,6],q[:,6])
        self.assertEqual(r['changed_count'],4)
        self.assertFalse(r['limits_relaxed'])
        self.assertAlmostEqual(r['max_abs_delta_rad'],.04637301)

    def test_float32_rounding_does_not_create_new_limit_violation(self):
        limits=np.array([[-.1,.1]]*6+[[0,.08]],dtype=np.float64)
        q=np.full((8,7),.2,dtype=np.float32);q[:,6]=.04
        out,_=project_arm_targets(q,limits)
        self.assertTrue(np.all(out[:,:6].astype(np.float64)<=.1))
        self.assertTrue(np.all(out[:,:6].astype(np.float64)>=-.1))

    def test_nonfinite_rejected_without_clipping(self):
        q=np.zeros((8,7),dtype=np.float32);q[0,0]=np.nan
        with self.assertRaises(ValueError):
            project_arm_targets(q,np.array([[-1,1]]*7))


if __name__=='__main__':unittest.main()
