import unittest
import numpy as np
import torch
from prediction_capture import predict_with_capture

class CaptureTest(unittest.TestCase):
    def test_mutating_postprocessor_cannot_rewrite_network_archive(self):
        class Policy:
            def predict_action_chunk(self,batch):return torch.ones(1,50,7)
        def post(value):return value.mul_(2)
        processed,raw,state=predict_with_capture(Policy(),lambda x:x,post,{'observation.state':torch.zeros(1,7)})
        self.assertTrue(np.all(raw==1));self.assertTrue(np.all(processed==2))
        self.assertEqual(state.shape,(1,7))

    def test_teacher_fields_are_not_added_to_policy_input(self):
        class Policy:
            def predict_action_chunk(self,batch):
                assert set(batch)=={'observation.state','task'}
                return torch.zeros(1,50,7)
        predict_with_capture(Policy(),lambda x:x,lambda x:x,{'observation.state':torch.zeros(1,7),'task':'original'})

if __name__=='__main__':unittest.main(verbosity=2)
