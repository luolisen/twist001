import copy
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from fusion_core import PublicMemory, CONTACT_TYPES, filtered_candidates, selection_request, target_request, command_digest

class RevisionTests(unittest.TestCase):
    def public(self):
        state=np.zeros(21,np.float32);state[6]=.04;raw=np.zeros((8,7),np.float32);raw[:,6]=.035
        candidates,_=filtered_candidates(raw,state,state[:7],np.array([[-1,1]]*6+[[0,.08]]),list('abcdefg'))
        forecasts=[{'name':c['name'],'candidate_index':c['candidate_index'],'command_sha256':c['command_sha256'],
            'actions':c['commands'].tolist(),'physical_contacts':dict.fromkeys(CONTACT_TYPES,.123456),
            'new_contacts':dict.fromkeys(CONTACT_TYPES,.001234),'unlocalized_contacts':dict.fromkeys(CONTACT_TYPES,0.),
            'any_grasp_estimate':.01,'grip_loss_estimate':.02,'motion_delta':[0.]*10} for c in candidates]
        return {'decision_id':'d0','task':'source','motor_state':state.tolist(),'history_valid':[0,0,1],
            'short_memory':PublicMemory().public(),'candidates':forecasts,'image_layout':'real','scope':'sim'}
    def context(self):
        return {'interface_revision':'consistent_v1','consecutive_hold_count':1,'remaining_consecutive_hold_budget':3,
            'remaining_observation_budget':25,'history_source':'captured_rgb_and_joint_encoders','history_valid_mask':[0,0,1]}
    def target(self):return {'choice':'red','probabilities':{'red':1.}}
    def test_legacy_request_exact(self):
        path=Path(__import__('os').environ.get('LEGACY_FUSION_CORE', str(Path(__file__).parents[2]/'online_fusion_20261009'/'fusion_core.py')))
        spec=importlib.util.spec_from_file_location('legacy_core',path); module=importlib.util.module_from_spec(spec)
        __import__('sys').modules[spec.name]=module;spec.loader.exec_module(module)
        public=self.public()
        self.assertEqual(selection_request(public,self.target()),module.selection_request(public,self.target()))
        self.assertEqual(target_request(public),module.target_request(public))
        public['decision_context']=dict(self.context(),interface_revision='baseline')
        self.assertEqual(selection_request(public,self.target()),module.selection_request({k:v for k,v in public.items() if k!='decision_context'},self.target()))
    def test_alias_same_exact_actions_and_forecast(self):
        public=self.public();public['decision_context']=self.context()
        choices=selection_request(public,self.target())['questions']['selection']['criteria']
        hold=json.loads(choices['command_hold']);observe=json.loads(choices['observe'])
        for key in ('actions','physical_contacts','new_contacts','unlocalized_contacts','motion_delta','name','candidate_index','command_sha256'):
            self.assertEqual(hold[key],observe[key])
        self.assertEqual(command_digest(observe['actions']),hold['command_sha256'])
        state=json.loads(selection_request(public,self.target())['state'])
        self.assertIn('J1-J6 rad',state['action_contract']['actions'])
        self.assertEqual(state['decision_context']['remaining_observation_budget'],25)
    def test_observe_not_executable_without_hold(self):
        public=self.public();public['decision_context']=self.context();public['candidates'].pop()
        choices=selection_request(public,self.target())['questions']['selection']['criteria']
        self.assertNotIn('observe',choices);self.assertIn('reject',choices)
    def test_observation_exhaustion_no_extra_action(self):
        public=self.public();public['decision_context']=dict(self.context(),remaining_observation_budget=0)
        choices=selection_request(public,self.target())['questions']['selection']['criteria']
        self.assertNotIn('observe',choices);self.assertIn('command_hold',choices)
    def test_shared_hold_is_model_visible_and_float32_exact(self):
        public=self.public()
        public['motor_state'][0]=float(np.float32(.123456789))
        hold=public['candidates'][-1]
        hold['actions'][0][0]=float(np.float32(.123456789))
        hold['command_sha256']=command_digest(hold['actions'])
        public['decision_context']=dict(self.context(),presentation_revision='shared_full_hold_v1')
        frozen=copy.deepcopy(public)
        request=selection_request(public,self.target())
        state=json.loads(request['state']);shared=state['shared_physical_actions']['command_hold']
        np.testing.assert_array_equal(np.asarray(shared['actions'],dtype=np.float32),np.asarray(hold['actions'],dtype=np.float32))
        self.assertEqual(command_digest(shared['actions']),hold['command_sha256'])
        self.assertEqual(public,frozen)
        self.assertIn('state.shared_physical_actions.command_hold',request['questions']['selection']['criteria']['command_hold'])
        self.assertIn('state.shared_physical_actions.command_hold',request['questions']['selection']['criteria']['observe'])
        self.assertEqual(shared['name'],'command_hold');self.assertEqual(shared['candidate_index'],hold['candidate_index'])
        for field in ('physical_contacts','new_contacts','unlocalized_contacts','motion_delta','any_grasp_estimate','grip_loss_estimate'):
            legacy=copy.deepcopy(public);legacy.pop('decision_context')
            legacy_hold=json.loads(selection_request(legacy,self.target())['questions']['selection']['criteria']['command_hold'])
            self.assertEqual(shared[field],legacy_hold[field])
    def test_shared_layout_cannot_expose_invalid_hold(self):
        public=self.public();public['candidates'].pop()
        public['decision_context']=dict(self.context(),presentation_revision='shared_full_hold_v1')
        request=selection_request(public,self.target())
        self.assertNotIn('observe',request['questions']['selection']['criteria'])
        self.assertNotIn('shared_physical_actions',json.loads(request['state']))
    def test_shared_layout_omits_trace_ids_only_from_presentation(self):
        public=self.public()
        public['short_memory']['executed_commands']=[{'decision_id':'previous','candidate':'command_hold','command_sha256':'abc','start_time':0.,'end_time':.32,'physics_steps':320}]
        public['decision_context']=dict(self.context(),presentation_revision='shared_full_hold_v1')
        state=json.loads(selection_request(public,self.target())['state'])
        receipt=state['short_memory']['executed_commands'][0]
        self.assertEqual(receipt,{'candidate':'command_hold','start_time':0.,'end_time':.32,'physics_steps':320})
        self.assertEqual(public['short_memory']['executed_commands'][0]['command_sha256'],'abc')
    def test_context_no_privileged_or_fake_history(self):
        for context in (dict(self.context(),object_position=[1,2,3]),dict(self.context(),history_valid_mask=[1,1,1]),
                        dict(self.context(),remaining_observation_budget=-1),dict(self.context(),history_capture_times_seconds=[1,1,1])):
            public=self.public();public['decision_context']=context
            with self.assertRaises(ValueError):selection_request(public,self.target())

if __name__=='__main__':unittest.main()
