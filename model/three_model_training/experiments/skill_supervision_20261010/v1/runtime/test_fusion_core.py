"""Behavior tests for mapping, missing history, bounded recovery and GT isolation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fusion_core import (Budget, PublicMemory, filtered_candidates, lookup_candidate, command_digest,
                         check_memory, check_public, history_arrays, recovery_change, selection_request, CONTACT_TYPES)
from fusion_core import execute_with_archive, fatal_suite_error
from action_intervention import apply_intervention
from model_binding import REQUIRED, verify_binding
from fusion_core import sha


class CoreTests(unittest.TestCase):
    def candidates(self):
        state=np.zeros(21,dtype=np.float32);state[6]=.04
        raw=np.zeros((8,7),dtype=np.float32);raw[:,6]=.03
        limits=np.array([[-1,1]]*6+[[0,.08]])
        return state,raw,limits

    def test_invalid_hold_is_isolated_and_indices_follow_legal_set(self):
        state,raw,limits=self.candidates();state[1]=1.01
        candidates,boundary=filtered_candidates(raw,state,np.r_[np.zeros(6),.04],limits,list('abcdefg'))
        self.assertIn('pose_hold',boundary['rejected_candidate_names'])
        self.assertNotIn('pose_hold',[c['name'] for c in candidates])
        for index,candidate in enumerate(candidates):
            selected=lookup_candidate(candidates,candidate['name'])
            self.assertEqual(index,selected['candidate_index'])
            self.assertEqual(command_digest(selected['commands']),selected['command_sha256'])
        with self.assertRaises(ValueError):lookup_candidate(candidates,'pose_hold')

    def test_post_forecast_mutation_rejected(self):
        state,raw,limits=self.candidates()
        candidates,_=filtered_candidates(raw,state,np.r_[np.zeros(6),.04],limits,list('abcdefg'))
        candidates[0]['commands'][0,0]=.1
        with self.assertRaises(ValueError):lookup_candidate(candidates,'vla_raw')

    def test_empty_candidate_set_does_not_map_to_index_zero(self):
        state,raw,limits=self.candidates();raw[:]=np.nan;state[:7]=np.nan
        candidates,boundary=filtered_candidates(raw,state,np.full(7,np.nan),limits,list('abcdefg'))
        self.assertFalse(candidates);self.assertTrue(boundary['stop_required'])

    def test_missing_history_is_zero_masked_not_duplicate(self):
        item={'time':4.16,'small':np.full((2,128,128,3),100,np.uint8),'state':np.ones(21,np.float32)}
        images,states,valid,times=history_arrays([item])
        np.testing.assert_array_equal(valid,[0,0,1])
        self.assertEqual(images[:2].sum(),0);self.assertEqual(states[:2].sum(),0)
        np.testing.assert_array_equal(images[-1],item['small'])
        self.assertEqual(times[-1],4.16)
        with self.assertRaises(ValueError):history_arrays([item,dict(item,time=4.16)])

    def test_actual_new_frames_advance_to_all_valid(self):
        history=[{'time':t,'small':np.full((2,2,2,3),i,np.uint8),'state':np.full(21,i,np.float32)}
                 for i,t in enumerate([4.16,4.20,4.24])]
        images,states,valid,times=history_arrays(history)
        np.testing.assert_array_equal(valid,[1,1,1]);self.assertEqual(images[-1,0,0,0,0],2)
        self.assertEqual(states[-1,0],2)

    def test_replan_change_is_candidate_change_not_success(self):
        state,raw,limits=self.candidates()
        first,_=filtered_candidates(raw,state,np.r_[np.zeros(6),.04],limits,list('abcdefg'))
        second=copy.deepcopy(first)
        self.assertFalse(recovery_change(first,second)['changed'])
        second[0]['commands'][0,0]=.1;second[0]['command_sha256']=command_digest(second[0]['commands'])
        report=recovery_change(first,second)
        self.assertTrue(report['changed']);self.assertEqual(report['changed_names'],['vla_raw'])
        self.assertIn('not proof',report['definition'])

    def test_local_and_suite_budgets_stop_without_hidden_calls(self):
        parent=Budget({'VLA_calls':2});local=Budget({'VLA_calls':3},parent=parent)
        local.charge('VLA_calls');local.charge('VLA_calls')
        with self.assertRaises(RuntimeError):local.charge('VLA_calls')
        self.assertEqual(parent.counts['VLA_calls'],2);self.assertEqual(local.counts['VLA_calls'],2)

    def test_encoder_closure_is_not_task_completion(self):
        memory=PublicMemory();before=np.zeros(21);after=np.zeros(21);before[6]=.04;after[6]=.03
        memory.receipt('d0','vla_raw',before,after,4.16,4.48,320,'abc')
        public=memory.public();check_memory(public)
        self.assertEqual(public['stage'],'closure_observed');self.assertIsNone(public['system_completion']['value'])
        self.assertTrue(public['verified_progress'][-1]['not_grasp_confirmation'])

    def test_GT_cannot_enter_public_memory(self):
        memory=PublicMemory().public()
        for polluted in [dict(memory,object_position=[1,2,3]),
                         dict(memory,verified_progress=[{'kind':'strict_grasp','source':'mujoco_truth'}]),
                         dict(memory,prior_decisions=[{'decision_id':'d0','proposal_only':True,'private_contact_id':5}]),
                         dict(memory,system_completion={'value':True})]:
            with self.assertRaises(ValueError):check_memory(polluted)

    def test_target_proposal_enters_dynamic_selection_and_complete_absent(self):
        state,raw,limits=self.candidates();candidates,_=filtered_candidates(raw,state,np.r_[np.zeros(6),.04],limits,list('abcdefg'))
        forecasts=[{'name':c['name'],'candidate_index':c['candidate_index'],'command_sha256':c['command_sha256'],
                    'actions':c['commands'].tolist(),'physical_contacts':dict.fromkeys(CONTACT_TYPES,0.),
                    'new_contacts':dict.fromkeys(CONTACT_TYPES,0.),'unlocalized_contacts':dict.fromkeys(CONTACT_TYPES,0.),
                    'any_grasp_estimate':0.,'grip_loss_estimate':0.,'motion_delta':[0.]*10} for c in candidates]
        public={'decision_id':'d0','task':'registered source text','motor_state':state.tolist(),'history_valid':[0,0,1],
                'short_memory':PublicMemory().public(),'candidates':forecasts,'image_layout':'actual','scope':'sim'}
        check_public(public)
        req=selection_request(public,{'choice':'red','probabilities':{'red':1.,'not_visible':0.,'uncertain':0.}})
        self.assertIn('native_RGB_target_proposal',req['state']);self.assertIn('observe',req['questions']['selection']['criteria'])
        self.assertNotIn('complete',req['questions']['selection']['criteria'])
        public['candidates']=public['candidates'][1:]
        for index,c in enumerate(public['candidates']):c['candidate_index']=index
        req=selection_request(public,{'choice':'red','probabilities':{'red':1.}})
        self.assertNotIn('vla_raw',req['questions']['selection']['criteria'])

    def test_Jev_criteria_actions_roundtrip_exactly_without_five_digit_rounding(self):
        state,raw,limits=self.candidates();raw[0,0]=np.float32(.123456789)
        candidates,_=filtered_candidates(raw,state,np.r_[np.zeros(6),.04],limits,list('abcdefg'))
        forecasts=[{'name':c['name'],'candidate_index':c['candidate_index'],'command_sha256':c['command_sha256'],
                    'actions':c['commands'].tolist(),'physical_contacts':dict.fromkeys(CONTACT_TYPES,0.),
                    'new_contacts':dict.fromkeys(CONTACT_TYPES,0.),'unlocalized_contacts':dict.fromkeys(CONTACT_TYPES,0.),
                    'any_grasp_estimate':0.,'grip_loss_estimate':0.,'motion_delta':[0.]*10} for c in candidates]
        public={'decision_id':'d0','task':'source','motor_state':state.tolist(),'history_valid':[1,1,1],
                'short_memory':PublicMemory().public(),'candidates':forecasts,'image_layout':'real','scope':'sim'}
        req=selection_request(public,{'choice':'red','probabilities':{'red':1.}})
        for candidate in candidates:
            presented=json.loads(req['questions']['selection']['criteria'][candidate['name']])
            np.testing.assert_array_equal(np.asarray(presented['actions'],dtype=np.float32),candidate['commands'])
            self.assertEqual(command_digest(presented['actions']),candidate['command_sha256'])

    def test_software_exception_archives_current_prefix_without_extra_steps(self):
        state={'step':0};archives=[];error=RuntimeError('wall_clock_stop')
        def operation():
            state['step']+=3
            raise error
        def archive(completed):archives.append((completed,state['step']))
        with self.assertRaises(RuntimeError) as caught:
            execute_with_archive(operation,archive,lambda error:self.fail('archive failed'))
        self.assertIs(caught.exception,error);self.assertEqual(state['step'],3)
        self.assertEqual(archives,[(False,3)])

    def test_archive_failure_does_not_replace_execution_exception(self):
        original=ValueError('original_execution_failure');failures=[]
        def operation():raise original
        def archive(completed):raise OSError('disk_error')
        with self.assertRaises(ValueError) as caught:
            execute_with_archive(operation,archive,lambda error:failures.append(str(error)))
        self.assertIs(caught.exception,original);self.assertEqual(failures,['disk_error'])

    def test_invalid_shared_Jev_state_is_fatal_not_reused(self):
        for reason in ('Jev_service_terminated','Jev_response_timeout','Jev IPC response mismatch',
                       'budget_exhausted:physics_steps','suite_maximum_wall_seconds_exhausted'):
            self.assertTrue(fatal_suite_error(RuntimeError(reason)))
        self.assertFalse(fatal_suite_error(RuntimeError('raw_VLA_candidate_rejected')))

    def test_residual_half_uses_public_anchor_and_keeps_gap(self):
        state,raw,_=self.candidates();state[:6]=.2;raw[:,:6]=.4
        half,record=apply_intervention(raw,state,'joint_residual_half')
        np.testing.assert_allclose(half[:,:6],.3);np.testing.assert_array_equal(half[:,6],raw[:,6])
        none,_=apply_intervention(raw,state,'none');np.testing.assert_array_equal(none,raw)
        self.assertTrue(record['not_pure_unassisted_VLA']);self.assertFalse(record['clipping'])

    def test_binding_rejects_model_only_path(self):
        with self.assertRaises(ValueError):
            verify_binding({'checkpoint_path':'/not_used','files_sha256':{'/not_used/model.safetensors':'x'}})

    def test_binding_checks_exact_normalization_state_and_processor_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in REQUIRED:(root/name).write_bytes(b'placeholder')
            (root/'config.json').write_text(json.dumps({'input_features':{
                'observation.state':{'shape':[7]},'observation.images.global':{'shape':[3,256,256]},
                'observation.images.wrist':{'shape':[3,512,512]}},'output_features':{'action':{'shape':[7]}}}))
            (root/'policy_preprocessor.json').write_text(json.dumps({'steps':[{'state_file':REQUIRED[4]}]}))
            (root/'policy_postprocessor.json').write_text(json.dumps({'steps':[{'state_file':REQUIRED[5]}]}))
            vlm=root/'vlm';vlm.mkdir()
            for name in ('config.json','tokenizer.json','tokenizer_config.json'):(vlm/name).write_text('{}')
            binding={'checkpoint_path':str(root),'files_sha256':{str(root/n):sha(root/n) for n in REQUIRED},
                     'VLM_source':str(vlm),'VLM_files_sha256':{str(p):sha(p) for p in vlm.iterdir()},'device':'mps',
                     'preprocessor_overrides':{'device_processor':{'device':'mps'},'tokenizer_processor':{'tokenizer_name':str(vlm)}}}
            self.assertEqual(verify_binding(binding)['files_verified'],6)
            mismatched=copy.deepcopy(binding);mismatched['preprocessor_overrides']['tokenizer_processor']['tokenizer_name']='/another/tokenizer'
            with self.assertRaises(ValueError):verify_binding(mismatched)
            (vlm/'tokenizer.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):verify_binding(binding)
            (vlm/'tokenizer.json').write_text('{}')
            (root/REQUIRED[4]).write_bytes(b'other normalization')
            with self.assertRaises(ValueError):verify_binding(binding)
            binding['files_sha256'][str(root/REQUIRED[4])]=sha(root/REQUIRED[4])
            (root/'policy_preprocessor.json').write_text(json.dumps({'steps':[{'state_file':'unregistered.safetensors'}]}))
            binding['files_sha256'][str(root/'policy_preprocessor.json')]=sha(root/'policy_preprocessor.json')
            with self.assertRaises(ValueError):verify_binding(binding)


if __name__=='__main__':unittest.main()
