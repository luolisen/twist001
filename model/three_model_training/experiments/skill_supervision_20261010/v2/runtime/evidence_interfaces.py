"""Different evidence contracts before release and after release. No truth inputs."""
PUBLIC_SOURCE='public_observation_and_execution'
SUPPORT_KIND='pre_release_support'
PLACEMENT_KIND='post_release_placement'

def valid_public_evidence(e,kind,validators,time_s,target):
    return (time_s is not None and target is not None and e.get('established') is True
            and e.get('evidence_kind')==kind and e.get('validator_id') in validators
            and e.get('source')==PUBLIC_SOURCE and e.get('observation_time_s')==time_s
            and e.get('target')==target)

def unresolved_support(time_s,target=None):
    return {'evidence_kind':SUPPORT_KIND,'established':False,'validator_id':None,
            'source':PUBLIC_SOURCE,'observation_time_s':time_s,'target':target,
            'reason':'descent receipt or robot geometry alone cannot confirm carried object support',
            'unknown':['object support before opening','held offset','support clearance']}

def unresolved_placement(time_s,target=None,release_receipt=None):
    return {'evidence_kind':PLACEMENT_KIND,'established':False,'validator_id':None,
            'source':PUBLIC_SOURCE,'observation_time_s':time_s,'target':target,
            'actual_release_receipt':release_receipt,
            'reason':'after actual release, object retention in region and gripper withdrawal need separately validated public observations',
            'unknown':['final object support','final object in registered region','actual gripper withdrawal']}
