"""Archive model output before a possibly mutating postprocessor, with no labels."""
def predict_with_capture(policy, pre, post, frame):
    import torch
    batch = pre(frame)
    network = policy.predict_action_chunk(batch)
    raw = network.detach().cpu().numpy().copy().reshape(50, 7)
    processed = post(network).detach().cpu().numpy().copy().reshape(50, 7)
    state = batch['observation.state'].detach().cpu().numpy().copy()
    return processed, raw, state
