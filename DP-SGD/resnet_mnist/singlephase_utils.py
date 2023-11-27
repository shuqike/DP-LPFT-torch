import os
import random
import numpy as np
import torch


def save(state_path, run_id, run_results, one_run_result, step, sampler, p_model, p_optimizer, privacy_engine):
    '''
    A checkpoint typically includes not just the model's state_dict (which contains the model's parameters), but also other elements of the training state, like the optimizer state, the step number, and potentially the state of the learning rate scheduler if you are using one.
    '''
    step += 1
    save_checkpoint(
        state={
            'run_id': run_id,
            'run_results': run_results,
            'one_run_result': one_run_result,
            'step': step,
            'sampler_state': sampler.get_state(),
            'python_random_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.random.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        state_path=state_path
    )
    privacy_engine.save_checkpoint(
        path=os.path.join(state_path, "priv_checkpoint.pth.tar"),
        module=p_model,
        optimizer=p_optimizer,
    )


def save_checkpoint(state, state_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(state_path, filename))


def load_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # resume random generator states
        random.setstate(checkpoint['python_random_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        torch.random.set_rng_state(checkpoint['torch_rng_state'])
        if checkpoint['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        if 'cuda_random_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_random_state'])
        # resume training states
        run_id = checkpoint['run_id']
        run_results = checkpoint['run_results']
        one_run_result = checkpoint['one_run_result']
        step = checkpoint['step']
        sampler_state = checkpoint['sampler_state']
    else:
        raise FileExistsError("No checkpoint found at {}".format(checkpoint_path))

    return run_id, run_results, one_run_result, step, sampler_state
