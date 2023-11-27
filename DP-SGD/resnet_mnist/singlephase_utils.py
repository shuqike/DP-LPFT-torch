import os
import random
import numpy as np
import torch


def save(state_path, run_id, run_results, one_run_result, epoch, model, optimizer, privacy_engine, train_loader):
    '''
    A checkpoint typically includes not just the model's state_dict (which contains the model's parameters), but also other elements of the training state, like the optimizer state, the epoch number, and potentially the state of the learning rate scheduler if you are using one.
    '''
    save_checkpoint(
        state={
            'run_id': run_id,
            'run_results': run_results,
            'one_run_result': one_run_result,
            'epoch': epoch + 1,
            'model': model,
            'optimizer': optimizer,
            'privacy_engine': privacy_engine,
            'train_loader': train_loader
        },
        state_path=state_path
    )


def save_checkpoint(state, state_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(state_path, filename))


def load_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # resume random generator states
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        torch.random.set_rng_state(checkpoint['torch_random_state'])
        if 'cuda_random_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_random_state'])
        # resume training states
        run_id = checkpoint['run_id']
        run_results = checkpoint['run_results']
        one_run_result = checkpoint['one_run_result']
        epoch = checkpoint['epoch']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        if 'privacy_engine' in checkpoint:
            privacy_engine = checkpoint['privacy_engine']
        train_loader = checkpoint['train_loader']
    else:
        raise FileExistsError("No checkpoint found at {}".format(checkpoint_path))

    return run_id, run_results, one_run_result, epoch, model, optimizer, privacy_engine, train_loader
