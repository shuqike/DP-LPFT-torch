import os
import random
import numpy as np
import torch


def save(state_path, epoch, model, optimizer, privacy_engine):
    '''
    A checkpoint typically includes not just the model's state_dict (which contains the model's parameters), but also other elements of the training state, like the optimizer state, the epoch number, and potentially the state of the learning rate scheduler if you are using one.
    '''
    save_checkpoint(
        state={
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'privacy_engine_state': privacy_engine.get_state(),
            # Include states of random number generators and other necessary states
        },
        state_path=state_path
    )


def save_checkpoint(state, state_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(state_path, filename))


def load_checkpoint(checkpoint_path, model, optimizer, privacy_engine):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # resume random generator states
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        torch.random.set_rng_state(checkpoint['torch_random_state'])
        if 'cuda_random_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_random_state'])
        # resume training states
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'privacy_engine_state' in checkpoint:
            privacy_engine.load_state_dict(checkpoint['privacy_engine_state'])
        # ... Load other states like random number generator states ...
    else:
        print("No checkpoint found at {}".format(checkpoint_path))

    return start_epoch, model, optimizer, privacy_engine