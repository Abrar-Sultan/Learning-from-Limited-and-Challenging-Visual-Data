import numpy as np
import torch.nn.functional as F


def softmax_mse_loss(input_logits, target_logits):
    '''
    Takes softmax on both sides and returns MSE loss
    this code is from offical mean teacher implementation https://github.com/CuriousAI/mean-teacher
    '''
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def sigmoid_rampup(current, rampup_length): 
    '''
    sigmoid rampup for the updating the consistancy weight for consistancy loss 
    from offical mean teacher implementation https://github.com/CuriousAI/mean-teacher
    '''

    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    

def linear_rampup(current, rampup_length):
    '''
    Linear rampup returns a value that increases linearly from 0 to 1 over rampup_length
    used for scheduing learning rate gradually at the begining 
    from offical mean teacher implementation https://github.com/CuriousAI/mean-teacher
    '''
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    '''
    cosine ranpdown 
    from offical mean teacher implementation https://github.com/CuriousAI/mean-teacher
    '''
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1)) 


def learning_rate_scheduler(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    '''
    Adjust the learning rate using linear ramp-up and cosine ramp-down
    from offical mean teacher implementation https://github.com/CuriousAI/mean-teacher
    '''
    # compute current progress within the entire training schedule
    progress = epoch + step_in_epoch / total_steps_in_epoch

    # linear lr ramp-up
    lr = linear_rampup(progress, args.lr_rampup) * (args.learning_rate - args.initial_learning_rate) + args.initial_learning_rate

    # cosine lr ramp-down
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs, \
            "Ramp-down period must be >= total epochs"
        lr *= cosine_rampdown(progress, args.lr_rampdown_epochs)

    # apply to all param groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr