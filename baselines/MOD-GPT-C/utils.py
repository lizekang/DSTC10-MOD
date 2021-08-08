import torch 
import logging
import os


def try_create_dir(dir_path):
    if os.path.exists(dir_path):
        return
    os.makedirs(dir_path)

def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# calculate the accuracy of response  
def accuracy_compute(lm_logits, targets, k=5):
    _, idx = torch.topk(lm_logits, k, 1)
    correct = idx.eq(targets.view(-1,1).expand_as(idx))
    correct_total = correct.view(-1).float().sum().item()
    nums = targets.view(-1).detach().cpu().numpy()
    length = 0
    for num in nums:
        if num != -100:
            length += 1
    return correct_total / float(length)


def meme_classify_accuracy(mf_logits, meme_flag):
    prediction = torch.argmax(mf_logits, 1) 
    return (prediction == meme_flag).sum()


# class for evaluation metric 
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count