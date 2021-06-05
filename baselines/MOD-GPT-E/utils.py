import torch 


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