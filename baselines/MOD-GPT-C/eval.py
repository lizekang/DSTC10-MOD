from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction 
from nltk.util import ngrams
import json 


def compute_bleu(hyps, refs, n=1, corpus_level=True):
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0), (0.25, 0.25, 0.25, 0.25)]
    bleu_weights = weights[n-1]
    if corpus_level:
        refs = [[ref] for ref in refs]
        return corpus_bleu(refs, hyps, weights=bleu_weights, smoothing_function=SmoothingFunction().method1)
    else:
        avg_score = 0
        for hyp, ref in zip(hyps, refs):
            # for chinese string, one character is a token, so no need to tokenize first
            score = sentence_bleu([ref], hyp, weights=bleu_weights, smoothing_function=SmoothingFunction().method1)
            avg_score += score
        return avg_score / len(hyps)

def compute_distinct(hyps, n=2):
    all_ngrams = []
    for hyp in hyps:
        all_ngrams.extend(ngrams(hyp, n=n))
    
    return len(set(all_ngrams)) / len(all_ngrams)

def preprocess(sent):
    return sent.replace('[BOS]', '')

def eval(file_name):
    hyps = []
    refs = []
    with open(file_name, encoding='utf-8') as f:
        a = json.load(f)
        for d in a:
            hyps.append(preprocess(d['hyp']))
            refs.append(preprocess(d['ref']))

    res = {}
    
    for n in range(1, 5):
        sent_bleu_score = compute_bleu(hyps, refs, n=n, corpus_level=False)
        res[f'sent_bleu_{n}'] = sent_bleu_score

    for n in range(1, 5):
        corpus_bleu_score = compute_bleu(hyps, refs, n=n, corpus_level=True)
        res[f'corpus_bleu_{n}'] = corpus_bleu_score

    for n in range(1, 5):
        distinct_score = compute_distinct(hyps, n=n)
        res[f'distinct_{n}'] = distinct_score

    
    print(res)

if __name__ == '__main__':
    eval('result.json')
