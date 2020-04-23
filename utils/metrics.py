import numpy as np
import Levenshtein
from tqdm import tqdm

def load_normal_disease_set(path):
    with open(path, 'r') as f:
        lines = [line for line in f.read().split('\n') if line != '']

    return lines

def most_similar_words_edit_distance(targets, normal_list, k=1):
    res = []
    for t in targets:
        sim = []
        for token in normal_list:
            dist = Levenshtein.distance(t, token)
            dist = 1 - 2 * dist / (len(t) + len(token))
            sim.append(dist)

        sim = np.array(sim)
        rank = np.argsort(sim)[::-1][0]
        res.append(normal_list[rank])
    return res

def find_similar_words(targets, normal_list, k=10):
    norm_target = np.linalg.norm(targets, ord=2, axis=1)[:, np.newaxis]
    norm_normal_list = np.linalg.norm(normal_list, ord=2, axis=1)[:, np.newaxis]
    targets /= norm_target + 1e-8
    normal_list /= norm_normal_list + 1e-8
    sim = normal_list @ targets.T

    idx = np.argsort(sim, axis=0)[::-1, :][:k, :]
    sim = np.take_along_axis(sim, idx, axis=0)
    return idx.T, sim.T


def load_test_data(path):
    with open(path, 'r') as f:
        lines = [line for line in f.read().split('\n') if line != '']
    lines = [line.split('\t') for line in lines]

    test_x = [line[0] for line in lines]
    test_y = [line[1] for line in lines]

    return test_x, test_y

def calculate_accuracy(test_x, preds, test_y):
    pos = 0
    neg = 0
    positive_example = []
    negative_example = []
    for true, pred, line in zip(test_y, preds, test_x):
        if true == pred:
            pos += 1
            positive_example.append('\t'.join([line, true, pred]))
        else:
            neg += 1
            negative_example.append('\t'.join([line, true, pred]))

    return pos/(pos+neg), positive_example, negative_example

if __name__ == '__main__':
    with open('datasets/S.txt', 'r') as f:
        lines = [line.split('\t')[2] for line in f.read().split('\n') if line != ""]
    for line in lines:
        if line == "僧帽弁逆流症":
            print(line)

    with open('datasets/normal_set.txt', 'w') as f:
        f.write(','.join(lines))
