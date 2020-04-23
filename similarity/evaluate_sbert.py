from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader, Subset, random_split
from sentence_transformers.readers import TripletReader, MyReader
from sentence_transformers.evaluation import TripletEvaluator, EmbeddingSimilarityEvaluator
from fairseq.models.transformer import TransformerModel
import logging
import numpy as np
import argparse
import os
import sys
import csv
import json
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils import metrics, expand_abbrev

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def evaluate(model, output_dir, normal_set, test_x, test_y, convert_fn=None):
    normal_list = model.encode(normal_set)
    
    if convert_fn is not None:
        input_set = [convert_fn(token) for token in test_x]
    else:
        input_set = [[token] for token in test_x]


    input_set_length = [len(sent) for sent in input_set]
    input_set = sum(input_set, [])

    targets = model.encode(input_set)

    idx, sim = metrics.find_similar_words(targets, normal_list, k=1)
    res_words = []

    cnt = 0
    for l in input_set_length:
        tmp_idx = idx[cnt:cnt+l, :].reshape(-1)
        tmp_sim = sim[cnt:cnt+l, :].reshape(-1)
        rank = np.argsort(tmp_sim)[::-1][0]
        res_words.append(normal_set[tmp_idx[rank]])
        cnt += l

    res = ["出現形\t正解\t予測"]
    for origin, normal, test in zip(test_x, res_words, test_y):
        res.append("\t".join([origin, test, normal]))
    accuracy, positive_example, negative_example = metrics.calculate_accuracy(test_x, res_words, test_y)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, 'result.txt'), 'w') as f:
        f.write('\n'.join([str(accuracy)] + res))

    with open(os.path.join(output_dir, 'pos_example.txt'), 'w') as f:
        f.write('\n'.join(positive_example))

    with open(os.path.join(output_dir, 'neg_example.txt'), 'w') as f:
        f.write('\n'.join(negative_example))

    return accuracy, positive_example, negative_example

if __name__ == '__main__':
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'models'
    output_dir = base_dir / 'results'
    parser = argparse.ArgumentParser(description='Train Sentence-BERT')
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--model_path', type=str, help='data path')
    parser.add_argument('--output_path', type=str, help='batch size')
    args = parser.parse_args()

    dataset_path = data_dir / args.data_path
    normal_path = dataset_path / 'normal_set.txt'
    med_dic_path = dataset_path / 'med_dic.json'
    test_path = str(dataset_path / 'test.txt')

    normal_set = metrics.load_normal_disease_set(normal_path)

    with open(med_dic_path, 'r') as f:
        med_dic = json.load(f)

    translater = TransformerModel.from_pretrained(
            str(base_dir / 'translation'),
            checkpoint_file='checkpoint16.pt',
            data_name_or_path=str(base_dir / 'translation/en_ja'))

    converter = expand_abbrev.Converter(med_dic, translater)

    model = SentenceTransformer(str(model_dir / args.model_path))

    output_path = output_dir / args.output_path
    test_y, test_x = metrics.load_test_data(test_path)

    accuracy, pos_example, neg_example = evaluate(model,
            output_path,
            normal_set,
            test_x,
            test_y,
            convert_fn=converter.convert)

    print(accuracy)

