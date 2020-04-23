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

def predict(model, output_dir, normal_set, test_x, convert_fn=None):
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

    res = ["出現形\t予測"]
    for origin, normal in zip(test_x, res_words):
        res.append("\t".join([origin, normal]))

    with open(output_dir, 'w') as f:
        f.write('\n'.join(res))


if __name__ == '__main__':
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'models'
    parser = argparse.ArgumentParser(description='Train Sentence-BERT')
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--input_path', type=str, help='data path')
    parser.add_argument('--model_path', type=str, help='data path')
    parser.add_argument('--output_path', type=str, help='batch size')
    args = parser.parse_args()

    dataset_path = data_dir / args.data_path
    normal_path = dataset_path / 'normal_set.txt'
    med_dic_path = dataset_path / 'med_dic.json'

    normal_set = metrics.load_normal_disease_set(normal_path)

    with open(med_dic_path, 'r') as f:
        med_dic = json.load(f)

    translater = TransformerModel.from_pretrained(
            str(base_dir / 'translation'),
            checkpoint_file='checkpoint16.pt',
            data_name_or_path=str(base_dir / 'translation/en_ja'))

    converter = expand_abbrev.Converter(med_dic, translater)

    model = SentenceTransformer(str(model_dir / args.model_path))

    test_x = metrics.load_predict_data(args.input_path)

    predict(model,
            args.output_path,
            normal_set,
            test_x,
            convert_fn=converter.convert)


