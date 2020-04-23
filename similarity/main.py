"""
This script trains sentence transformers with a triplet loss function.

As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.

See docs/pretrained-models/wikipedia-sections-modesl.md for further details.

You can get the dataset by running examples/datasets/get_data.py
"""

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader, Subset, random_split
from sentence_transformers.readers import TripletReader, MyReader
from sentence_transformers.evaluation import TripletEvaluator, EmbeddingSimilarityEvaluator
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split

from evaluation_func import load_normal_disease_set, load_test_data, most_similar_words_edit_distance, find_similar_words, calculate_accuracy
from expand_abbrev import Converter
import csv
import logging
from fairseq.models.transformer import TransformerModel

from omegaconf import DictConfig
import hydra
import pickle
import numpy as np

import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

@hydra.main('config.yaml')
def main(cfg: DictConfig) -> None:
    batch_size = cfg.model.batch_size
    data_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.dir)
    triplet_reader = TripletReader(data_path, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter='\t', quoting=csv.QUOTE_NONE, has_header=False)

    normal_set = load_normal_disease_set(os.path.join(hydra.utils.get_original_cwd(), cfg.data.normal_set_path))

    with open(os.path.join(hydra.utils.get_original_cwd(), cfg.data.med_dic_path), 'rb') as f:
        med_dic = pickle.load(f)

    translater = TransformerModel.from_pretrained(
            '/home/is/ujiie/fairseq/checkpoints',
            checkpoint_file='checkpoint16.pt',
            data_name_or_path='/home/is/ujiie/fairseq/data-bin/en_ja')

    converter = Converter(med_dic, "all", translater)

    if cfg.model.type == 'cross_validation':
        cross_validate(triplet_reader,
                data_path,
                normal_set,
                os.path.join(hydra.utils.get_original_cwd(), cfg.eval.output_dir),
                validated_model_dir=cfg.model.dev_dir,
                batch_size=batch_size,
                convert_fn=converter.convert)

    elif cfg.model.type == 'train':
        # Use BERT for mapping tokens to embeddings
        #word_embedding_model = models.BERT('bert-base-japanese-char')
        word_embedding_model = models.BERT('bert-base-multilingual-cased')

        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        output_path = os.path.join(hydra.utils.get_original_cwd(), cfg.model.output_path)
        train_dev_data = SentencesDataset(examples=triplet_reader.get_examples('train_han.txt'), model=model)

        train(model,
                output_path,
                train_dev_data,
                num_epochs=cfg.model.epoch,
                dev_size=cfg.model.dev_size,
                batch_size=batch_size)

    else:
        model = SentenceTransformer(os.path.join(hydra.utils.get_original_cwd(), cfg.model.input_path))
        test_path = os.path.join(data_path, 'test_en_han.txt')
        test_y, test_x = load_test_data(test_path)

        accuracy, pos_example, neg_example = evaluate(model,
                os.path.join(hydra.utils.get_original_cwd(), cfg.eval.output_dir),
                normal_set,
                test_x,
                test_y,
                #convert_fn=converter.convert)
                convert_fn=None)

        print(accuracy)

def train(model, output_path, train_dev_data, num_epochs=1, dev_size=0.1, batch_size=16):
    logging.info("Read Triplet train dataset")

    dev_size = int(len(train_dev_data) * 0.1)
    train_size = len(train_dev_data) - dev_size
    train_data, dev_data = random_split(train_dev_data, [train_size, dev_size])
    
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.TripletLoss(model=model)

    logging.info("Read Wikipedia Triplet dev dataset")
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
    evaluator = TripletEvaluator(dev_dataloader)


    warmup_steps = int(len(train_data)*num_epochs/batch_size*0.1) #10% of train data


# Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=output_path)

def evaluate(model, output_dir, normal_set, test_x, test_y, convert_fn=None):
    normal_list = model.encode(normal_set)
    
    if convert_fn is not None:
        input_set = [convert_fn(token) for token in test_x]
    else:
        input_set = [[token] for token in test_x]


    input_set_length = [len(sent) for sent in input_set]
    input_set = sum(input_set, [])

    targets = model.encode(input_set)

    idx, sim = find_similar_words(targets, normal_list, k=1)
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
    accuracy, positive_example, negative_example = calculate_accuracy(test_x, res_words, test_y)

    with open(os.path.join(output_dir, 'result.txt'), 'w') as f:
        f.write('\n'.join([str(accuracy)] + res))

    with open(os.path.join(output_dir, 'pos_example.txt'), 'w') as f:
        f.write('\n'.join(positive_example))

    with open(os.path.join(output_dir, 'neg_example.txt'), 'w') as f:
        f.write('\n'.join(negative_example))

    return accuracy, positive_example, negative_example

def cross_validate(reader, data_path, normal_set, output_dir, validated_model_dir='models', batch_size=16, convert_fn=None):
    ### Configure sentence transformers for training and train on the provided dataset
    # Use BERT for mapping tokens to embeddings
    word_embedding_model = models.BERT('bert-base-multilingual-cased')

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    kf = KFold(n_splits=5)

    total_accuracy = 0
    for fold in range(1, 6):
        validated_model_path = os.path.join(validated_model_dir, "dev{}.model".format(str(fold)))

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        train_dev_data = SentencesDataset(examples=reader.get_examples('validation' + str(fold) + '_train.txt'), model=model)

        train(model, validated_model_path, train_dev_data, num_epochs=1, dev_size=0.1, batch_size=batch_size)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

        test_path = os.path.join(data_path, 'validation' + str(fold) + '_test.txt')
        test_y, test_x = load_test_data(test_path)

        accuracy, pos_example, neg_example = evaluate(model, output_dir, normal_set, test_x, test_y, convert_fn=convert_fn)
        fold += 1
        print(accuracy)
        total_accuracy += accuracy
    print(total_accuracy / 5)


if __name__ == '__main__':
    main()
