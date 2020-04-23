from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader, Subset, random_split
from sentence_transformers.readers import TripletReader, MyReader
from sentence_transformers.evaluation import TripletEvaluator, EmbeddingSimilarityEvaluator
from fairseq.models.transformer import TransformerModel
import logging
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

def train(model, output_path, train_data, dev_data, num_epochs=1, batch_size=16):
    logging.info("Read Triplet train dataset")

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

if __name__ == '__main__':
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'models'
    parser = argparse.ArgumentParser(description='Train Sentence-BERT')
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--output_path', type=str, help='batch size')
    args = parser.parse_args()

    dataset_path = data_dir / args.data_path
    normal_path = dataset_path / 'normal_set.txt'
    med_dic_path = dataset_path / 'med_dic.json'

    triplet_reader = TripletReader(dataset_path, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter='\t', quoting=csv.QUOTE_NONE, has_header=False)

    normal_set = metrics.load_normal_disease_set(normal_path)

    with open(med_dic_path, 'r') as f:
        med_dic = json.load(f)

    translater = TransformerModel.from_pretrained(
            str(base_dir / 'translation'),
            checkpoint_file='checkpoint16.pt',
            data_name_or_path=str(base_dir / 'translation/en_ja'))

    converter = expand_abbrev.Converter(med_dic, translater)
        # Use BERT for mapping tokens to embeddings
    word_embedding_model = models.BERT('bert-base-japanese-char')

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    output_path = output_dir / args.output_path
    train_data = SentencesDataset(examples=triplet_reader.get_examples('train.txt'), model=model)
    dev_data = SentencesDataset(examples=triplet_reader.get_examples('valid.txt'), model=model)

    train(model,
            output_path,
            train_data,
            dev_data,
            num_epochs=1,
            batch_size=args.batch_size)
