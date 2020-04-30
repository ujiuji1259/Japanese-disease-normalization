from transformers import BertForTokenClassification, BertJapaneseTokenizer, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from fairseq.models.transformer import TransformerModel
from flask import Flask, render_template, request
import argparse
import json
import torch
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils import metrics, expand_abbrev
from predict_sbert import predict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)

@app.route("/", methods=['POST'])
def index():
    word = request.form["text"].split(',')
    res_words = predict(model,
            None,
            normal_set,
            word,
            convert_fn=converter.convert,
            normal_vecs=normal_list)

    return ','.join(res_words)

if __name__ == '__main__':
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'models'
    parser = argparse.ArgumentParser(description='Train BERT')
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--model_path', type=str, help='data path')
    args = parser.parse_args()

    tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese-char")

    dataset_path = data_dir / args.data_path
    normal_path = dataset_path / 'normal_set.txt'
    med_dic_path = dataset_path / 'med_dic.json'

    with open(med_dic_path, 'r') as f:
        med_dic = json.load(f)

    normal_set = metrics.load_normal_disease_set(normal_path)

    translater = TransformerModel.from_pretrained(
            str(base_dir / 'translation'),
            checkpoint_file='checkpoint16.pt',
            data_name_or_path=str(base_dir / 'translation/en_ja'))
    converter = expand_abbrev.Converter(med_dic, translater)
    model = SentenceTransformer(str(model_dir / args.model_path))
    normal_list = model.encode(normal_set)

    app.run(port='8001', host='0.0.0.0', debug=True)
