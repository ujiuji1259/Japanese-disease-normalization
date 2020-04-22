# Japanese-disease-normalization
## 概要
日本語の病名標準化モジュールです．

## データセット
モデルの学習，評価には[万病辞書](http://sociocom.jp/~data/2018-manbyo/index.html)を使用しています．

## 手法
[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks:](https://arxiv.org/abs/1908.10084)でTripletLossを学習し，
コサイン類似度によって標準病名の予測を行っています．
