# Cross-language Sentence Selection via Data Augmentation and Rationale Training

This is the implementation of the paper [Cross-language Sentence Selection via Data Augmentation and Rationale Training](https://aclanthology.org/2021.acl-long.300.pdf). 
## Quick links

* [Overview](#overview)
* [Requirements](#requirements)
* [Preprocess the data](#preprocess-the-data)
* [Build the vocabulary](#build-the-vocabulary)
* [Augment the relevance dataset](#augment-the-relevance-dataset)
* [Train the SECLR-RT relevance model](#train-the-seclr-rt-relevance-model)
* [Predict relevance with SECLR-RT](#predict-with-the-trained-model)
* [Citation](#citation)


## Overview
In this work we present an effective model for cross-language sentence retrieval in low-resource settings 
where a labeled relevance dataset (cross-lingual query-sentence pairs with relevance judgment) is not available. Our 
approach includes:
1. A data augmentation method with negative sampling techniques that augment a labeled relevance dataset based on noisy
parallel sentence data.
2. A simple yet very effective cross-lingual embedding-based query relevance model (SECLR) that directly optimizes the
cross-lingual query relevance objective.
3. A rationale training technique (SECLR-RT) that guides SECLR's training with additional word alignment supervision 
from a statistical machine translation model. 
   
You could find more details of this work in our [paper](https://aclanthology.org/2021.acl-long.300.pdf).

## Requirements

To run our code, please install all the dependency packages by using the following command:
```
pip install -r requirements.txt
```

## Preprocess the data
To install our normalization and tokenization package, please run the following commands:
```
cd src/updated_normalization
python setup.py install --user
```

We provide a script `src/preprocess_bitext.py` to preprocess a parallel sentence corpus. The preprocessing step involves 
morphological normalization, sentence tokenization and stopwords removal. 
You can run the script by:

```bash
python preprocess_bitext.py [-h] [--src_language SRC_LANGUAGE]
                            [--en_bitext_path EN_BITEXT_PATH]
                            [--src_bitext_path SRC_BITEXT_PATH]
                            [--en_stopwords_path EN_STOPWORDS_PATH]
                            [--src_stopwords_path SRC_STOPWORDS_PATH]
                            [--output_preprocessed_en_bitext_path OUTPUT_PREPROCESSED_EN_BITEXT_PATH]
                            [--output_preprocessed_src_bitext_path OUTPUT_PREPROCESSED_SRC_BITEXT_PATH]
```
The input bitexts file are expected to have one sentence per line. See ```parallel_corpus/en-so/en-raw.txt```
and ```parallel_corpus/en-so/so-raw.txt``` as example. The stopwords are expected to have one word per line. See
```stopwords/stopwords-en.txt``` and ```stopwords/stopwords-so.txt``` as example.

We provide concrete commands to run a demo example with the English-Somali language pair. We include the English-Somali 
parallel corpus from ParaCrawl in the `parallel_corpus/` directory for the demo example. To run the demo example, please 
run:
```bash
python preprocess_bitext.py --src_language so \
                            --en_bitext_path ../parallel_corpus/en-so/en-raw.txt \
                            --src_bitext_path ../parallel_corpus/en-so/so-raw.txt \
                            --en_stopwords_path ../stopwords/stopwords-en.txt \
                            --src_stopwords_path ../stopwords/stopwords-so.txt\
                            --output_preprocessed_en_bitext_path ../parallel_corpus/en-so/en-preprocessed.txt \
                            --output_preprocessed_src_bitext_path ../parallel_corpus/en-so/so-preprocessed.txt
```

## Build the vocabulary
We build the vocabulary set from the parallel bitext using the ```src/build_vocab.py``` This script will 
1. build the vocabulary of both English and the low-resource language from the bitext,
2. load the monolingual word embeddings of both English and the low-resource language, 
3. vectorize the bitext using the built vocabulary set.

Before running the script, please download the monolingual word embeddings of English and the low-resource language to 
folder `embeddings/`.

You can run the script by:
```bash
python build_vocab.py [-h] [--src_language SRC_LANGUAGE]
                      [--en_preprocessed_bitext_path EN_PREPROCESSED_BITEXT_PATH]
                      [--src_preprocessed_bitext_path SRC_PREPROCESSED_BITEXT_PATH]
                      [--en_embedding_path EN_EMBEDDING_PATH]
                      [--src_embedding_path SRC_EMBEDDING_PATH]
                      [--vocab_min_count VOCAB_MIN_COUNT]
                      [--output_dir OUTPUT_DIR]
```

To run the English-Somali demo example, please first download English monolingual word embeddings (from word2vec) to 
`embeddings/GoogleNews-vectors-negative300.bin` and Somali monolingual word embeddings (from fasttext) to 
`embeddings/cc.so.300.vec`.
```bash
python build_vocab.py --src_language so\
                      --en_preprocessed_bitext_path ../parallel_corpus/en-so/en-preprocessed.txt\
                      --src_preprocessed_bitext_path ../parallel_corpus/en-so/so-preprocessed.txt\
                      --en_embedding_path ../embeddings/GoogleNews-vectors-negative300.bin\
                      --src_embedding_path ../embeddings/cc.so.300.vec\
                      --vocab_min_count 3 --output_dir ../data/en-so/
```

## Augment the relevance dataset
To augment the relevance dataset based on parallel sentence corpus, please run the following script:
```bash
python generate_relevance_dataset.py [-h] [--data_dir DATA_DIR]
                                     [--device DEVICE]
                                     [--psq_filename PSQ_FILENAME]
                                     [--output_dir OUTPUT_DIR]
```

To run the English-Somali demo example on a GPU, please run
```bash
python generate_relevance_dataset.py --data_dir ../data/en-so/\
                                     --device cuda\
                                     --psq_filename ../PSQ_matrix/en_nonstem-so_nonstem.uni.json\
                                     --output_dir ../relevance_dataset/en-so/
```
which will store the augmented relevance dataset at `../relevance_dataset/en-so/`.


## Train the SECLR-RT relevance model
To train a SECLR-RT model on the augmented relevance dataset, please run the following command
```bash
python nn_experiment.py [-h] [--relevance_dataset_path RELEVANCE_DATASET_PATH]
                        [--en_embedding_path EN_EMBEDDING_PATH]
                        [--src_embedding_path SRC_EMBEDDING_PATH]
                        [--device DEVICE] [--output_dir OUTPUT_DIR]
```
To run the English-Somali demo example on a GPU, please run
```bash
python nn_experiment.py --relevance_dataset_path ../relevance_dataset/en-so/dataset.pkl\
                        --en_embedding_path ../data/en-so/en_embed.npy --src_embedding_path ../data/en-so/src_embed.npy\
                        --device cuda:0 --output_dir ../experiments/en-so/
```

## Predict with the trained model
We provide `predict_relevance_interface.py` as an interface that can predict cross-lingual relevance score with a
trained SECLR-RT model. To predict cross-lingual query relevance, please run the following command:
```bash
python predict_relevance_interface.py [-h] [--src_language SRC_LANGUAGE]
                                      [--vocab_dir VOCAB_DIR]
                                      [--model_path MODEL_PATH]
                                      [--device DEVICE]
```
To run the English-Somali demo example on a GPU, please run
```bash
python predict_relevance_interface.py --src_language so\
                                      --vocab_dir ../data/en-so/\
                                      --model_path ../experiments/en-so/model.pkl\
                                      --device cuda:0
```

## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at `yc3384@columbia.edu`.

## Citation

```bibtex
@inproceedings{chen-etal-2021-cross-language,
    title = "Cross-language Sentence Selection via Data Augmentation and Rationale Training",
    author = "Chen, Yanda  and
      Kedzie, Chris  and
      Nair, Suraj  and
      Galuscakova, Petra  and
      Zhang, Rui  and
      Oard, Douglas  and
      McKeown, Kathleen",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.300",
    doi = "10.18653/v1/2021.acl-long.300",
    pages = "3881--3895",
    abstract = "This paper proposes an approach to cross-language sentence selection in a low-resource setting. It uses data augmentation and negative sampling techniques on noisy parallel sentence data to directly learn a cross-lingual embedding-based query relevance model. Results show that this approach performs as well as or better than multiple state-of-the-art machine translation + monolingual retrieval systems trained on the same parallel data. Moreover, when a rationale training secondary objective is applied to encourage the model to match word alignment hints from a phrase-based statistical machine translation model, consistent improvements are seen across three language pairs (English-Somali, English-Swahili and English-Tagalog) over a variety of state-of-the-art baselines.",
}
```
