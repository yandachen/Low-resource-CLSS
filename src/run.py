from preprocess_bitext import preprocess_tokenize_raw_texts
from preprocess import Bilingual_preprocess
from generate_query import Query_generator
import numpy as np
from nn_experiment import NN_experiment
import os
from data_loader import load_data
from evaluation import evaluate


def train_model():
    lang = 'ka'
    vocab_min_count = 3     # set vocab_min_count for all experiments
    device = "cuda:0"

    # Getting started:
    # step 1, download stopwords and save as "stopwords/stopwords-xx.txt", where xx=language code
    # step 2, download cleaned parallel bitext to "parallel_corpus/cleaned_bitext.{en,xx}"
    # step 3, download PSQ matrix to "PSQ_matrix/xx/en_nonstem-xx_nonstem.uni.json"
    # step 4, update the normalization script on the server
    #         install with "python3 setup.py install --user" command
    #         test that the normalization script is ready: from updated_normalization.normalization import normalization
    # step 5, download fasttext embedding to "embeddings/cc.xx.300.vec"

    # Start running:
    # step 1, preprocessing
    preprocess_tokenize_raw_texts(lang)
    print('preprocessing bitext done.')

    experiment_name = '%s_mincount=%d' % (lang, vocab_min_count)

    preprocessor = Bilingual_preprocess(src_language=lang,
                                        en_corpus_dir='../parallel_corpus/%s/cleaned_bitext.en_nosw' % lang,
                                        src_corpus_dir='../parallel_corpus/%s/cleaned_bitext.%s_nosw' % (lang, lang),
                                        output_dir=experiment_name,
                                        vocab_min_count=vocab_min_count)
    preprocessor.preprocess_both_languages()
    print('preprocessing done.')

    # step 2, dataset generation
    data_generator = Query_generator(model_dir='../model/%s_mincount=%d/' % (lang, vocab_min_count),
                                     cuda_device=device,
                                     output_dir=experiment_name,
                                     psq_filename='../PSQ_matrix/%s/en_nonstem-%s_nonstem.uni.json' % (lang, lang))
    data_generator.generate_positive_samples()
    data_generator.generate_negative_samples()
    print('dataset generation done.')

    # step 3, train model



def evaluate_model():
    lang = 'ka'
    device = "cuda:0"
    experiment_name = '%s_mincount=2' % lang

    query_list_dir = "../CLIR_data/%s/speech_query_list.tsv" % lang
    document_folder_dir = "../CLIR_data/%s/speech" % lang
    output_dir = '../CLIR_results/%s/speech/' % experiment_name

    queries, documents = load_data(lang, query_list_dir, document_folder_dir)
    evaluate(experiment_name, queries, documents,
             output_dir=output_dir, device=device)

    query_list_dir = "../CLIR_data/%s/text_query_list.tsv" % lang
    document_folder_dir = "../CLIR_data/%s/text" % lang
    output_dir = '../CLIR_results/%s/text/' % experiment_name

    queries, documents = load_data(lang, query_list_dir, document_folder_dir)
    evaluate(experiment_name, queries, documents,
             output_dir=output_dir, device=device)

    import argparse

    parser = argparse.ArgumentParser(description='Augment a relevance dataset based on parallel bitext.')
    parser.add_argument('--data_dir', action='store', type=str,
                        help='data directory of the vocabulary set and vectorized bitext.')
    parser.add_argument('--device', action='store', type=str,
                        help='device used to perform data augmentation, e.g., "cuda:0"')
    parser.add_argument('--psq_filename', action='store', type=str,
                        help='input path of the PSQ word translation matrix.')
    parser.add_argument('--output_dir', action='store', type=str,
                        help='output directory of the augmented relevance dataset.')

    args = parser.parse_args()
    data_generator = Query_generator(data_dir=args.data_dir, device=args.device,
                                     psq_filename=args.psq_filename, output_dir=args.output_dir)
    data_generator.generate_positive_samples()
    data_generator.generate_negative_samples()

if __name__ == '__main__':
    train_model()
    evaluate_model()