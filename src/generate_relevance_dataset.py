import random
import pickle as pkl
import numpy as np
import torch
import os
from util import to_device
import subprocess
from PSQ import PSQ


class Query_generator():
    def __init__(self, data_dir, device, output_dir, psq_filename, psq_cleaning_language=None,
                 use_percentage_data=1):
        self.corpus_en = pkl.load(open(data_dir + 'int_arr_en.pkl', 'rb'))
        self.corpus_src = pkl.load(open(data_dir + 'int_arr_src.pkl', 'rb'))
        self.en_id2word = pkl.load(open(data_dir + 'en_id2word.pkl', 'rb'))
        self.src_id2word = pkl.load(open(data_dir + 'src_id2word.pkl', 'rb'))

        # Saves relevant/irrelevant label between unigram English queries and English sentences
        self.en_embed = np.loadtxt(data_dir + "en_embed.npy")
        self.normalized_en_embed = to_device(torch.FloatTensor([embed / np.linalg.norm(embed) for embed in self.en_embed]),
                                             "%s" % device)

        # If using clean tokenized script for low-resource language, no need to specify psq_cleaning_language (should be None)
        # If not using clean tokenized script for low-resource language, psq_cleaning_language = low-resource language
        self.psq = PSQ(psq_filename=psq_filename, cleaning_language=psq_cleaning_language)

        self.output_dir = output_dir

        # Use how much percentage of the data for training
        # First split the data into train/val/test according to 0.96, 0.03, 0.01
        # Then choose a subset of training set of size use_percentage_data * training set size
        self.use_percentage_data = use_percentage_data


    def generate_positive_samples(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if os.path.exists('%s/sentence_pair_split_ids.pkl' % self.output_dir):
            self.sentence_pair_split_ids = pkl.load(open('%s/sentence_pair_split_ids.pkl' % self.output_dir, 'rb'))
        else:
            num_sentence_pairs = len(self.corpus_src)
            all_idx = np.random.permutation(list(range(num_sentence_pairs)))
            #Partition all sentence pairs to train, validation, test.
            train_proportion, test_porportion, val_proportion = 0.96, 0.03, 0.01
            train_val_boundary = int(train_proportion * num_sentence_pairs)
            val_test_boundary = int((train_proportion + val_proportion) * num_sentence_pairs)

            train_sentence_pair_ids = all_idx[: train_val_boundary]
            # Sample a portion of the training set sentence ids according to the use_percentage_data parameter
            sampled_train_size = int(len(train_sentence_pair_ids) * self.use_percentage_data)
            train_sentence_pair_ids = random.sample(list(train_sentence_pair_ids), k=sampled_train_size)

            val_sentence_pair_ids = all_idx[train_val_boundary: val_test_boundary]
            test_sentence_pair_ids = all_idx[val_test_boundary: ]
            self.sentence_pair_split_ids = {'train': train_sentence_pair_ids, 'val': val_sentence_pair_ids,
                                            'test': test_sentence_pair_ids}
            pkl.dump(self.sentence_pair_split_ids, open('%s/sentence_pair_split_ids.pkl' % self.output_dir, 'wb'))

        self.positive_data = {}
        for dataset in ['train', 'val', 'test']:
            pair_ids = self.sentence_pair_split_ids[dataset]
            sampled_en_queries = []
            sampled_relevant_src_sentences = []
            sampled_en_query_word_idxs = []
            sampled_en_query_sentence_ids = []
            sampled_rationale_distribution = []

            not_founds = []

            #Sample positive queries for all sentences
            for idx in range(len(pair_ids)):
                sentence_pair_id = pair_ids[idx]
                en_int_arr = self.corpus_en[sentence_pair_id]
                src_int_arr = self.corpus_src[sentence_pair_id]
                for word_idx in range(len(en_int_arr)):
                    if en_int_arr[word_idx] != 1: #not UNK
                        sampled_relevant_src_sentences.append(src_int_arr)
                        sampled_en_queries.append([en_int_arr[word_idx]])
                        sampled_en_query_word_idxs.append([word_idx])
                        sampled_en_query_sentence_ids.append(sentence_pair_id)

                        query_words = [self.en_id2word[en_int_arr[word_idx]]]
                        en_sentence = [self.en_id2word[word_id] for word_id in en_int_arr]
                        src_sentence = [self.src_id2word[word_id] for word_id in src_int_arr]
                        rationale_distr = self.psq.compute_gt_rationale_distribution(query_words, src_sentence)
                        sampled_rationale_distribution.append(rationale_distr)
                        if rationale_distr is None:
                            not_founds.append([query_words, en_sentence, src_sentence])
            pkl.dump(not_founds, open('%s/notfounds.pkl' % self.output_dir, 'wb'))

            sampled_data = {'en_queries': sampled_en_queries, 'en_queries_word_idxs': sampled_en_query_word_idxs,
                            'relevant_src_sentences': sampled_relevant_src_sentences,
                            'en_query_sentence_ids': sampled_en_query_sentence_ids,
                            'rationale_distr': sampled_rationale_distribution
                           }

            if dataset == 'train':
                psq_query_not_found_count = self.psq.query_not_exist_count / len(sampled_rationale_distribution)
                psq_translation_not_found_count = self.psq.translation_not_exist_count / len(sampled_rationale_distribution)
                with open('%s/README' % self.output_dir, 'w') as f:
                    f.write('Portion of query not found in PSQ: %.4f\n' % psq_query_not_found_count)
                    f.write('Portion of translation not found in PSQ: %.4f\n\n' % psq_translation_not_found_count)
                    f.write('Number of positive samples: %d\n' % len(sampled_en_query_sentence_ids))

            self.positive_data[dataset] = sampled_data
        pkl.dump(self.positive_data, open("%s/positive_samples.pkl" % self.output_dir, 'wb'))


    def generate_negative_samples(self):
        """
        In the previous datasets we notice that sometimes a source sentence is generated with only positive labels or
        negative labels and thus the model remembers the label of the source sentences instead of making comparison.
        In this dataset, we aim at having most of the source sentences appear as both positive and negative.
        We generate negative samples iteratively by pairing up the source sentences and the queries in the positive dataset.
        """
        for dataset in ['train', 'val', 'test']:
            en_queries = self.positive_data[dataset]['en_queries']
            pair_ids = self.positive_data[dataset]['en_query_sentence_ids']

            irrelevant_src_sentences = []
            irrelevant_sentence_ids = []
            all_sentence_ids = list(set(pair_ids))

            for idx in range(len(en_queries)):
                query = en_queries[idx]
                sentence_id = all_sentence_ids[random.randint(0, len(all_sentence_ids) - 1)]
                while self.check_query_en_sentence_relevance(query, sentence_id):
                    sentence_id = all_sentence_ids[random.randint(0, len(all_sentence_ids) - 1)]
                irrelevant_sentence_ids.append(sentence_id)
                irrelevant_src_sentences.append(self.corpus_src[sentence_id])
            self.positive_data[dataset]['irrelevant_src_sentence_ids'] = [irrelevant_sentence_ids]
            self.positive_data[dataset]['irrelevant_src_sentences'] = [irrelevant_src_sentences]
            pkl.dump(self.positive_data, open("%s/dataset.pkl" % self.output_dir, 'wb'))
        pkl.dump(self.positive_data, open("%s/dataset.pkl" % self.output_dir, 'wb'))


    def check_query_en_sentence_relevance(self, en_query, en_sentence_id, threshold=0.4):
        query_embedding = [self.normalized_en_embed[en_unigram] for en_unigram in en_query]
        en_sentence_embedding = torch.stack([self.normalized_en_embed[word_id] for word_id in self.corpus_en[en_sentence_id]])
        for unigram_embedding in query_embedding:
            similarity = torch.matmul(unigram_embedding, en_sentence_embedding.permute(dims=(1,0)))
            max_similarity = torch.max(similarity)
            if max_similarity >= threshold:
                return True
        return False


if __name__ == '__main__':
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