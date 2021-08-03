from model_def import NN_architecture
from model_wrapper import Model
import pickle as pkl
import numpy as np
from sklearn.metrics import f1_score, classification_report
import os
import random


class NN_experiment():
    def __init__(self, experiment_dir, kwargs,
                 generated_natural_query_docs_dir=None,
                 num_epochs=40, patience=3, batch_size=1024, min_epochs=2, lr=0.001, embed_lr=0.001, device='cpu',
                 dataset="natural", optimizer='adam', embed_optimizer='sparse_adam',
                 en_num_operations=10000, src_num_operations=20000,
                 subword_embed_dim=300,
                 tokenized_text_dir=None, src_language=None,
                 train_rationale=True, lambda_attention=3, rationale_distr_equal=False,
                 start_from_epoch=None,
                 rationale_percentage=1,
                 ):
        self.kwargs = kwargs
        self.generated_natural_query_docs_dir = generated_natural_query_docs_dir
        self.experiment_dir = experiment_dir

        self.num_epochs, self.patience, self.batch_size, self.min_epochs, self.lr, self.embed_lr =\
            num_epochs, patience, batch_size, min_epochs, lr, embed_lr
        self.device = device
        self.optimizer, self.embed_optimizer = optimizer, embed_optimizer

        self.dataset = dataset
        if self.dataset == 'natural':
            assert self.generated_natural_query_docs_dir is not None

        if os.path.exists(self.experiment_dir) is True and not start_from_epoch:
            assert False
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir, exist_ok=True)

        self.train_rationale = train_rationale
        self.lambda_attention = lambda_attention
        self.rationale_distr_equal = rationale_distr_equal

        self.start_from_epoch = start_from_epoch
        self.rationale_percentage = rationale_percentage

        self.hyper_params = {'dataset': dataset,
                             'num_epochs': num_epochs, 'patience': patience,
                             'batch_size': batch_size, 'min_epochs': min_epochs, 'lr': lr, 'embed_lr': embed_lr,
                             'device': device,
                             'optimizer': optimizer, 'embed_optimizer': embed_optimizer,
                             'en_num_operations': en_num_operations, 'src_num_operations': src_num_operations,
                             'subword_embed_dim': subword_embed_dim,
                             'train_rationale': train_rationale, 'lambda_attention': lambda_attention, 'rationale_distr_equal': rationale_distr_equal}


    def load_train_test(self):
        data = pkl.load(open(self.generated_natural_query_docs_dir, 'rb'))
        features, labels = {}, {}
        for dataset in ['train', 'val', 'test']:
            dataset_features, dataset_labels = [], []
            irrelevant_src_sentences = data[dataset]['irrelevant_src_sentences']
            en_queries = data[dataset]['en_queries']
            irrelevant_src_sentence_ids = data[dataset]['irrelevant_src_sentence_ids']
            en_queries_word_idxs = data[dataset]['en_queries_word_idxs']
            en_query_sentence_ids = data[dataset]['en_query_sentence_ids']
            relevant_src_sentences = data[dataset]['relevant_src_sentences']
            rationale_distrs = data[dataset]['rationale_distr']

            for _ in range(len(irrelevant_src_sentence_ids)):
                one_epoch_features, one_epoch_labels = [], []
                for idx in range(len(en_queries)):
                    feature = {'src': irrelevant_src_sentences[_][idx],
                               'query': en_queries[idx],
                               }

                    one_epoch_features.append(feature)
                    one_epoch_labels.append(0)

                    feature = {'src': relevant_src_sentences[idx],
                               'query': en_queries[idx]}

                    if self.train_rationale:
                        if rationale_distrs[idx] is not None:
                            if random.random() < self.rationale_percentage:
                                if not self.rationale_distr_equal:
                                    feature['rationale_distr'] = rationale_distrs[idx]
                                else:
                                    rationale_distrs[idx] = np.array(rationale_distrs[idx])
                                    feature['rationale_distr'] = (rationale_distrs[idx] != 0) / sum(rationale_distrs[idx] != 0)

                    one_epoch_features.append(feature)
                    one_epoch_labels.append(1)

                dataset_features.append(one_epoch_features)
                dataset_labels.append(one_epoch_labels)

            features[dataset] = dataset_features
            labels[dataset] = dataset_labels

        # Do not need to sample multiple rounds (epochs) of different negative samples for validation/testing.
        features['val'] = features['val'][0]
        labels['val'] = labels['val'][0]
        features['test'] = features['test'][0]
        labels['test'] = labels['test'][0]
        self.train_features, self.val_features, self.test_features = features['train'], features['val'], features['test']
        self.train_labels, self.val_labels, self.test_labels = labels['train'], labels['val'], labels['test']

        data = {'val': (self.val_features, self.val_labels), 'test': (self.test_features, self.test_labels)}
        pkl.dump(data, open(self.experiment_dir + 'data.pkl', 'wb'))


    def train_eval_model(self):
        """
        Train the model and evaluate on validation/test set.
        """
        pkl.dump(self.train_features, open(self.experiment_dir + 'original_train_features.pkl', 'wb'))
        pkl.dump(self.val_features, open(self.experiment_dir + 'original_val_features.pkl', 'wb'))
        pkl.dump(self.test_features, open(self.experiment_dir + 'original_test_features.pkl', 'wb'))

        with open(self.experiment_dir + 'README', 'a') as f:
            f.write("Experiment Training Parameters: \n")
            f.write(str(self.hyper_params) + '\n\n')

        self.kwargs['en_pad_word_id'] = 0

        if not self.train_rationale:
            self.model = NN_architecture(**self.kwargs)
        else:
            self.model = NN_architecture(**self.kwargs, return_rationale=True)


        if not self.start_from_epoch:
            start_from_model_dir = None
        else:
            start_from_model_dir = self.experiment_dir + 'model_epoch%d.pkl' % self.start_from_epoch

        model_wrap = Model(model=self.model, mode='train',
                           train_X=self.train_features, train_y=self.train_labels,
                           dev_X=self.val_features, dev_y=self.val_labels,
                           experiment_dir=self.experiment_dir,
                           num_epochs=self.num_epochs, patience=self.patience, batch_size=self.batch_size,
                           min_epochs=self.min_epochs, lr=self.lr, embed_lr=self.embed_lr, device=self.device,
                           optimizer=self.optimizer, embed_optimizer=self.embed_optimizer,
                           train_rationale=self.train_rationale, lambda_attention=self.lambda_attention,
                           start_from_epoch=self.start_from_epoch, start_from_model_dir=start_from_model_dir
                           )

        num_epochs_until_optimal, train_loss_at_optimal = model_wrap.train()
        if self.train_rationale:
            self.model = NN_architecture(**self.kwargs) #In evaluation process, model does not return attention weights.
        model_wrap = Model(model=self.model, mode='eval',
                           batch_size=self.batch_size,
                           weight_dir=self.experiment_dir + 'model.pkl', device=self.device,
                           )

        #tune the threshold on validation set
        val_pred_scores = model_wrap.predict(self.val_features)
        max_f1 = 0
        best_threshold = 0
        for threshold in np.arange(0, 1, 0.01):
            val_pred_labels = [1 if val_pred_scores[idx] >= threshold else 0 for idx in range(len(val_pred_scores))]
            f1 = f1_score(self.val_labels, val_pred_labels, average='macro')
            if f1 > max_f1:
                max_f1 = f1
                best_threshold = threshold

        #evaluate label f1 on val set
        val_pred_scores = model_wrap.predict(self.val_features)
        pkl.dump(val_pred_scores, open(self.experiment_dir + "val_pred_scores.pkl", 'wb'))
        val_pred_labels = [1 if val_pred_scores[idx] >= best_threshold else 0 for idx in range(len(val_pred_scores))]
        val_f1 = f1_score(self.val_labels, val_pred_labels, average='macro')
        pkl.dump(val_pred_labels, open(self.experiment_dir + "val_pred_labels.pkl", 'wb'))

        #evaluate label f1 on test set
        test_pred_scores = model_wrap.predict(self.test_features)
        pkl.dump(test_pred_scores, open(self.experiment_dir + "test_pred_scores.pkl", 'wb'))
        test_pred_labels = [1 if test_pred_scores[idx] >= best_threshold else 0 for idx in range(len(test_pred_scores))]
        test_f1 = f1_score(self.test_labels, test_pred_labels, average='macro')
        pkl.dump(test_pred_labels, open(self.experiment_dir + 'test_pred_labels.pkl', 'wb'))

        val_loss = model_wrap.compute_score_loss(self.val_features, self.val_labels)
        test_loss = model_wrap.compute_score_loss(self.test_features, self.test_labels)

        original_train_features = pkl.load(open(self.experiment_dir + 'original_train_features.pkl', 'rb'))
        original_val_features = pkl.load(open(self.experiment_dir + 'original_val_features.pkl', 'rb'))
        original_test_features = pkl.load(open(self.experiment_dir + 'original_test_features.pkl', 'rb'))

        # Calculate the f1-score across query lengths
        val_ngram_query_idx = {1: [], 2: [], 3: [], 4: []}
        for idx in range(len(self.val_features)):
            val_ngram_query_idx[len(original_val_features[idx]['query'])].append(idx)

        test_ngram_query_idx = {1: [], 2: [], 3: [], 4: []}
        for idx in range(len(self.test_features)):
            test_ngram_query_idx[len(original_test_features[idx]['query'])].append(idx)

        scores = {'threshold': best_threshold, 'val_loss': val_loss, 'test_loss': test_loss,
                  'val_f1': val_f1, 'test_f1': test_f1,
                  'num_epochs_optimal': num_epochs_until_optimal, 'train_loss_at_optimal': train_loss_at_optimal,
                  }

        val_f1_by_query_length = []
        test_f1_by_query_length = []

        for ngram in range(1, 5):
            val_pred_sub_scores = [val_pred_scores[idx] for idx in val_ngram_query_idx[ngram]]
            val_pred_sub_labels = [1 if s >= best_threshold else 0 for s in val_pred_sub_scores]
            val_sub_labels = [self.val_labels[idx] for idx in val_ngram_query_idx[ngram]]
            val_f1_by_query_length.append(f1_score(val_sub_labels, val_pred_sub_labels, average='macro'))

            test_pred_sub_scores = [test_pred_scores[idx] for idx in test_ngram_query_idx[ngram]]
            test_pred_sub_labels = [1 if s >= best_threshold else 0 for s in test_pred_sub_scores]
            test_sub_labels = [self.test_labels[idx] for idx in test_ngram_query_idx[ngram]]
            test_f1_by_query_length.append(f1_score(test_sub_labels, test_pred_sub_labels, average='macro'))

        scores['val_f1_by_query_length'] = val_f1_by_query_length
        scores['test_f1_by_query_length'] = test_f1_by_query_length
        scores['model_architecture'] = self.model.model_property

        scores['val_query_lengths_distribution'] = [len(val_ngram_query_idx[length]) / len(self.val_features)
                                                    for length in [1, 2, 3, 4]]
        scores['test_query_lengths_distribution'] = [len(test_ngram_query_idx[length]) / len(self.test_features)
                                                     for length in [1, 2, 3, 4]]

        with open(self.experiment_dir + 'README', 'a') as f:
            f.write('\nResults: Test on %d samples.\n' % len(self.test_features))
            f.write('threshold: %.3f\n' % best_threshold)
            f.write('Validation loss: %.3f\n' % val_loss)
            f.write('Validation label f1: %.1f\n' % (val_f1 * 100))
            f.write('Validation label f1 by query length: %.1f %.1f %.1f %.1f\n' % (val_f1_by_query_length[0] * 100,
                                                                               val_f1_by_query_length[1] * 100,
                                                                               val_f1_by_query_length[2] * 100,
                                                                               val_f1_by_query_length[3] * 100))
            f.write(classification_report(self.val_labels, val_pred_labels) + '\n')

            f.write('Testing loss: %.3f\n' % test_loss)
            f.write('Testing label f1: %.1f\n' % (test_f1 * 100))
            f.write('Testing label f1 by query length: %.1f %.1f %.1f %.1f\n' % (test_f1_by_query_length[0] * 100,
                                                                            test_f1_by_query_length[1] * 100,
                                                                            test_f1_by_query_length[2] * 100,
                                                                            test_f1_by_query_length[3] * 100))
            f.write(classification_report(self.test_labels, test_pred_labels) + '\n')

        # Calculate f-score by seen/unseen queries
        train_queries = set()
        train_unigram_queries = set()
        for epoch_train_features in self.train_features[: num_epochs_until_optimal]:
            train_queries = train_queries.union(set([tuple(feature['query']) for feature in epoch_train_features]))
            for feature in epoch_train_features:
                for unigram in feature['query']:
                    train_unigram_queries.add(unigram)

        pkl.dump(train_queries, open(self.experiment_dir + "train_queries.pkl", 'wb'))
        pkl.dump(train_unigram_queries, open(self.experiment_dir + "train_query_unigrams.pkl", 'wb'))

        val_queries = [feature['query'] for feature in original_val_features]
        test_queries = [feature['query'] for feature in original_test_features]

        self.val_query_unseen_idx, self.test_query_unseen_idx = [], []
        self.val_query_unigram_unseen_idx, self.test_query_unigram_unseen_idx = [], []

        for idx in range(len(val_queries)):
            if tuple(val_queries[idx]) not in train_queries:
                self.val_query_unseen_idx.append(idx)
            for unigram in val_queries[idx]:
                if unigram not in train_unigram_queries:
                    self.val_query_unigram_unseen_idx.append(idx)
                    break

        for idx in range(len(test_queries)):
            if tuple(test_queries[idx]) not in train_queries:
                self.test_query_unseen_idx.append(idx)
            for unigram in test_queries[idx]:
                if unigram not in train_unigram_queries:
                    self.test_query_unigram_unseen_idx.append(idx)
                    break

        # Seen queries vs unseen queries
        val_seen_true_labels, val_seen_pred_labels, val_unseen_true_labels, val_unseen_pred_labels = [], [], [], []
        for idx in range(len(val_pred_labels)):
            if idx in self.val_query_unseen_idx:
                val_unseen_pred_labels.append(val_pred_labels[idx])
                val_unseen_true_labels.append(self.val_labels[idx])
            else:
                val_seen_pred_labels.append(val_pred_labels[idx])
                val_seen_true_labels.append(self.val_labels[idx])
        val_unseen_fscore = f1_score(val_unseen_true_labels, val_unseen_pred_labels, average='macro')
        val_seen_fscore = f1_score(val_seen_true_labels, val_seen_pred_labels, average='macro')

        test_seen_true_labels, test_seen_pred_labels, test_unseen_true_labels, test_unseen_pred_labels = [], [], [], []
        for idx in range(len(test_pred_labels)):
            if idx in self.test_query_unseen_idx:
                test_unseen_pred_labels.append(test_pred_labels[idx])
                test_unseen_true_labels.append(self.test_labels[idx])
            else:
                test_seen_pred_labels.append(test_pred_labels[idx])
                test_seen_true_labels.append(self.test_labels[idx])
        test_unseen_fscore = f1_score(test_unseen_true_labels, test_unseen_pred_labels, average='macro')
        test_seen_fscore = f1_score(test_seen_true_labels, test_seen_pred_labels, average='macro')

        # Queries with/without unseen unigrams
        val_seen_true_labels, val_seen_pred_labels, val_unseen_true_labels, val_unseen_pred_labels = [], [], [], []
        for idx in range(len(val_pred_labels)):
            if idx in self.val_query_unigram_unseen_idx:
                val_unseen_pred_labels.append(val_pred_labels[idx])
                val_unseen_true_labels.append(self.val_labels[idx])
            else:
                val_seen_pred_labels.append(val_pred_labels[idx])
                val_seen_true_labels.append(self.val_labels[idx])
        val_with_unseen_unigram_fscore = f1_score(val_unseen_true_labels, val_unseen_pred_labels, average='macro')
        val_with_seen_unigram_fscore = f1_score(val_seen_true_labels, val_seen_pred_labels, average='macro')

        test_seen_true_labels, test_seen_pred_labels, test_unseen_true_labels, test_unseen_pred_labels = [], [], [], []
        for idx in range(len(test_pred_labels)):
            if idx in self.test_query_unigram_unseen_idx:
                test_unseen_pred_labels.append(test_pred_labels[idx])
                test_unseen_true_labels.append(self.test_labels[idx])
            else:
                test_seen_pred_labels.append(test_pred_labels[idx])
                test_seen_true_labels.append(self.test_labels[idx])
        test_with_unseen_unigram_fscore = f1_score(test_unseen_true_labels, test_unseen_pred_labels, average='macro')
        test_with_seen_unigram_fscore = f1_score(test_seen_true_labels, test_seen_pred_labels, average='macro')

        # Unseen/seen unigram queries
        val_seen_true_labels, val_seen_pred_labels, val_unseen_true_labels, val_unseen_pred_labels = [], [], [], []
        for idx in range(len(val_pred_labels)):
            query = val_queries[idx]
            if len(query) == 1:
                query_word = query[0]
                if query_word in train_unigram_queries:
                    val_seen_true_labels.append(self.val_labels[idx])
                    val_seen_pred_labels.append(val_pred_labels[idx])
                else:
                    val_unseen_true_labels.append(self.val_labels[idx])
                    val_unseen_pred_labels.append(val_pred_labels[idx])
        val_unseen_unigram_fscore = f1_score(val_unseen_true_labels, val_unseen_pred_labels, average='macro')
        val_seen_unigram_fscore = f1_score(val_seen_true_labels, val_seen_pred_labels, average='macro')

        test_seen_true_labels, test_seen_pred_labels, test_unseen_true_labels, test_unseen_pred_labels = [], [], [], []
        for idx in range(len(test_pred_labels)):
            query = test_queries[idx]
            if len(query) == 1:
                query_word = query[0]
                if query_word in train_unigram_queries:
                    test_seen_true_labels.append(self.test_labels[idx])
                    test_seen_pred_labels.append(test_pred_labels[idx])
                else:
                    test_unseen_true_labels.append(self.test_labels[idx])
                    test_unseen_pred_labels.append(test_pred_labels[idx])
        test_unseen_unigram_fscore = f1_score(test_unseen_true_labels, test_unseen_pred_labels, average='macro')
        test_seen_unigram_fscore = f1_score(test_seen_true_labels, test_seen_pred_labels, average='macro')

        with open(self.experiment_dir + "README", 'a') as f:
            f.write('Unseen Query Analysis:\n')
            f.write("%.2f%% of the queries are unseen on validation set.\n" %
                    (len(self.val_query_unseen_idx) / len(self.val_features) * 100))
            f.write("%.2f%% of the queries are unseen on test set.\n" %
                    (len(self.test_query_unseen_idx) / len(self.test_features) * 100))
            f.write('Validation unseen queries f-score: %.1f\n' % (val_unseen_fscore * 100))
            f.write('Validation seen queries f-score: %.1f\n' % (val_seen_fscore * 100))
            f.write('Test unseen queries f-score: %.1f\n' % (test_unseen_fscore * 100))
            f.write('Test seen queries f-score: %.1f\n' % (test_seen_fscore * 100))

            f.write('Query with unseen unigram Analysis:\n')
            f.write("%.2f%% of the queries have unseen unigrams on validation set.\n" %
                    (len(self.val_query_unigram_unseen_idx) / len(self.val_features) * 100))
            f.write("%.2f%% of the queries have unseen unigrams on test set.\n" %
                    (len(self.test_query_unigram_unseen_idx) / len(self.test_features) * 100))
            f.write('Validation queries with unseen unigram f-score: %.1f\n' % (val_with_unseen_unigram_fscore * 100))
            f.write('Validation queries without unseen unigram f-score: %.1f\n' % (val_with_seen_unigram_fscore * 100))
            f.write('Test queries with unseen unigram f-score: %.1f\n' % (test_with_unseen_unigram_fscore * 100))
            f.write('Test queries without unseen unigram f-score: %.1f\n' % (test_with_seen_unigram_fscore * 100))

            f.write('Unseen Unigram Query Analysis:\n')
            f.write('Validation unseen unigram queries f-score: %.1f\n' % (val_unseen_unigram_fscore * 100))
            f.write('Validation seen unigram queries f-score: %.1f\n' % (val_seen_unigram_fscore * 100))
            f.write('Test unseen unigram queries f-score: %.1f\n' % (test_unseen_unigram_fscore * 100))
            f.write('Test seen unigram queries f-score: %.1f\n' % (test_seen_unigram_fscore * 100))

        scores['val_unseen_fscore'], scores['val_seen_fscore'], scores['test_unseen_fscore'], scores['test_seen_fscore'] \
            = val_unseen_fscore, val_seen_fscore, test_unseen_fscore, test_seen_fscore
        scores['val_with_unseen_unigram_fscore'], scores['val_with_seen_unigram_fscore'], \
        scores['test_with_unseen_unigram_fscore'], scores['test_with_seen_unigram_fscore'] \
            = val_with_unseen_unigram_fscore, val_with_seen_unigram_fscore, test_with_unseen_unigram_fscore, test_with_seen_unigram_fscore
        scores['val_unseen_unigram_fscore'], scores['val_seen_unigram_fscore'], scores['test_unseen_unigram_fscore'], scores['test_seen_unigram_fscore'] \
            = val_unseen_unigram_fscore, val_seen_unigram_fscore, test_unseen_unigram_fscore, test_seen_unigram_fscore
        pkl.dump(scores, open(self.experiment_dir + 'result.pkl', 'wb'))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a SECLR-RT model using the augmented relevance dataset.')
    parser.add_argument('--relevance_dataset_path', action='store', type=str,
                        help='path of the augmented relevance dataset.')
    parser.add_argument('--en_embedding_path', action='store', type=str,
                        help='path of the loaded English word embeddings in numpy format.')
    parser.add_argument('--src_embedding_path', action='store', type=str,
                        help='path of the loaded src-language word embeddings in numpy format.')
    parser.add_argument('--device', action='store', type=str,
                        help='device for training, e.g., "cuda:0"')
    parser.add_argument('--output_dir', action='store', type=str,
                        help='output directory for model training.')

    args = parser.parse_args()
    kwargs = {'en_embedding_matrix': np.loadtxt(args.en_embedding_path),
              'src_embedding_matrix': np.loadtxt(args.src_embedding_path)}

    experiment = NN_experiment(experiment_dir=args.output_dir,
                               kwargs=kwargs,
                               generated_natural_query_docs_dir=args.relevance_dataset_path,
                               device=args.device)
    experiment.load_train_test()
    experiment.train_eval_model()