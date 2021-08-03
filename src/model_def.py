from torch import nn
from torch.nn import Embedding
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
import time
from util import to_device


class NN_architecture(nn.Module):
    """
    LSTM-attention based model architecture for sentence-level relevance between Lithuanian source sentence and English
    query.
    """
    def __init__(self, en_input_format='word', src_input_format='word',
                 en_embedding_matrix=None, src_embedding_matrix=None,
                 embed_dropout=0.2, transform_embedding=False, embedding_transformed_dim=None, embedding_sparse=True,
                 en_pad_word_id=0, src_pad_word_id=0,
                 query_use_lstm=False, query_lstm_hidden=128, query_lstm_dropout=0.2,
                 src_use_lstm=False, src_lstm_hidden=128, src_lstm_dropout=0.2,
                 subword_lstm_hidden=150,
                 compose_subwords_first=False,
                 return_rationale=False,
                 hubness_normalization=False, hubness_normalized_constant=0.5,
                 embedding_length_normalization=False,
                 idf_normalization=False, src_wordid2frequency=None, idf_normalization_constant=1,
                 ):

        super(NN_architecture, self).__init__()
        self.en_input_format, self.src_input_format = en_input_format, src_input_format

        self.transform_embedding = transform_embedding
        if not self.transform_embedding:
            assert embedding_transformed_dim is None

        if self.transform_embedding:
            freeze_embedding = True
            embedding_sparse = False
        else:
            freeze_embedding = False
            embedding_sparse = embedding_sparse

        self.en_embed = Embedding.from_pretrained(torch.from_numpy(en_embedding_matrix), freeze=freeze_embedding, sparse=embedding_sparse)
        self.src_embed = Embedding.from_pretrained(torch.from_numpy(src_embedding_matrix), freeze=freeze_embedding, sparse=embedding_sparse)

        embedding_dim = en_embedding_matrix.shape[1]

        if self.transform_embedding:
            assert embedding_transformed_dim is not None
            self.embedding_transformed_dim = embedding_transformed_dim
            self.en_embed_transform = nn.Linear(embedding_dim, embedding_transformed_dim)
            self.src_embed_transform = nn.Linear(embedding_dim, embedding_transformed_dim)
            self.embedding_output_dim = embedding_transformed_dim
        else:
            self.embedding_output_dim = embedding_dim

        self.en_pad_word_id, self.src_pad_word_id = en_pad_word_id, src_pad_word_id

        self.en_embed_dropout = nn.Dropout(embed_dropout)
        self.src_embed_dropout = nn.Dropout(embed_dropout)

        if self.en_input_format != 'word':
            self.subword_lstm_hidden = subword_lstm_hidden
            self.subword_en_lstm = nn.LSTM(input_size=self.embedding_output_dim, hidden_size=self.subword_lstm_hidden, bidirectional=True)
        if self.src_input_format != 'word':
            self.subword_lstm_hidden = subword_lstm_hidden
            self.subword_src_lstm = nn.LSTM(input_size=self.embedding_output_dim, hidden_size=self.subword_lstm_hidden, bidirectional=True)

        self.model_property = {'en_input_format': en_input_format, 'src_input_format': src_input_format,
                               'embed_dropout': embed_dropout, 'embedding_sparse': embedding_sparse,
                               'transform_embedding': transform_embedding,
                               'embedding_transformed_dim': embedding_transformed_dim,
                               'query_use_lstm': query_use_lstm, 'src_use_lstm': src_use_lstm,
                               }
        if self.en_input_format != 'word' or self.src_input_format != 'word':
            self.model_property['subword_lstm_hidden'] = subword_lstm_hidden

        if self.en_input_format == 'word':
            self.en_lstm_input_dim = self.embedding_output_dim
        else:
            self.en_lstm_input_dim = self.subword_lstm_hidden * 2
        if self.src_input_format == 'word':
            self.src_lstm_input_dim = self.embedding_output_dim
        else:
            self.src_lstm_input_dim = self.subword_lstm_hidden * 2

        self.query_use_lstm, self.src_use_lstm = query_use_lstm, src_use_lstm
        if self.query_use_lstm:
            self.query_output_dim = query_lstm_hidden * 2 #2 for bidirectional LSTM
            self.query_lstm = nn.LSTM(input_size=self.en_lstm_input_dim, hidden_size=query_lstm_hidden, bidirectional=True)
            self.query_dropout = nn.Dropout(query_lstm_dropout)
            self.model_property['query_lstm_hidden'] = query_lstm_hidden
            self.model_property['query_lstm_dropout'] = query_lstm_dropout
        else:
            self.query_output_dim = self.en_lstm_input_dim

        if self.src_use_lstm:
            self.src_output_dim = src_lstm_hidden * 2
            self.src_lstm = nn.LSTM(input_size=self.src_lstm_input_dim, hidden_size=src_lstm_hidden, bidirectional=True)
            self.src_dropout = nn.Dropout(src_lstm_dropout)
            self.model_property['src_lstm_hidden'] = src_lstm_hidden
            self.model_property['src_lstm_dropout'] = src_lstm_dropout
        else:
            self.src_output_dim = self.src_lstm_input_dim

        assert self.src_output_dim == self.query_output_dim

        self.return_rationale = return_rationale

        if self.en_input_format != 'word' or self.src_input_format != 'word':
            # if set to True, model uses a LSTM for each word to get the word-level meaning and then compare word pairwise
            # if set to False, model directly compares the subwords pairwise
            self.compose_subword_first = compose_subwords_first

        # if set to true, the embeddings will be first normalized to l2-norm = 1 before operations.
        self.embedding_length_normalization = embedding_length_normalization

        # If hubness_normalization set to true (usually used in inference time), then apply CSLS normalization to
        # reduce the effect of the hubness problem.
        # Under CSLS normalization sim(x, y) = 2cos(x, y) - r(x) - r(y) where r(x) and r(y) is average cosine similarity
        # of x and y for their k nearest neighbors in the other language respectively. Take k=10 following other authors.
        self.hubness_normalization = hubness_normalization
        if self.hubness_normalization:
            # only use CSLS normalization when no lstm is used on either the query side or the source sentence side.
            assert not self.src_use_lstm and not self.query_use_lstm
            # only use CSLS normalization during inference time
            assert not self.return_rationale
            self.hubness_normalized_constant = hubness_normalized_constant

            # preprocess to compute the hubness scores of each individual word
            self.csls_hubness_normalization()


        # if set to true, normalized by inverse-document frequency
        self.idf_normalization = idf_normalization

        if self.idf_normalization:
            # IDF-normalization only used during inference testing not training
            assert not self.return_rationale
            self.src_wordid2frequency = src_wordid2frequency
            # If IDF-normalization is used, then must specify source language vocabulary frequency
            assert self.src_wordid2frequency is not None
            self.idf_normalization_constant = idf_normalization_constant


    def forward(self, X, mode):
        if self.en_input_format == 'word':
            query_rep, query_lengths = self.forward_word_format(X, language='en')
            # query_rep shape (max query length, batch size, embed dimension)
        else:
            query_rep, query_lengths = self.forward_subword_format(X, language='en')

        if self.src_input_format == 'word':
            src_rep, src_lengths = self.forward_word_format(X, language='src')
            # src_rep shape (max src length, batch size, embed dimension)
        else:
            src_rep, src_lengths = self.forward_subword_format(X, language='src')

        query_rep = query_rep.permute(dims=(1, 0, 2))  # shape (batch size, max query length, embed dimension)
        src_rep = src_rep.permute(dims=(1, 2, 0))  # shape (batch size, embed dimension, max src length)
        comparison_scores = torch.matmul(query_rep, src_rep)  # shape (batch size, max query length, max src length)

        if self.embedding_length_normalization:
            query_norms = torch.norm(query_rep, 2, 2).unsqueeze(dim=2) # shape (batch size, max query length, 1)
            src_norms = torch.norm(src_rep, 2, 1).unsqueeze(dim=1) # shape (batch size, 1, max src length)
            norms_product = torch.matmul(query_norms, src_norms)
            eps = torch.FloatTensor([10 ** -8])
            comparison_scores = torch.div(comparison_scores, torch.max(norms_product,
                                                                       to_device(eps.expand_as(norms_product), comparison_scores.get_device())))

        relev_scores = []
        if self.return_rationale:
            rationale = []

        assert query_rep.shape[0] == len(X)
        assert mode in ['train', 'test']
        if mode == 'train':
            # compute scores on torch tensors so that backprop can compute correctly.
            for idx in range(query_rep.shape[0]):
                scores = comparison_scores[idx][: query_lengths[idx], : src_lengths[idx]]  # shape (actual query length, actual src length)
                relev_score = torch.min(torch.max(scores, dim=1)[0], keepdim=True, dim=0)[0]
                relev_scores.append(relev_score)
                if self.return_rationale:
                    rationale.append(torch.softmax(scores[0], dim=0))

            relev_scores = torch.stack(relev_scores)
            if not self.return_rationale:
                return torch.sigmoid(relev_scores)  # Format [[0.9904], [0.9741],[0.1198]]
            else:
                return torch.sigmoid(relev_scores), rationale

        elif mode == 'test':
            # First transform tensor to numpy because row-wise/column-wise slicing/maximum/minimum operation are faster
            # on numpy matrices than torch tensors
            assert not self.return_rationale # return_rationale cannot be used in inference stage.

            comparison_scores = comparison_scores.cpu().numpy()

            for idx in range(query_rep.shape[0]):
                scores = comparison_scores[idx][: query_lengths[idx],: src_lengths[idx]]  # shape (actual query length, actual src length)

                if self.hubness_normalization:
                    query_ids, src_ids = X[idx]['query'].cpu().numpy(), X[idx]['src'].cpu().numpy()
                    for query_idx in range(scores.shape[0]):
                        scores[query_idx] -= self.hubness_normalized_constant * self.hubness_scores_en[query_ids[query_idx]]
                    for src_idx in range(scores.shape[1]):
                        scores[:, src_idx] -= self.hubness_normalized_constant * self.hubness_scores_src[src_ids[src_idx]]

                if self.idf_normalization:
                    src_ids = X[idx]['src'].cpu().numpy()
                    for src_idx in range(scores.shape[1]):
                        scores[:, src_idx] = scores[:, src_idx] / (self.src_wordid2frequency[src_ids[src_idx]] ** self.idf_normalization_constant)

                relev_score = np.min(np.max(scores, axis=1), axis=0)
                relev_scores.append(relev_score)

            relev_scores = np.array(relev_scores)
            return 1 / (1 + np.exp(-relev_scores)) # format [0.3, 0.2, 0.9], numpy array



    def forward_word_format(self, X, language):
        """
        The Lithuanian source sentence and English query are each based through a LSTM network to get its representation.
        An attention mechanism is used across the source sentence, where attention weights are calculated based on
        [query LSTM final state, source sentence LSTM hidden state].
        The weighted attended representation is then concatenated with the query-side LSTM's final state, fed through a
        dense layer and sigmoid to compute whether the query and src language are relevant.
        Note that the attention weights do not necessarily sum up to one, because we want the attention weight to also
        represent the level of relevance between each Lithuanian token and the English query.
        """
        # Input format of X is list of dict of tensors
        # We need to pad the sequences for embedding layer even if we do not use LSTM.
        if language == 'en':
            query_lengths = [len(x['query']) for x in X]
            queries = pad_sequence([x['query'] for x in X], padding_value=self.en_pad_word_id) # shape (max query length, batch size)

            if not self.transform_embedding:
                query_rep = self.en_embed_dropout(self.en_embed(queries).float())  # shape (max query length, batch size, embed dimension)
            else:
                query_rep = self.en_embed_dropout(self.en_embed_transform(self.en_embed(queries).float()))

            if self.query_use_lstm:
                query_hs, _ = self.pack_batch_lstm(query_rep, query_lengths, self.query_lstm)
                query_rep = self.query_dropout(query_hs)

            return query_rep, query_lengths

        if language == 'src':
            src_lengths = [len(x['src']) for x in X]
            srcs = pad_sequence([x['src'] for x in X], padding_value=self.src_pad_word_id)

            if not self.transform_embedding:
                src_rep = self.src_embed_dropout(self.src_embed(srcs).float())  # shape (max src length, batch size, embed dimension)
            else:
                src_rep = self.src_embed_dropout(self.src_embed_transform(self.src_embed(srcs).float()))

            if self.src_use_lstm:
                src_hs, _ = self.pack_batch_lstm(src_rep, src_lengths, self.src_lstm)
                src_rep = self.src_dropout(src_hs)

            return src_rep, src_lengths


    def csls_hubness_normalization(self, k=10):
        en_vocab = list(range(self.en_embed.weight.shape[0]))
        src_vocab = list(range(self.src_embed.weight.shape[0]))

        en_embedding_vals = self.en_embed(torch.LongTensor(en_vocab)).float() # shape (number of en vocabs, embed dimension)
        src_embedding_vals = self.src_embed(torch.LongTensor(src_vocab)).float().permute(dims=(1, 0)) # shape (embed dimension, number of src vocabs)
        pairwise_scores = torch.matmul(en_embedding_vals, src_embedding_vals) # shape (number of en vocabs, number of src vocabs)

        if self.embedding_length_normalization:
            query_norms = torch.norm(en_embedding_vals, 2, 1).unsqueeze(dim=1)  # shape (number of en vocabs, 1)
            src_norms = torch.norm(src_embedding_vals, 2, 0).unsqueeze(dim=1)  # shape (number of src vocabs, 1)
            pairwise_scores= torch.div(pairwise_scores, query_norms)
            pairwise_scores = torch.div(pairwise_scores, src_norms)

        # compute hubness score for each english word
        hubness_scores_en = [0, ] # initialize hubness score 0 for the PAD token
        for word_id in range(1, len(en_vocab)):
            scores = pairwise_scores[word_id, 1:] # exclude PAD in computing hubness score
            hubness_scores_en.append(torch.mean(torch.topk(scores, k=k, sorted=False)[0]).item())

        # compute hubness score for each source language word
        hubness_scores_src = [0, ]  # initialize hubness score 0 for the PAD token
        for word_id in range(1, len(src_vocab)):
            scores = pairwise_scores[1:, word_id]  # exclude PAD in computing hubness score
            hubness_scores_src.append(torch.mean(torch.topk(scores, k=k, sorted=False)[0]).item())

        self.hubness_scores_en, self.hubness_scores_src = hubness_scores_en, hubness_scores_src


    def forward_subword_format(self, X, language):
        """
        The Lithuanian source sentence and English query are each based through a LSTM network to get its representation.
        An attention mechanism is used across the source sentence, where attention weights are calculated based on
        [query LSTM final state, source sentence LSTM hidden state].
        The weighted attended representation is then concatenated with the query-side LSTM's final state, fed through a
        dense layer and sigmoid to compute whether the query and src language are relevant.
        Note that the attention weights do not necessarily sum up to one, because we want the attention weight to also
        represent the level of relevance between each Lithuanian token and the English query.
        """
        if language == 'en':
            query_subword_int_arrs = [x['query'] for x in X]
            query_subword_int_arr_reps = self.process_subword_int_arrs(query_subword_int_arrs,
                                                                       self.en_embed,
                                                                       self.en_embed_dropout,
                                                                       compose_subword_first=self.compose_subword_first,
                                                                       subword_lstm=self.subword_en_lstm)
            # subword_int_arr_reps shape: list of shape (seq len, dim) tensors
            query_lengths = [len(query_subword_int_arr_rep) for query_subword_int_arr_rep in query_subword_int_arr_reps]
            query_rep = pad_sequence(query_subword_int_arr_reps) # shape (max len, batch, dim)

            if self.query_use_lstm:
                query_hs, _ = self.pack_batch_lstm(query_rep, query_lengths, self.query_lstm)  # shape (seq len, batch, dim)
                query_rep = self.query_dropout(query_hs)
            return query_rep, query_lengths

        if language == 'src':
            src_subword_int_arrs = [x['src'] for x in X]
            src_subword_int_arr_reps = self.process_subword_int_arrs(src_subword_int_arrs, self.src_embed,
                                                                     self.src_embed_dropout,
                                                                     compose_subword_first=self.compose_subword_first,
                                                                     subword_lstm=self.subword_src_lstm,)
            # subword_int_arr_reps shape: list of shape (seq len, dim) tensors
            src_lengths = [len(src_subword_int_arr_rep) for src_subword_int_arr_rep in src_subword_int_arr_reps]
            src_rep = pad_sequence(src_subword_int_arr_reps)

            if self.src_use_lstm:
                src_hs, _ = self.pack_batch_lstm(src_rep, src_lengths, self.src_lstm)  # shape (seq len, batch, dim)
                src_rep = self.src_dropout(src_hs)
            return src_rep, src_lengths


    def process_subword_int_arrs(self, subword_int_arrs, embedding, embed_dropout, compose_subword_first=True, subword_lstm=None):
        # subword_int_arrs format: 'query': [[word1part1, word1part2, ..., word1partn,], [word2part1, word2part2, ..., word2partn]]
        if compose_subword_first:
            subword_unigram_int_arrs = []
            subword_idx_start_unigram, subword_idx_end_unigram = [], []

            for idx in range(len(subword_int_arrs)):
                subword_int_arr = subword_int_arrs[idx]
                subword_idx_start_unigram.append(len(subword_unigram_int_arrs))
                subword_unigram_int_arrs.extend(subword_int_arr)
                subword_idx_end_unigram.append(len(subword_unigram_int_arrs))

            subword_unigram_int_arr_len = [len(subword_unigram_int_arr) for subword_unigram_int_arr in subword_unigram_int_arrs]
            subword_unigram_int_arrs = pad_sequence(subword_unigram_int_arrs, padding_value=0)

            # Note that BPE only allows update subword embedding and does not allow transform frozen subword embedding
            # This is because pretrained BPE has very low coverage
            subword_unigram_embedded_rep = embed_dropout(embedding(subword_unigram_int_arrs).float())  # shape (query length, batch size, embed dimension)

            _, subword_unigram_rep_q = self.pack_batch_lstm(subword_unigram_embedded_rep, subword_unigram_int_arr_len, subword_lstm)

            subword_int_arr_reps = [] # list of shape (seq_len, rep dim)
            for idx in range(len(subword_int_arrs)):
                subword_int_arr_reps.append(subword_unigram_rep_q[subword_idx_start_unigram[idx]: subword_idx_end_unigram[idx]])

        else:
            subword_int_arr_reps = []
            for idx in range(len(subword_int_arrs)):
                subword_int_arr = subword_int_arrs[idx]
                # concatenate all subwords
                concatenated_subword_int_arr = []
                for subwords in subword_int_arr:
                    concatenated_subword_int_arr.extend(subwords)
                subword_int_arr_reps.append(embed_dropout(embedding(torch.stack(concatenated_subword_int_arr))).float())
        return subword_int_arr_reps


    def pack_batch_lstm(self, features, seq_lens, lstm):
        # features shape (max query length, batch size, embed dimension)
        # PackedSequence object
        packed_features_seq = pack_padded_sequence(features, seq_lens, enforce_sorted=False)

        # Pass through a bidirectional LSTM, LSTM input dimension (seq_len, batch, input_size)
        lstm.flatten_parameters()
        packed_seq_hs, (seq_q, _) = lstm(packed_features_seq.float())  # input shape: seq_len, batch_size, input_size

        seq_q = torch.cat((seq_q[0], seq_q[1]), dim=1) # shape (batch, LSTM output dim)

        seq_hs, _ = pad_packed_sequence(packed_seq_hs, )

        return seq_hs, seq_q


    def normalize_embedding_norms(self):
        self.en_embed.weight.data[1:] = self.en_embed.weight.data[1:] / torch.norm(self.en_embed.weight.data[1:], dim=1).unsqueeze(dim=1)
        self.src_embed.weight.data[1:] = self.src_embed.weight.data[1:] / torch.norm(self.src_embed.weight.data[1:], dim=1).unsqueeze(dim=1)
