import torch.optim as optim
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from util import to_device
import pickle as pkl
import time

torch.set_num_threads(4)

class Model:
    """
    Model wrapper class that implements minibatch training and early stopping mechanism in PyTorch.
    Also has predict score function and compute loss function for model evaluation.
    All outside inputs to this class is numpy, and all outputs of this class are tensors.
    All outside inputs are assumed to be on cpu. It's this class's responsibility to transfer the tensors to cuda if gpu
    is used.
    """
    def __init__(self, model, mode='train', train_X=None, train_y=None, dev_X=None, dev_y=None, experiment_dir=None,
                 weight_dir=None, num_epochs=None, patience=None, batch_size=None, min_epochs=None, lr=None, embed_lr=None, device='cpu',
                 optimizer=None, embed_optimizer=None,
                 train_rationale=None, lambda_attention=None,
                 start_from_epoch=None, start_from_model_dir=None,
                 ):

        self.model = model
        self.mode = mode

        self.model = to_device(self.model, device)

        if self.mode == 'train':
            assert train_X is not None and train_y is not None
            self.train_X, self.train_y, self.dev_X, self.dev_y = train_X, train_y, dev_X, dev_y
            self.num_epochs, self.patience, self.min_epochs, self.batch_size = num_epochs, patience, min_epochs, batch_size
            self.optimizer_name, self.embed_optimizer_name = optimizer, embed_optimizer
            self.optimizer, self.embed_optimizer = None, None
            self.experiment_dir = experiment_dir
            self.train_rationale, self.lambda_attention = train_rationale, lambda_attention

            # define the loss functions here
            self.loss_function = nn.BCELoss()
            if self.train_rationale:
                self.attention_loss_function = nn.KLDivLoss(reduction="sum")
                assert self.lambda_attention is not None

            # define the optimizer and bind it to model parameters
            assert len(list(self.model.named_parameters())) == len(list(self.model.parameters()))
            params = []
            embed_params = []
            for p in self.model.named_parameters():
                if p[1].requires_grad:
                    if p[0] in ['en_embed.weight', 'src_embed.weight']:
                        embed_params.append(p[1])
                    else:
                        params.append(p[1])

            if self.embed_optimizer_name is not None and len(embed_params) != 0:
                self.embed_lr = embed_lr
                if self.embed_optimizer_name == 'adam':
                    self.embed_optimizer = optim.Adam(embed_params, self.embed_lr, weight_decay=0.001)
                elif self.embed_optimizer_name == 'sgd':
                    self.embed_optimizer = optim.SGD(embed_params, self.embed_lr, weight_decay=0.001)
                elif self.embed_optimizer_name == 'sparse_adam':
                    self.embed_optimizer = optim.SparseAdam(embed_params, self.embed_lr)
                else:
                    raise Exception("No such optimizer for embedding.")

            if self.optimizer_name is not None:
                if self.embed_optimizer_name is not None:
                    optimizer_params = params
                else:
                    optimizer_params = params + embed_params
                if len(optimizer_params) != 0:
                    self.lr = lr
                    if self.optimizer_name == 'adam':
                        self.optimizer = optim.Adam(optimizer_params, self.lr, weight_decay=0.001)
                    elif self.optimizer_name == 'sgd':
                        self.optimizer = optim.SGD(optimizer_params, self.lr, weight_decay=0.001)
                    elif self.optimizer_name == 'adagrad':
                        self.optimizer = optim.Adagrad(optimizer_params, self.lr, weight_decay=0.001)
                    else:
                        raise Exception("No such optimizer.")
            self.device = device

            self.start_from_epoch = start_from_epoch
            if start_from_epoch:
                state_dict = torch.load(start_from_model_dir, map_location=device)
                self.model.load_state_dict(state_dict)

        else:
            self.weight_dir = weight_dir
            self.batch_size = batch_size
            self.model.eval()

            state_dict = torch.load(self.weight_dir, map_location=device)
            self.model.load_state_dict(state_dict)

            # Default BCELoss has reduction mean, use the mean of a batch's loss to do gradient descent.
            self.loss_function = nn.BCELoss()
            self.device = device


    def train_one_batch(self, batch_X, batch_y):
        """
        Train the model on one minibatch of data.
        Returns the loss on this batch.
        Note that the loss is returned as a pytorch tensor because this function is not supposed to be called by functions
        outside of this class.
        """
        self.model.train()
        if self.train_rationale:
            self.model.return_rationale = True

        # Load batch training data
        batch_X_cuda = []
        for sample in batch_X:
            if self.model.en_input_format == 'word' and self.model.src_input_format == 'word':
                cuda_sample = {'query': to_device(torch.LongTensor(sample['query']), self.device),
                               'src': to_device(torch.LongTensor(sample['src']), self.device)}
            else:
                cuda_sample = {'query': [to_device(torch.LongTensor(query), self.device) for query in sample['query']],
                               'src': [to_device(torch.LongTensor(src), self.device) for src in sample['src']]}
            if 'subword_rationale_distr' in sample and \
                    self.model.en_input_format == 'bpe' and self.model.src_input_format == 'bpe' and \
                    not self.model.compose_subword_first:
                cuda_sample['rationale_distr'] = sample['subword_rationale_distr']
            elif 'rationale_distr' in sample:
                cuda_sample['rationale_distr'] = sample['rationale_distr']
            batch_X_cuda.append(cuda_sample)
        batch_X = batch_X_cuda

        if not self.train_rationale:
            result = self.model(batch_X, mode='train').float()
            batch_y = torch.stack([torch.FloatTensor([y]) for y in batch_y])
            batch_y = to_device(batch_y, self.device)
            loss = self.loss_function(result, batch_y)
        else:
            result, rationale_pred = self.model(batch_X, mode='train')
            batch_y = torch.stack([torch.FloatTensor([y]) for y in batch_y])
            batch_y = to_device(batch_y, self.device)
            try:
                bce_loss = self.loss_function(result, batch_y)
            except:
                import pdb
                pdb.set_trace()

            # compute batch loss
            attention_loss = 0
            for batch_idx in range(len(batch_X)):
                if 'rationale_distr' in batch_X[batch_idx]:
                    gt_rationale_distr = batch_X[batch_idx]['rationale_distr']
                    model_pred_rationale_distr = rationale_pred[batch_idx]
                    attention_loss += self.attention_loss_function(torch.log(model_pred_rationale_distr),
                                                                   to_device(torch.FloatTensor(gt_rationale_distr), self.device))
            loss = bce_loss + self.lambda_attention * attention_loss / len(batch_X)

        loss.backward()

        if self.optimizer is not None:
            self.optimizer.step()
        if self.embed_optimizer is not None:
            self.embed_optimizer.step()

        self.model.zero_grad()
        return loss.item()


    def train(self):
        """
        Train the model on minibatches of data with early stopping mechanism on validation set.
        Saves optimal model state at local.
        """
        optimal_loss = np.inf
        patience_count = 0
        optimal_model_state = None
        num_epochs_until_optimal = None
        training_loss_at_optimal = None
        if self.device != 'cpu':
            assert torch.get_num_threads() == 4

        with open(self.experiment_dir + "README", 'a') as f:
            f.write("Model architecture parameters: \n%s\n\n" % str(self.model.model_property))
            f.write("Train on %d samples, Validate on %d samples.\n" % (len(self.train_X[0]), len(self.dev_X)))

        if not self.start_from_epoch:
            start_epoch_number = 1
        else:
            start_epoch_number = self.start_from_epoch + 1

        for loop_idx in range(start_epoch_number, self.num_epochs + 1):
            self.model.train()
            start_time = time.time()
            print("Epoch %d:" % loop_idx)
            # create minibatches
            self.epoch_train_X = self.train_X[min(loop_idx - 1, len(self.train_X) - 1)]
            self.epoch_train_y = self.train_y[min(loop_idx - 1, len(self.train_y) - 1)]

            shuffle_idx = np.random.permutation(range(len(self.epoch_train_X)))
            self.epoch_train_X = [self.epoch_train_X[i] for i in shuffle_idx]
            self.epoch_train_y = [self.epoch_train_y[i] for i in shuffle_idx]
            num_minibatches = len(self.epoch_train_X) // self.batch_size if len(self.epoch_train_X) % self.batch_size == 0 \
                    else len(self.epoch_train_X) // self.batch_size + 1
            # train minibatches
            training_loss = []

            time1 = time.time()
            for batch_idx in range(1, num_minibatches+1):
                loss = self.train_one_batch(self.epoch_train_X[(batch_idx - 1) * self.batch_size: min(batch_idx * self.batch_size, len(self.epoch_train_X))],
                                     self.epoch_train_y[(batch_idx - 1) * self.batch_size: min(batch_idx * self.batch_size, len(self.epoch_train_X))])
                training_loss.append(loss)
                if batch_idx % 1000 == 0 or batch_idx == num_minibatches:
                    print("Finish training %d/%d batches. Cumulative average training loss: %.4f." % (batch_idx, num_minibatches, np.mean(training_loss)))
                    time2 = time.time()
                    print("Uses time: %d minutes." % (int(time2 - time1) // 60))
                    time1 = time.time()

            train_loss = np.mean(training_loss)
            self.model.eval()

            dev_pred = self.predict(self.dev_X)
            dev_y = torch.stack([torch.FloatTensor([y]) for y in self.dev_y])
            val_predictions = torch.FloatTensor(dev_pred).view((-1, 1))
            current_loss = self.loss_function(val_predictions, dev_y).item()
            pkl.dump(val_predictions, open(self.experiment_dir + "val_pred_scores_epoch%d.pkl" % loop_idx, 'wb'))

            used_time = (time.time() - start_time) // 60
            with open(self.experiment_dir + "README", 'a') as f:
                f.write("Epoch %d: Validation Loss: %.4f, Training Loss: %.4f, Use Time: %d Minutes\n"
                        % (loop_idx, current_loss, train_loss, used_time))

            torch.save(self.model.state_dict(), self.experiment_dir + "model_epoch%d.pkl" % loop_idx)
            optimal_loss = min(optimal_loss, current_loss)
            if optimal_loss == current_loss:
                num_epochs_until_optimal = loop_idx
                training_loss_at_optimal = train_loss
                optimal_model_state = deepcopy(self.model.state_dict())
                patience_count = 0
            else:
                patience_count += 1
            if loop_idx >= self.min_epochs and patience_count >= self.patience:
                break

        torch.save(optimal_model_state, self.experiment_dir + "model.pkl")
        return num_epochs_until_optimal, training_loss_at_optimal


    def predict(self, X):
        """
        Predict numerical scores for the batch of features with the model.
        If model is returning attention, the method will return an array of predictions and an array of attention weights.
        If model is not returning attention, the method will return an array of predictions only.
        Scores are returned as a numpy array of floats and attention weights are returned as a numpy array.
        """
        self.model.eval()
        self.model.return_rationale = False

        # train rationale can only be True during training.
        assert not self.model.return_rationale

        num_batches = len(X) // self.batch_size if len(X) % self.batch_size == 0 else \
            len(X) // self.batch_size + 1

        pred = []
        for batch_idx in range(1, num_batches+1):
            batch_X = X[(batch_idx - 1) * self.batch_size: min(batch_idx * self.batch_size, len(X))]

            # Load batch prediction data
            batch_X_cuda = []
            for sample in batch_X:
                if self.model.en_input_format == 'word' and self.model.src_input_format == 'word':
                    cuda_sample = {'query': to_device(torch.LongTensor(sample['query']), self.device),
                                   'src': to_device(torch.LongTensor(sample['src']), self.device)}
                else:
                    cuda_sample = {
                        'query': [to_device(torch.LongTensor(query), self.device) for query in sample['query']],
                        'src': [to_device(torch.LongTensor(src), self.device) for src in sample['src']]}
                batch_X_cuda.append(cuda_sample)
            batch_X = batch_X_cuda

            with torch.no_grad():
                pred_batch = self.model(batch_X, mode='test')

            pred.extend(pred_batch)

        return pred


    def compute_score_loss(self, X, truth_y):
        """
        Compute loss between model's numerical prediction of X and the ground truth label y averaged across samples.
        X and y passed as input are supposed to be numpy arrays/python builtin int/float.
        This method is responsible for wrapping input with tensor.
        Returns average loss as a float number so outside methods can use the return value directly.
        Note that this method is used only for prediction not training and is not weighted.
        """
        self.model.eval()
        prediction = self.predict(X)

        # Change the list of tensors to a large tensor
        truth_y = torch.stack([torch.FloatTensor([y]) for y in truth_y])
        prediction = torch.FloatTensor(prediction).view((-1, 1))
        loss = self.loss_function(prediction, truth_y)
        return loss.item()
