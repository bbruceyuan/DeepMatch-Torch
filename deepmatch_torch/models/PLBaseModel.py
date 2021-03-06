from abc import abstractmethod
from functools import partial

import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch
import torch.nn as nn
from deepctr_torch.inputs import build_input_features
from deepctr_torch.inputs import varlen_embedding_lookup, get_varlen_pooling_list
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from ..inputs import create_embedding_matrix


# DeepMatch_Torch Linear Model
class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.my_device = device
        self.feature_index = feature_index
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,)
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.my_device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(sparse_embedding_list[0].device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


# ?????? Pl ?????? fit ?????????
class PLBaseModel(LightningModule):
    """Base class for all DeepMatch_Torch models.
    This model inspired from: https://github.com/Rose-STL-Lab/torchTS/blob/main/torchts/nn/model.py
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        opimizer_args (dict): Arguments for the optimizer
        criterion: Loss function
        criterion_args (dict): Arguments for the loss function
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        scheduler_args (dict): Arguments for the scheduler
        scaler (torchts.utils.scaler.Scaler): Scaler
    """

    def __init__(
        self,
        user_feature_columns,
        item_feature_columns,
        optimizer=None,
        optimizer_args=None,
        criterion=F.mse_loss,
        criterion_args=None,
        scheduler=None,
        scheduler_args=None,
        scaler=None,
        config={},
        **kwargs,
    ):
        super().__init__()
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        # DeepMatch side ??????????????????
        self.config = config  # ?????? config ???????????? DeepMatch ??????????????????
        # ???????????? kwargs ?????????
        # TODO: ???????????????????????? device , 
        self.config.update(kwargs)
        
        # ?????????????????? ?????? logits ?????? user/item ??? vector ??????
        self.mode = self.config.get('mode', 'train')

        self.linear_feature_columns = user_feature_columns + item_feature_columns
        self.dnn_feature_columns = self.linear_feature_columns 

        # ??? pl ???????????? to(device)
        self.reg_loss = torch.zeros((1,))
        self.aux_loss = torch.zeros((1,))

        self.feature_index = build_input_features(
            self.linear_feature_columns)

        self.embedding_dict = create_embedding_matrix(self.dnn_feature_columns, 
                    self.config.get("init_std"), sparse=False, )

        self.linear_model = Linear(
            self.linear_feature_columns, self.feature_index, device=self.device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), 
            l2=self.config.get("l2_reg_embedding"))
        self.add_regularization_weight(self.linear_model.parameters(), 
            l2=self.config.get("l2_reg_linear"))

        self.criterion = criterion
        self.criterion_args = criterion_args
        self.scaler = scaler
        self.gpus = self.config.get("gpus", 0)

        # ??????????????? optimizer ?????????
        optimizer = self.init_optimizer(optimizer)
        if optimizer_args is not None:
            self.optimizer = partial(optimizer, **optimizer_args)
        else:
            self.optimizer = optimizer

        if scheduler is not None and scheduler_args is not None:
            self.scheduler = partial(scheduler, **scheduler_args)
        else:
            self.scheduler = scheduler

    def fit(self, x, y, max_epochs=10, batch_size=128):
        """Fits model to the given data.
        Args:
            x (torch.Tensor): Input data
            y (torch.Tensor): Output data
            max_epochs (int): Number of training epochs
            batch_size (int): Batch size for torch.utils.data.DataLoader
        """
        x, y = self.construct_input(x, y)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        trainer = Trainer(max_epochs=max_epochs, gpus=self.gpus)
        trainer.fit(self, loader)

    def prepare_batch(self, batch):
        return batch

    def _step(self, batch, batch_idx, num_batches):
        """
        Args:
            batch: Output of the torch.utils.data.DataLoader
            batch_idx: Integer displaying index of this batch
            dataset: Data set to use
        Returns: loss for the batch
        """
        x, y = self.prepare_batch(batch)

        if self.training:
            batches_seen = batch_idx + self.current_epoch * num_batches
        else:
            batches_seen = batch_idx

        pred = self(x)

        if self.scaler is not None:
            y = self.scaler.inverse_transform(y)
            pred = self.scaler.inverse_transform(pred)

        # ????????? softmax, ????????? y ????????? Long??????
        if self.criterion == F.cross_entropy:
            y = y.long()
        if self.criterion_args is not None:
            loss = self.criterion(pred, y, **self.criterion_args)
        else:
            loss = self.criterion(pred, y)

        return loss

    def training_step(self, batch, batch_idx):
        """Trains model for one step.
        Args:
            batch (torch.Tensor): Output of the torch.utils.data.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        train_loss = self._step(batch, batch_idx, len(self.trainer.train_dataloader))
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        """Validates model for one step.
        Args:
            batch (torch.Tensor): Output of the torch.utils.data.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        val_loss = self._step(batch, batch_idx, len(self.trainer.val_dataloader))
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Tests model for one step.
        Args:
            batch (torch.Tensor): Output of the torch.utils.data.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        test_loss = self._step(batch, batch_idx, len(self.trainer.test_dataloader))
        self.log("test_loss", test_loss)
        return test_loss

    @abstractmethod
    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Predicted data
        """

    def predict(self, x):
        """Runs model inference.
        Args:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Predicted data
        """
        return self(x).detach()

    def configure_optimizers(self):
        """Configure optimizer.
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        optimizer = self.optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]

        return optimizer

    def init_optimizer(self, optimizer_or_name):
        """init_optimizer ?????? optimizer ?????????string, ???????????? torch.optim????????? optimizer
            ?????????????????????
        Args:
            optimizer_or_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(optimizer_or_name, str):
            if optimizer_or_name == "Adam":
                return torch.optim.Adam
            elif optimizer_or_name == "SGD":
                return torch.optim.SGD
        else:
            return optimizer_or_name
            
    def construct_input(self, x, y):
        # ?????????????????????????????? tensor
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = torch.from_numpy(np.concatenate(x, axis=-1))
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), )
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss
    
    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
        
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def rebuild_feature_index(self, feature_columns):
        # ???????????? ???????????? user/item vector ?????????????????????????????? feature_columns ?????????
        self.feature_index = build_input_features(feature_columns)
        return self

    def full_predict(self, x, batch_size=256):
        # ?????????????????????????????? tensor
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = torch.from_numpy(np.concatenate(x, axis=-1))
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        ret = []
        for batch in loader:            
            tmp_result = self.predict(batch[0])
            ret.append(tmp_result)
        return torch.cat(ret, axis=0)
        
