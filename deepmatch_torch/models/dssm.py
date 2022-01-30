
import torch.nn.functional as F

from .BaseModel import BaseModel
from deepctr_torch.layers import DNN
from ..utils import combined_dnn_input


class DSSM(BaseModel):
    def __init__(self, user_feature_columns, item_feature_columns, 
                dnn_hidden_units, dnn_activation="relu", gamma=0.01,
                l2_reg_linear=0.00001, l2_reg_embedding=0.00001, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super().__init__(user_feature_columns, item_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.user_dnn = DNN(self.compute_input_dim(user_feature_columns), dnn_hidden_units,
                        activation=dnn_activation, init_std=init_std, device=device)

        self.item_dnn = DNN(self.compute_input_dim(item_feature_columns), dnn_hidden_units,
                            activation=dnn_activation, init_std=init_std, device=device)

        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus

    def forward(self, inputs):
        user_sparse_embedding_list, user_dense_value_list = \
            self.input_from_feature_columns(inputs, self.user_feature_columns, self.embedding_dict)

        user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
        self.user_dnn_embedding = self.user_dnn(user_dnn_input)

        item_sparse_embedding_list, item_dense_value_list = \
            self.input_from_feature_columns(inputs, self.item_feature_columns, self.embedding_dict)

        item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
        self.item_dnn_embedding = self.item_dnn(item_dnn_input)
        print(self.item_dnn_embedding.shape)
        score = F.cosine_similarity(self.user_dnn_embedding, self.item_dnn_embedding, dim=1)
        # shape is (batch)
        return score

        