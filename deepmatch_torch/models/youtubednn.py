from .PLBaseModel import PLBaseModel
from deepctr_torch.layers import DNN
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from deepctr_torch.inputs import get_varlen_pooling_list, varlen_embedding_lookup
from ..utils import combined_dnn_input
from ..inputs import create_embedding_matrix
import torch.nn.functional as F


# youtube dnn 的结构没有什么改变
class YouTubeDNN(PLBaseModel):
    def __init__(self, user_feature_columns, item_feature_columns, 
                num_sampled=5, user_dnn_hidden_units=[64, 32], 
                dnn_activation='relu', dnn_use_bn=False,
                device="cpu", init_std=0.002,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, 
               dnn_dropout=0, activation='relu', seed=1024, **kwargs):
        super(YouTubeDNN, self).__init__(user_feature_columns, item_feature_columns, 
                l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                init_std=0.0001, seed=1024, task='binary', device='cpu', **kwargs)
        self.num_sampled = num_sampled

        self.user_dnn = DNN(self.compute_input_dim(user_feature_columns), user_dnn_hidden_units,
                        activation=dnn_activation, init_std=init_std, device=device)

    def forward(self, X):
        batch_size = X.size(0)
        user_embedding = self.user_tower(X)
        item_embedding = self.item_tower(X)

        if self.mode == "user_representation":
            return user_embedding
        if self.mode == "item_representation":
            return item_embedding

        score = F.cosine_similarity(user_embedding, item_embedding, dim=-1)
        score = score.view(batch_size, -1)
        return score
    
    def item_tower(self, X):
        if self.mode == "user_representation":
            return None
        
        item_embedding_list, _ = self.input_from_item_feature_columns(X, self.item_feature_columns, self.embedding_dict)
        item_embedding = item_embedding_list[0]  # (batch, movie_list_len, feat_dim)
        return item_embedding

    def user_tower(self, X):
        if self.mode == "item_representation":
            return None
        # sample softmax 可以通过 构造样本实现
        user_sparse_embedding_list, user_dense_value_list = \
            self.input_from_feature_columns(X, self.user_feature_columns, self.embedding_dict)
        
        user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
        user_embedding = self.user_dnn(user_dnn_input)  # (batch_size, embedding_dim)
        user_embedding = user_embedding.unsqueeze(1) # (batch, 1, embedding_dim)
        return user_embedding
    
    def input_from_item_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
    
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

        # 这里返回的就是 movie_id 的 embedding
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        feat_name = varlen_sparse_feature_columns[0].name
        item_embedding = sequence_embed_dict[feat_name] 
        # shape is (batch, movie_id_len, feat_dim)

        varlen_sparse_embedding_list = [item_embedding]
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list