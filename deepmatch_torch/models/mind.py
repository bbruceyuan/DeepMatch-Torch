from .PLBaseModel import PLBaseModel
from deepctr_torch.layers import DNN
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from deepctr_torch.inputs import get_varlen_pooling_list, varlen_embedding_lookup
from ..utils import combined_dnn_input
from ..inputs import create_embedding_matrix
import torch.nn.functional as F
from ..layers.core import CapsuleLayer
import torch


class MIND(PLBaseModel):
    def __init__(self, user_feature_columns, item_feature_columns, 
                num_sampled=5, user_dnn_hidden_units=[64, 32], 
                dnn_activation='relu', dnn_use_bn=False,
                dynamic_k=False, p=1, k_max=2,
                device="cpu", init_std=0.002,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, 
               dnn_dropout=0, activation='relu', seed=1024, **kwargs):
        super(MIND, self).__init__(user_feature_columns, item_feature_columns, 
                l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                init_std=0.0001, seed=1024, task='binary', device='cpu', **kwargs)
        self.num_sampled = num_sampled
        self.pow_p = 2
        self.capsule_layer = CapsuleLayer(user_feature_columns[0].embedding_dim, 1, 
                    4, user_feature_columns[0].embedding_dim, 20)
        # self.label_att = LabelAwareAttention(4, 1)
        self.user_dnn = DNN(self.compute_input_dim(user_feature_columns), user_dnn_hidden_units,
                        activation=dnn_activation, init_std=init_std, device=device)

    def forward(self, X):
        
        # user_embedding = user_embedding.unsqueeze(1) # (batch, 1, embedding_dim)

        # item_embedding_list, _ = self.input_from_item_feature_columns(X, self.item_feature_columns, self.embedding_dict)
        # item_embedding = item_embedding_list[0]  # (batch, movie_list_len, feat_dim)
        user_embedding = self.user_tower(X)
        item_embedding = self.item_tower(X)

        if self.mode == "user_representation":
            return user_embedding
        if self.mode == "item_representation":
            return item_embedding
        # item_embedding_list 目前 size = 1
        # random sample 的数量
        batch_size = X.size(0)
        movie_list_len = item_embedding.shape[1]

        final_output_list = []
        for i in range(movie_list_len):
            # label attention_part
            tmp_input = [user_embedding, item_embedding[:, i, :]]
            keys = tmp_input[0]
            query = tmp_input[1].unsqueeze(1)

            weights = torch.sum(keys * query, axis=-1, keepdim=True)
            weights = torch.pow(weights, self.pow_p)
            weights = torch.softmax(weights, dim=1)
            output = torch.sum(keys * weights, axis=1)
            final_output_list.append(output)
        user_embedding = torch.cat(final_output_list, dim=1).view(batch_size, movie_list_len, -1)
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
        # 找到 user_profile 相关的 feature
        # sample softmax 可以通过 构造样本实现
        user_sparse_feature_columns = [feat for feat in self.user_feature_columns if not feat.name.startswith("hist")]
        user_sparse_embedding_list, user_dense_value_list = \
            self.input_from_feature_columns(X, user_sparse_feature_columns, self.embedding_dict)
        
        # user history 序列特征
        user_history_feature_columns = [feat for feat in self.user_feature_columns if not feat.name.startswith("hist")]
        histroy_feature_embedding_list, _ = self.input_from_feature_columns(X, user_history_feature_columns, self.embedding_dict)

        capsule_input = histroy_feature_embedding_list[0]  # (batch, feat_embedding_dim)
        capsule_output = self.capsule_layer(capsule_input)
        
        cap_cnt = capsule_output.size(1) 
        user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

        user_dnn_input = torch.tile(user_dnn_input.unsqueeze(1), [1, cap_cnt, 1])
        user_dnn_input = torch.cat([user_dnn_input, capsule_output], dim=-1)

        user_embedding = self.user_dnn(user_dnn_input)  # (batch_size, interest_cnt, embedding_dim)
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