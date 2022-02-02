from deepctr_torch.inputs import create_embedding_matrix

from deepctr_torch.inputs  import SparseFeat, DenseFeat, build_input_features
from deepctr_torch.layers.core import DNN

from .PLBaseModel import PLBaseModel
from ..utils import combined_dnn_input


class NCF(PLBaseModel):
    def __init__(self, user_feature_columns, item_feature_columns, 
                user_gmf_embedding_dim=20, item_gmf_embedding_dim=20,
                user_mlp_embedding_dim=20, item_mlp_embedding_dim=20, dnn_use_bn=False,
                dnn_hidden_units=[64, 32], dnn_activation='relu', l2_reg_dnn=0,
                dnn_dropout=0,
                l2_reg_linear=0.00001, l2_reg_embedding=0.00001, init_std=0.0001, seed=1024, task='binary', device='cpu', **kwargs):
        super().__init__(user_feature_columns, item_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, **kwargs)
        
        if len(user_feature_columns) > 1 or len(item_feature_columns) > 1:
            raise ValueError("目前 NCF 只支持 UserId 和 ItemId 作为特征")
        # 这里的 user_feature_columns 和 dssm 之类的不同，因为 gmf 和 mlp 部分的 embeding 层不一样
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activation = dnn_activation
        self.l2_reg_dnn = l2_reg_dnn
        self.dnn_dropout = dnn_dropout
        self.user_mlp_embedding_dim= user_mlp_embedding_dim
        self.item_mlp_embedding_dim = item_mlp_embedding_dim
        self.dnn_use_bn = dnn_use_bn
        self.init_std = init_std
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_linear = l2_reg_linear
        self.seed = seed
        self.task = task 
        # self.device = device

        # 计算模型的复杂度
        self.user_gmf_embedding_dim = user_gmf_embedding_dim
        self.item_gmf_embedding_dim = item_gmf_embedding_dim

        # GMF Part
        self.user_gmf_feature_columns = [SparseFeat(feat, vocabulary_size=size, 
                                    embedding_dim=self.user_gmf_embedding_dim)
                                    for feat, size, *_ in self.user_feature_columns]
        self.item_gmf_feature_columns = [SparseFeat(feat, vocabulary_size=size, embedding_dim=self.item_gmf_embedding_dim)
                                    for feat, size, *_ in self.item_feature_columns]

        self.feature_index = build_input_features(self.user_gmf_feature_columns + self.item_gmf_feature_columns)

        self.gmf_embedding_dict = create_embedding_matrix(self.user_gmf_feature_columns + self.item_gmf_feature_columns, 
                                                        self.init_std, linear=False, sparse=False,
                                                        device=self.device)

        self.user_mlp_feature_columns = [SparseFeat(feat, vocabulary_size=size, embedding_dim=self.user_mlp_embedding_dim)
                                for feat, size, *_ in self.user_feature_columns]
        self.item_mlp_feature_columns = [SparseFeat(feat, vocabulary_size=size, embedding_dim=self.item_mlp_embedding_dim)
                                    for feat, size, *_ in self.item_feature_columns]
        
        self.mlp_embedding_dict = create_embedding_matrix(self.user_mlp_feature_columns + self.item_mlp_feature_columns, 
                                                        self.init_std, linear=False, sparse=False,
                                                        device=self.device)

        self.mlp_dnn = DNN(self.compute_input_dim(self.user_mlp_feature_columns + self.item_mlp_feature_columns), self.dnn_hidden_units, self.dnn_activation,
                            seed = self.seed, device=device)
        
        self.predict_layer = DNN(self.dnn_hidden_units[-1] + item_gmf_embedding_dim, hidden_units=[1], activation='sigmoid',seed=self.seed, device=device)

    def forward(self, X):
        user_gmf_sparse_embedding_list, user_gmf_dense_value_list = self.input_from_feature_columns(X,
                                                                                            self.user_gmf_feature_columns,
                                                                                            self.gmf_embedding_dict
                                                                                            )
        item_gmf_sparse_embedding_list, item_gmf_dense_value_list = self.input_from_feature_columns(X,
                                                                                            self.item_gmf_feature_columns,
                                                                                            self.gmf_embedding_dict)
        user_gmf_input = combined_dnn_input(user_gmf_sparse_embedding_list, user_gmf_dense_value_list)
        item_gmf_input = combined_dnn_input(item_gmf_sparse_embedding_list, item_gmf_dense_value_list)
        
        # 对应元素相差
        gmf_out = user_gmf_input * item_gmf_input
        
        # MLP part         
        user_mlp_sparse_embedding_list, user_mlp_dense_value_list = self.input_from_feature_columns(X,
                                                                                            self.user_mlp_feature_columns,
                                                                                            self.mlp_embedding_dict)
        item_mlp_sparse_embedding_list, item_mlp_dense_value_list = self.input_from_feature_columns(X,
                                                                                            self.item_mlp_feature_columns,
                                                                                            self.mlp_embedding_dict)
        user_mlp_input = combined_dnn_input(
            user_mlp_sparse_embedding_list, user_mlp_dense_value_list)
        item_mlp_input = combined_dnn_input(
            item_mlp_sparse_embedding_list, item_mlp_dense_value_list)

        mlp_input = combined_dnn_input([user_mlp_input, item_mlp_input])
        mlp_out = self.mlp_dnn(mlp_input)

        # Fusion of GMF and MLP
        neumf_input = combined_dnn_input([gmf_out, mlp_out])

        neumf_out = self.predict_layer(neumf_input)
        neumf_out = neumf_out.squeeze()
        return neumf_out
