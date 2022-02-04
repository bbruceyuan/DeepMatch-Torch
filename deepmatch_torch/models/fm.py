import torch

from .PLBaseModel import PLBaseModel
from ..layers import FactorizationMachines


class FM(PLBaseModel):
    def __init__(self, user_feature_columns, item_feature_columns, 
        l2_reg_linear=0.00001, l2_reg_embedding=0.00001, init_std=0.0001, 
        seed=1024, task='binary', device='cpu', gpus=None, **kwargs):
        super().__init__(user_feature_columns, item_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus, **kwargs)
        self.fm = FactorizationMachines()

    def forward(self, X):
        # (batch, N) N 表示 多列
        # X 包括两部分，一部分是 linear part, 一部分是 fm
        if self.mode == 'train':
            linear_logit = self.linear_model(X)
        #  TODO: 增加 dense_value interaction
        user_fm_input = self.user_tower(X)
        item_fm_input = self.item_tower(X)
        if self.mode == "user_representation":
            return user_fm_input
        if self.mode == "item_representation":
            return item_fm_input
        
        fm_input = torch.cat([user_fm_input, item_fm_input], dim=1)
        # 需要确定的是那些 fields 需要 intersection
        fm_logit = self.fm(fm_input)

        res = linear_logit + fm_logit
        # shape is (batch_size, 1) predict 部分进行了 squeeze 
        res = res.squeeze()
        return res
    
    def item_tower(self, X):
        if self.mode == "user_representation":
            return None
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.item_feature_columns,
                                                                                  self.embedding_dict)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        return fm_input

    def user_tower(self, X):
        if self.mode == "item_representation":
            return None
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.user_feature_columns,
                                                                                  self.embedding_dict)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        return fm_input