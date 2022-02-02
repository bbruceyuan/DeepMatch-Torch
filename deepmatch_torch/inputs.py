from torch.utils.data import Dataset, DataLoader
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat
import torch.nn as nn


class CommonDataSet(Dataset):
    # TODO, 可以生成完整的 训练数据生成逻辑
    def __init__(self) -> None:
        super().__init__()
        
        self.data = None
    
    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return len(self.data)

# copy from DeepMatch and remove device
def create_embedding_matrix(feature_columns, init_std=0.001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    # for feat in sparse_feature_columns:
    #     print(feat.embedding_name)
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict
