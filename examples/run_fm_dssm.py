# deepmath_torch 加入 syspath
import sys
sys.path.append("../")

import pandas as pd
# TODO:: 后续修改成 deepmatch_torch.inputs 中的 Feature
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
from deepmatch_torch.models import FM, DSSM


if __name__ == "__main__":

    data = pd.read_csvdata = pd.read_csv("./movielens_sample.txt")
    
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]
    SEQ_LEN = 50
    negsample = 3

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set = gen_data_set(data, negsample)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    embedding_dim = 8

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
                            SparseFeat("age", feature_max_idx['age'], embedding_dim),
                            SparseFeat("occupation", feature_max_idx['occupation'], embedding_dim),
                            SparseFeat("zip", feature_max_idx['zip'], embedding_dim),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    # 3.Define Model and train

    # model = DSSM(user_feature_columns, 
    #     item_feature_columns, 
    #     dnn_hidden_units=[128, 52],
    #     optimizer='Adam',
    #     config={
    #         'gpus': 1
    #         }
    #     )  
    model = FM(user_feature_columns, 
        item_feature_columns, 
        optimizer='Adam',
        config={
            'gpus': '1'
        }
    )  
    
    model.fit(train_model_input, train_label, 
                        max_epochs=1, batch_size=128 )
    
    # 4. Generate user features for testing and full item features for retrieval
    test_user_model_input = test_model_input
    model.mode = "user_representation"
    user_embedding_model = model

    user_embs = user_embedding_model.full_predict(test_user_model_input, batch_size=2)
    print(user_embs.shape)

    model.mode = "item_representation"
    all_item_model_input = {"movie_id": item_profile['movie_id'].values}
    item_embedding_model = model.rebuild_feature_index(item_feature_columns)
    item_embs = item_embedding_model.full_predict(all_item_model_input, batch_size=2 ** 12)
    print(item_embs.shape)
    
