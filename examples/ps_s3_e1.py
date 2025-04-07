import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from mlchemy.model import LightGBMPredictor
from mlchemy.preprocessing import Preprocessor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from  lightgbm import LGBMRegressor ,log_evaluation, early_stopping

from mlchemy.validation import get_kfold
from mlchemy.config import TrainerConfig
from mlchemy.trainer import Trainer



# cat_cols = df.select_dtypes(include=['object', 'category']).columns

# config = TrainerConfig(
#     num_folds=5,
#     num_repeats=1,
#     target_col="target",
#     fillna_num=None,
#     fillna_cat="NA",
#     cat_features=cat_cols,
#     cat_encoding='label' # Literal['label', 'onehot', 'frequency'] = 'label'
# )


# trainer = Trainer(config=config, data=df)
# # trainer.fit()
# trainer.fit(train_data=df)

seed = 0
n_splits = 6
shuffle=True
iterations = 5000
early_stopping_rounds = 100
verbose_eval = 0
baseline_rounds = 1
cb_learning_rate = 0.006
xgb_learning_rate = 0.01


cat_params = {'iterations':iterations,
             'learning_rate':cb_learning_rate,
             'depth':7,
             'bootstrap_type':'Bernoulli',
             'random_strength':1,
             'min_data_in_leaf':10,
             'l2_leaf_reg':3,
             'loss_function':'RMSE', 
             'eval_metric':'RMSE',
             'random_seed':seed,
             'grow_policy':'Depthwise',
             'max_bin':1024, 
             'model_size_reg': 0,
             'task_type': 'GPU',
             'od_type':'IncToDec',
             'od_wait':100,
             'metric_period':500,
             'verbose':verbose_eval,
             'subsample':0.8,
             'od_pval':1e-10,
             'max_ctr_complexity': 8,
             'has_time': False,
             'simple_ctr' : 'FeatureFreq',
             'combinations_ctr': 'FeatureFreq'
            }

xgb_params= {'objective': 'reg:squarederror',
             'max_depth': 6,
             'eta': xgb_learning_rate,
             'colsample_bytree': 0.4,
             'subsample': 0.6,
             'reg_alpha' : 6,
             'min_child_weight': 100,
             'n_jobs': 2,
             'seed': 2001,
             'tree_method': 'gpu_hist',
             'gpu_id': 0,
             'predictor': 'gpu_predictor',
            }

lgb_params = {'max_depth': 16,
               'subsample': 0.8032697250789377, 
               'colsample_bytree': 0.21067140508531404,
               'learning_rate': 0.009867383057779643,
               'reg_lambda': 10.987474846877767,
               'reg_alpha': 17.335285595031994,
               'min_child_samples': 31, 
               'num_leaves': 66,
               'max_bin': 522,
               'cat_smooth': 81,
               'cat_l2': 0.029690334194270022,
               'metric': 'rmse',
            #    'n_jobs': -1, 
               'verbose':-1,
               'num_threads': 2,
               'n_estimators': iterations
              }








if __name__ == "__main__":


    data_dir = "./data/ps_s3_e1"
    df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    sub = pd.read_csv(f"{data_dir}/sample_submission.csv")

    target_col = "MedHouseVal"

    print(df.columns)
    cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)

    drop_cols = ["id", target_col]
    num_cols = df.select_dtypes(include=['number']).columns
    num_cols = [col for col in num_cols if col not in drop_cols]

    print("CAT_COLS", cat_cols)
    print("NUM_COLS", num_cols)


    config = TrainerConfig(
        num_folds=5,
        num_repeats=1,
        target_col=target_col,
        fillna_num=None,
        fillna_cat=None,
        cat_features=cat_cols,
        num_features=num_cols,
        cat_encoding='label', # Literal['label', 'onehot', 'frequency'] = 'label'
        label_encoder_cols=cat_cols,
        objective='regression',
        models=[
            # (LGBMRegressor(**lgb_params),'lgb'),
            (CatBoostRegressor(**cat_params),'cat'),
            # (XGBRegressor(**xgb_params),'xgb')
        ],
        drop_cols=drop_cols,
        log_eval_steps=500,
        early_stopping=300
    )

    # for col in df.columns:
    #     print(col, df[col].isna().sum())


    trainer = Trainer(config=config, data=df)
    trainer.fit(df)
    preds_dict = trainer.predict(df)
    print(preds_dict)

    df['preds'] = preds_dict['cat']

    print(df['preds'].value_counts())