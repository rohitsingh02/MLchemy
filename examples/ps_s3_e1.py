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



    # X_train = df.drop(['id', 'target'], axis=1)
    # y_train = df.target
    # X_test = test_df.drop(['id'], axis=1)


    # cat_cols = [feature for feature in df.columns if 'cat' in feature]
    # cont_cols = [feature for feature in df.columns if 'con' in feature]

    # for feature in cat_cols:
    #     le = LabelEncoder()
    #     le.fit(df[feature])
    #     X_train[feature] = le.transform(X_train[feature])
    #     X_test[feature] = le.transform(X_test[feature])


    # seed = 0
    # n_splits = 6
    # shuffle=True
    # iterations = 50000
    # early_stopping_rounds = 400
    # verbose_eval = 0
    # baseline_rounds = 1
    # cb_learning_rate = 0.006
    # xgb_learning_rate = 0.01



    # split = KFold(n_splits=n_splits, shuffle=True, random_state=seed)



    # preds_list = []
    # oof_cb = np.zeros((len(df)))
    # oof_xgb = np.zeros((len(df)))
    # oof_cbx = np.zeros((len(df)))
    # oof_xgbx = np.zeros((len(df)))
    # oof_lgb = np.zeros((len(df)))
    # oof_lgb_incremental = np.zeros((len(df)))
    # stack_oof = np.zeros((len(df)))
    # stack_preds = np.zeros((len(test_df)))

    # for fold, (train_idx, val_idx) in enumerate(split.split(X_train)):
    #     print(f'Fold {fold+1}')
    #     X_tr = X_train.iloc[train_idx]
    #     X_val = X_train.iloc[val_idx]
    #     y_tr = y_train.iloc[train_idx]
    #     y_val = y_train.iloc[val_idx]
    #     fold_stack_oof = np.zeros((len(X_val), 6))
    #     fold_stack_preds = np.zeros((len(test_df), 6))
    #     ptrain = Pool(data=X_tr, label=y_tr, cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #     pvalid = Pool(data=X_val, label=y_val, cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #     ptest = Pool(data=X_test, cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #     CModel = CatBoostRegressor(**cb_params)
    #     CModel.fit(ptrain,
    #             eval_set=pvalid,
    #             use_best_model=True,
    #             early_stopping_rounds=early_stopping_rounds)
    #     temp_fold_preds = CModel.predict(pvalid)
    #     oof_cb[val_idx] = temp_fold_preds
    #     first_cb_rmse = mean_squared_error(y_val, temp_fold_preds, squared=False)
    #     print(f'RMSE of CB model is {first_cb_rmse}')
    #     baseline_preds_tr_cb = CModel.predict(ptrain)
    #     baseline_preds_vl_cb = temp_fold_preds
    #     test_preds_cb = CModel.predict(ptest)   
    #     fold_stack_oof[:,0] = temp_fold_preds
    #     fold_stack_preds[:,0] = test_preds_cb
    #     xtrain = DMatrix(data=X_tr, label=y_tr, nthread=2)
    #     xvalid = DMatrix(data=X_val, label=y_val, nthread=2)
    #     xtest = DMatrix(data=X_test, nthread=2)
    #     XModel = xgb.train(xgb_params, xtrain,
    #                     evals=[(xvalid,'validation')],
    #                     verbose_eval=verbose_eval,
    #                     early_stopping_rounds=early_stopping_rounds,
    #                     xgb_model=None,
    #                     num_boost_round=iterations)
    #     temp_fold_preds = XModel.predict(xvalid)
    #     oof_xgb[val_idx] = temp_fold_preds
    #     first_xgb_rmse = mean_squared_error(y_val, temp_fold_preds, squared=False)
    #     print(f'RMSE of XGB model is {first_xgb_rmse}')
    #     baseline_preds_tr_xgb = XModel.predict(xtrain)
    #     baseline_preds_vl_xgb = temp_fold_preds
    #     test_preds_xgb = XModel.predict(xtest)
    #     fold_stack_oof[:,1] = temp_fold_preds
    #     fold_stack_preds[:,1] = test_preds_xgb
    #     ltrain = lgbm.Dataset(X_tr, label=y_tr, init_score=None, categorical_feature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], free_raw_data=False)
    #     lvalid = lgbm.Dataset(X_val, label=y_val, init_score=None, categorical_feature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], free_raw_data=False)
    #     ltest =  lgbm.Dataset(X_test, label=y_val, init_score=None, categorical_feature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], free_raw_data=False)
    #     LModel = lgbm.train(lgbm_params,
    #                         train_set=ltrain,
    #                         num_boost_round=iterations,
    #                         valid_sets=lvalid, 
    #                         init_model=None,
    #                         early_stopping_rounds=early_stopping_rounds,
    #                         verbose_eval=verbose_eval)           
    #     temp_fold_preds = LModel.predict(X_val)
    #     oof_lgb[val_idx] = temp_fold_preds
    #     fold_stack_oof[:,2] = temp_fold_preds
    #     fold_stack_preds[:,2] = LModel.predict(X_test)
    #     first_lgb_rmse = mean_squared_error(y_val, temp_fold_preds, squared=False)
    #     print(f'RMSE of LGBM model is {first_lgb_rmse}')
    #     params = lgbm_params.copy()     
    #     params.update({'learning_rate': 0.003})
    #     for i in range(1, 9):
    #         if i > 2:                      
    #             params['reg_lambda'] *= 0.9
    #             params['reg_alpha']  *= 0.9
    #             params['num_leaves'] += 40                   
            
    #         LModel = lgbm.train(lgbm_params,
    #                             train_set=ltrain,
    #                             num_boost_round=iterations,
    #                             valid_sets=lvalid, 
    #                             init_model=LModel,
    #                             early_stopping_rounds=early_stopping_rounds,
    #                             verbose_eval=verbose_eval)           
    #     temp_fold_preds = LModel.predict(X_val)
    #     oof_lgb_incremental[val_idx] = temp_fold_preds
    #     second_lgb_rmse = mean_squared_error(y_val, temp_fold_preds, squared=False)
    #     print(f'RMSE of LGBM model is {second_lgb_rmse}')
    #     print(f'LGBM improvement using Incremental Improvements {first_lgb_rmse - second_lgb_rmse}')
    #     baseline_preds_tr_lgb = LModel.predict(X_tr)
    #     baseline_preds_vl_lgb = temp_fold_preds
    #     test_preds_lgb = LModel.predict(X_test)
    #     fold_stack_oof[:,3] = temp_fold_preds
    #     fold_stack_preds[:,3] = test_preds_lgb
        
    #     baseline_train = (baseline_preds_tr_xgb+baseline_preds_tr_lgb+baseline_preds_tr_cb)/3
    #     baseline_valid = (baseline_preds_vl_xgb+baseline_preds_vl_lgb+baseline_preds_vl_cb)/3
    #     baseline_test = (test_preds_xgb+test_preds_lgb+test_preds_cb)/3
        
    #     for baseline in range(baseline_rounds):
    #         ptrain = Pool(data=X_tr, label=y_tr, cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], baseline=baseline_train)
    #         pvalid = Pool(data=X_val, label=y_val, cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], baseline=baseline_valid)
    #         ptest = Pool(data=X_test, cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], baseline=baseline_test)
    #         cb_params_ = cb_params.copy()
    #         cb_params_.update({'learning_rate': cb_learning_rate*(1/(baseline+1))})
    #         CModel = CatBoostRegressor(**cb_params_)
    #         CModel.fit(ptrain, 
    #                 eval_set=pvalid,
    #                 use_best_model=True,
    #                 early_stopping_rounds=early_stopping_rounds)
    #         temp_fold_preds = CModel.predict(pvalid)
    #         oof_cbx[val_idx] = temp_fold_preds
    #         second_cb_rmse = mean_squared_error(y_val, temp_fold_preds, squared=False)
    #         print(f'RMSE of CB model with baseline round {baseline+1} {second_cb_rmse}')   
    #         baseline_train = CModel.predict(ptrain)
    #         baseline_valid = CModel.predict(pvalid)
    #         baseline_test = CModel.predict(ptest)
    #         if baseline == baseline_rounds - 1:
    #             fold_stack_oof[:,4] = temp_fold_preds
    #             fold_stack_preds[:,4] = baseline_test
            
    #         xtrain = DMatrix(data=X_tr, label=y_tr, base_margin=baseline_train)
    #         xvalid = DMatrix(data=X_val, label=y_val, base_margin=baseline_valid)
    #         xtest =  DMatrix(data=X_test, base_margin=baseline_test)
    #         xgb_params_ = xgb_params.copy()
    #         xgb_params_.update({'learning_rate': xgb_learning_rate*(1/(baseline+1))})
    #         XModel = xgb.train(xgb_params_, xtrain,
    #                         evals=[(xvalid,'validation')],
    #                         verbose_eval=verbose_eval,
    #                         early_stopping_rounds=early_stopping_rounds,
    #                         xgb_model=None,
    #                         num_boost_round=iterations)
    #         temp_fold_preds = XModel.predict(xvalid)
    #         oof_xgbx[val_idx] = temp_fold_preds
    #         baseline_train = XModel.predict(xtrain)
    #         baseline_valid = temp_fold_preds
    #         baseline_test = XModel.predict(xtest)
    #         if baseline == baseline_rounds - 1:
    #             fold_stack_oof[:,5] = temp_fold_preds
    #             fold_stack_preds[:,5] = baseline_test
    #         second_xgb_rmse = mean_squared_error(y_val, temp_fold_preds, squared=False)
    #         print(f'RMSE of XGB model with baseline round {baseline+1} {second_xgb_rmse}')
    #         print(f'CB Improvement  using Baseline round {baseline+1}: {first_cb_rmse - second_cb_rmse}')
    #         print(f'XGB Improvement using Baseline round {baseline+1}: {first_xgb_rmse - second_xgb_rmse}')
    #         first_cb_rmse = second_cb_rmse
    #         first_xgb_rmse = second_xgb_rmse
        
    #     stacker = RidgeCV().fit(fold_stack_oof, y_val)
    #     temp_stack_fold_pred = stacker.predict(fold_stack_oof)
    #     stack_oof[val_idx] = temp_stack_fold_pred
    #     stack_rmse = mean_squared_error(y_val, temp_stack_fold_pred, squared=False)
    #     print(f'RMSE of stack model  {stack_rmse}')
    #     stack_preds += stacker.predict(fold_stack_preds)/n_splits
    #     print('-' * 100)
    #     print('',end='\n')
        
        






