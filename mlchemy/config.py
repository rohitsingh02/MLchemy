from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Union
import os

    # num_folds: int = 5
    # num_repeats: int = 1
    # target_col: str = 'target'
    # group_col:str=None
    # seed: int = 2025

    # fillna_num: Union[int, Literal['mean', 'mode', 'min', 'max']] = "mean" #-1 # None to ignore filling null values
    # fillna_cat: Union[str, Literal['mean', 'mode', 'min', 'max']] = "NA" # None to ignore filling null values
    # cat_encoding: Literal['label', 'onehot', 'frequency'] = 'label'
    # drop_cols: list[str] = [] # unused columns to drop from master df (before fold iteration)
    # cat_features: List[str] = [] # categorical_features
    # label_encoder_cols: List[str] = [] # categorical features for le
    # num_features: List[str] = [] # numerical_features
    # objective: Literal['regression', 'binary', 'multi'] = "regression"
    # fold_strategy: Literal['kfold', 'stratified', 'group', 'repeated'] = "kfold"
    # models:List[Tuple[object, str]] = [] # categorical_features
    # log_file_path:str = "trainer.log"

@dataclass
class TrainerConfig:
    num_folds: int = 5
    num_repeats: int = 1
    target_col: str = 'target'
    group_col: str = None
    seed: int = 2025

    fillna_num: Union[int, Literal['mean', 'mode', 'min', 'max']] = "mean"
    fillna_cat: Union[str, Literal['mean', 'mode', 'min', 'max']] = "NA"
    cat_encoding: Literal['label', 'onehot', 'frequency'] = 'label'
    
    drop_cols: List[str] = field(default_factory=list)  
    cat_features: List[str] = field(default_factory=list)
    label_encoder_cols: List[str] = field(default_factory=list)
    num_features: List[str] = field(default_factory=list)

    objective: Literal['regression', 'binary', 'multi'] = "regression"
    metric: Literal['rmse', 'mae', 'mse', 'rmsle', 'msle', 'mape', 'r2', 'smape'] = "rmse" 

    
    fold_strategy: Literal['kfold', 'stratified', 'group', 'repeated'] = "kfold"
    
    models: List[Tuple[object, str]] = field(default_factory=list)  
    log_file_path: str = "trainer.log"

    ### model specific params 
    log_eval_steps: int = 250
    early_stopping: int = 300

    output_dir: str = "outputs"
    save_oof: bool = True






@dataclass
class PreprocessorConfig:
    num_folds: int = 5
    num_repeats: int = 1
    target_col: str = 'target'
    seed: int = 2025





# @dataclass
# class TrainerConfig:
#     num_folds: int = 5
#     n_repeats: int = 1
#     models: List[Tuple] = field(default_factory=list)
#     FE: Optional[object] = None
#     CV_sample: Optional[object] = None
#     group_col: Optional[str] = None
#     target_col: str = 'target'
#     weight_col: str = 'weight'
#     kfold_col: str = 'fold'
#     drop_cols: List[str] = field(default_factory=list)
#     seed: int = 2025
#     objective: Literal['binary', 'multi_class', 'regression'] = 'regression'
#     metric: str = 'mse'
#     nan_margin: float = 0.95
#     num_classes: Optional[int] = None
#     infer_size: int = 10000
#     save_oof_preds: bool = True
#     save_test_preds: bool = True
#     device: str = 'cpu'
#     one_hot_max: int = 50
#     one_hot_cols: Optional[List[str]] = None
#     custom_metric: Optional[object] = None
#     use_optuna_find_params: int = 0
#     optuna_direction: Optional[str] = None
#     early_stop: int = 100
#     use_pseudo_label: bool = False
#     use_high_corr_feat: bool = True
#     cross_cols: List[str] = field(default_factory=list)
#     labelencoder_cols: List[str] = field(default_factory=list)
#     list_stat: List[Tuple] = field(default_factory=list)
#     word2vec_models: List[Tuple] = field(default_factory=list)
#     text_cols: List[str] = field(default_factory=list)
#     plot_feature_importance: bool = False
#     log: int = 100
#     exp_mode: bool = False
#     use_reduce_memory: bool = False
#     use_data_augmentation: bool = False
#     use_oof_as_feature: bool = False
#     use_CIR: bool = False
#     use_median_as_pred: bool = False
#     use_scaler: bool = False
#     use_TTA: bool = False
#     use_eval_metric: bool = True
#     feats_stat: List[Tuple] = field(default_factory=list)
#     target_stat: List[Tuple] = field(default_factory=list)
#     use_spellchecker: bool = False
#     AGGREGATIONS: List[str] = field(default_factory=lambda: ['nunique','count','min','max','first', 'last', 'mean','median','sum','std','skew','kurtosis'])

#     def __post_init__(self):
#         supported_objectives = ['binary', 'multi_class', 'regression']
#         supported_metrics = ['mae', 'rmse', 'mse', 'medae', 'rmsle', 'msle', 'mape', 'r2', 'smape', 'auc', 'pr_auc', 'logloss', 'f1_score', 'mcc', 'accuracy', 'multi_logloss']
#         supported_kfolds = ['KFold', 'GroupKFold', 'StratifiedKFold', 'StratifiedGroupKFold', 'purged_CV', 'custom_kfold']

#         if self.objective not in supported_objectives:
#             raise ValueError(f"Unsupported objective: {self.objective}")
#         if self.metric not in supported_metrics and not self.custom_metric:
#             raise ValueError(f"Unsupported metric: {self.metric}")
#         if self.nan_margin < 0 or self.nan_margin > 1:
#             raise ValueError("nan_margin must be within the range of 0 to 1.")
#         if self.infer_size <= 0:
#             raise ValueError("infer_size must be greater than 0.")
#         if self.objective == 'binary' and self.num_classes != 2:
#             raise ValueError("num_classes must be 2 for binary classification.")
#         if self.objective == 'multi_class' and self.num_classes is None:
#             raise ValueError("num_classes must be defined for multi-class classification.")
#         if self.use_oof_as_feature and self.use_pseudo_label:
#             raise ValueError("use_oof_as_feature and use_pseudo_label cannot both be True.")

#         os.makedirs("mlchemy_models", exist_ok=True)