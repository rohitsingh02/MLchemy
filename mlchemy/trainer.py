import os
import json
import dill
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, RepeatedKFold
from typing import List, Optional, Tuple, Literal, Union, Dict
from  lightgbm import log_evaluation, early_stopping

from .model import LightGBMPredictor
from .validation import get_kfold
from .config import TrainerConfig
from .preprocessing import Preprocessor
from .utils import get_logger


class Trainer:
    def __init__(self, config: TrainerConfig, data: pd.DataFrame):
        self.config = config
        self.data = data
        self.le_mappings = {}  # Store category → integer mappings for label encoding
        self.onehot_encoder = {}  # Store fitted OneHotEncoder per column

        # Create output directory if it doesn't exist
        if hasattr(self.config, "output_dir"):
            os.makedirs(self.config.output_dir, exist_ok=True)

        self.logger = get_logger(f"{self.config.output_dir}/{self.config.log_file_path}")
        self.logger.info(f"Initialized Trainer with config: {self.config}")
        # Save TrainerConfig
        config_path = os.path.join(self.config.output_dir, "trainer_config.json")
        self._save_config_as_json(self.config, config_path)


    def _pickle_dump(self, obj, path):
        with open(path, mode="wb") as f:
            dill.dump(obj, f, protocol=4)

    def _pickle_load(self, path):
        with open(path, mode="rb") as f:
            data = dill.load(f)
            return data
    
    def _save_config_as_json(self, config, path):
        """Dump only JSON-serializable parts of the config."""
        def safe_convert(value):
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            elif isinstance(value, list):
                return [safe_convert(v) for v in value]
            elif isinstance(value, dict):
                return {k: safe_convert(v) for k, v in value.items()}
            else:
                return str(value)  # Fallback for unserializable objects (e.g., models)

        serializable_config = {
            k: safe_convert(v) for k, v in config.__dict__.items() if k != "models"
        }

        with open(path, "w") as f:
            json.dump(serializable_config, f, indent=4)


    def _preprocess_data(self):
        """Basic preprocessing steps like dropping columns, handling missing values."""

        # pre_processor = Preprocessor(self.config)
        # data = pre_processor.fit_transform(data)
        # X, y = data.drop(columns=["target"]), data["target"]
        prep = Preprocessor()
        self.data = prep.fit_transform(self.data)
        # X, y = data.drop(columns=["target"]), data["target"]
        # self.data = data

    def _load_data(self, 
                train_data: str | pd.DataFrame | pl.DataFrame = "train.csv", 
                mode: str = "") -> None | pd.DataFrame:
        """Load data from a file or DataFrame with error handling."""

        # Load data if a file path is provided
        if isinstance(train_data, str):
            try:
                if train_data.endswith(".csv"):
                    df = pl.read_csv(train_data)
                elif train_data.endswith(".parquet"):
                    df = pl.read_parquet(train_data)
                else:
                    self.logger.error(f"Unsupported file format: {train_data}")
                    raise ValueError("Only CSV and Parquet formats are supported.")
                df = df.to_pandas()
                self.logger.info(f"Successfully loaded data from {train_data}")
            except FileNotFoundError:
                self.logger.error(f"File not found: {train_data}")
                raise
        else:
            # If already a DataFrame, copy and ensure it’s Pandas format
            df = train_data.copy()
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
            if not isinstance(df, pd.DataFrame):
                self.logger.error(f"Invalid input type for {mode}. Expected a Pandas DataFrame.")
                raise ValueError(f"Invalid input type for {mode}. Expected a Pandas DataFrame.")

        # If no mode is specified, just return the loaded DataFrame
        return df


    # def _label_encoder(self, df:pd.DataFrame, config:TrainerConfig, mode="train"):
    #     self.logger.info("Using LabelEncoder to encode categorical variables ...")
    #     for col in config.label_encoder_cols:
    #         if mode == 'train':
    #             le = LabelEncoder()
    #             df[col] = le.fit_transform(df[col].astype(str))  # Convert to str for safety
    #             self.le_mappings[f'le_{col}.model'] = dict(zip(le.classes_, le.transform(le.classes_)))
    #             self.logger.info(f"Column {col} LE -> {dict(zip(le.classes_, le.transform(le.classes_)))}")
    #         else:  # Apply saved mapping
    #             ### Assuming no new classes in test
    #             self.logger.info(f"Loading {col} LE -> {self.le_mappings.get(f'le_{col}.model', {})}")
    #             df[col] = df[col].astype(str).map(self.le_mappings.get(f'le_{col}.model', {})) #.fillna(-1).astype(int)
    #     return df
    
    def _label_encoder(self, df: pd.DataFrame, config: TrainerConfig, mode="train"):
        self.logger.info("Using LabelEncoder to encode categorical variables ...")
        # Define the save/load path
        le_mapping_path = os.path.join(config.output_dir, "le_mappings.model")
        
        if mode == "train":
            for col in config.label_encoder_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))  # Ensure type safety
                self.le_mappings[f'le_{col}.model'] = dict(zip(le.classes_, le.transform(le.classes_)))
                self.logger.info(f"Column {col} LE -> {self.le_mappings[f'le_{col}.model']}")
            # Save mappings to disk
            self._pickle_dump(self.le_mappings, le_mapping_path)
            self.logger.info(f"Label encoder mappings saved to: {le_mapping_path}")
        else:  # mode == "inference" or "valid"
            # Load mappings from disk
            self.le_mappings = self._pickle_load(le_mapping_path)
            self.logger.info(f"Loaded label encoder mappings from: {le_mapping_path}")
            
            for col in config.label_encoder_cols:
                mapping = self.le_mappings.get(f'le_{col}.model', {})
                self.logger.info(f"Loading {col} LE -> {mapping}")
                df[col] = df[col].astype(str).map(mapping)  # You could optionally fillna(-1).astype(int)

        return df



    def _basic_preprocessing(self, df, config, mode):
        """Basic preprocessing steps like dropping columns, handling missing values."""
        self.logger.info("Starting basic data preprocessing...")
        df = df.copy()

        # Handle Missing values - Fill numerical columns
        # num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = config.num_features
        if config.fillna_num is not None:  # Only fill if a strategy is provided
            if isinstance(config.fillna_num, int):  # Direct integer replacement
                df[num_cols] = df[num_cols].fillna(config.fillna_num)
            elif config.fillna_num in ['mean', 'mode', 'min', 'max']:
                if config.fillna_num == 'mean':
                    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                elif config.fillna_num == 'mode':
                    df[num_cols] = df[num_cols].fillna(df[num_cols].mode().iloc[0])
                elif config.fillna_num == 'min':
                    df[num_cols] = df[num_cols].fillna(df[num_cols].min())
                elif config.fillna_num == 'max':
                    df[num_cols] = df[num_cols].fillna(df[num_cols].max())


        # Handle Missing values - Fill categorical columns
        # cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_cols = config.cat_features
        if isinstance(config.fillna_cat, str):  # Direct string replacement
            df[cat_cols] = df[cat_cols].fillna(config.fillna_cat)

        ### handle categorical features encoding [Current Support label|onehot|frequency ]
        if len(config.label_encoder_cols) > 0:
            df = self._label_encoder(df, config=config, mode=mode)

        for col in cat_cols:
            if config.cat_encoding == 'onehot':
                if mode == 'train':
                    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.uint8)
                    transformed = ohe.fit_transform(df[[col]])
                    self.onehot_encoder[col] = ohe  # Store fitted encoder
                    
                    # Create a DataFrame with new one-hot encoded columns
                    ohe_df = pd.DataFrame(transformed, columns=[f"{col}_{cat}" for cat in ohe.categories_[0]])
                    df = df.drop(columns=[col]).reset_index(drop=True)
                    df = pd.concat([df, ohe_df], axis=1)
                
                else:  # Apply fitted encoder
                    ohe = self.onehot_encoder.get(col)
                    if ohe is not None:
                        transformed = ohe.transform(df[[col]])
                        ohe_df = pd.DataFrame(transformed, columns=[f"{col}_{cat}" for cat in ohe.categories_[0]])
                        df = df.drop(columns=[col], errors='ignore').reset_index(drop=True)
                        df = pd.concat([df, ohe_df], axis=1)


            elif config.cat_encoding == 'frequency':
                    freq_map = df[col].value_counts().to_dict()
                    df[col] = df[col].map(freq_map)

        print(self.le_mappings)
        return df



    def _create_folds(self, df: pd.DataFrame, config:TrainerConfig) -> pd.DataFrame:
        """Adds a 'fold' column based on cross-validation strategy."""
        self.logger.info(f"Creating {self.config.num_folds} folds using {self.config.fold_strategy} strategy...")
        
        df = df.copy()
        df["fold"] = -1  # Initialize fold column

        try:
            if self.config.fold_strategy == "kfold":
                kf = KFold(n_splits=self.config.num_folds, shuffle=True, random_state=self.config.seed)
                for fold, (_, val_idx) in enumerate(kf.split(df)):
                    df.loc[val_idx, "fold"] = fold

            elif self.config.fold_strategy == "stratified":
                if self.config.target_col is None:
                    self.logger.error("Target column is required for StratifiedKFold.")
                    raise ValueError("target_col must be provided for StratifiedKFold.")
                skf = StratifiedKFold(n_splits=self.config.num_folds, shuffle=True, random_state=self.config.seed)
                for fold, (_, val_idx) in enumerate(skf.split(df, df[self.config.target_col])):
                    df.loc[val_idx, "fold"] = fold

            elif self.config.fold_strategy == "group":
                if self.config.group_col is None:
                    self.logger.error("Group column is required for GroupKFold.")
                    raise ValueError("group_col must be provided for GroupKFold.")
                gkf = GroupKFold(n_splits=self.config.num_folds)
                for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df[self.config.group_col])):
                    df.loc[val_idx, "fold"] = fold

            elif self.config.fold_strategy == "repeated":
                rkf = RepeatedKFold(n_splits=self.config.num_folds, n_repeats=self.config.num_repeats, random_state=self.config.seed)
                for fold, (_, val_idx) in enumerate(rkf.split(df)):
                    df.loc[val_idx, "fold"] = fold % self.config.num_folds
            else:
                self.logger.error(f"Unsupported fold strategy: {self.config.fold_strategy}")
                raise ValueError("Unsupported fold strategy.")

            self.logger.info("Fold creation completed.")
        except Exception as e:
            self.logger.error(f"Error creating folds: {e}")
            raise
        return df
    

    def _compute_cv(self, config, y_true, y_pred):
        if config.objective=='regression':
            if config.metric=='mae':
                return np.mean(np.abs(y_true-y_pred))
            elif config.metric=='rmse':
                return np.sqrt(np.mean((y_true-y_pred)**2))
            elif config.metric=='mse':
                return np.mean((y_true-y_pred)**2)
            elif config.metric=='rmsle':
                y_pred=np.clip(y_pred,0,1e20)
                return np.sqrt(np.mean((np.log1p(y_true)-np.log1p(y_pred))**2))
            elif config.metric=='msle':
                y_pred=np.clip(y_pred,0,1e20)
                return np.mean((np.log1p(y_true)-np.log1p(y_pred))**2)
            elif config.metric=='mape':
                y_true[y_true<1]=1
                return np.mean(np.abs(y_true-y_pred)/(np.abs(y_true)+self.eps))
            elif config.metric=='r2':
                return 1-np.sum ((y_true-y_pred)**2)/np.sum ((y_true-np.mean(y_true))**2)
            elif config.metric=='smape':
                return 200*np.mean(np.abs(y_true-y_pred) / ( np.abs(y_true)+np.abs(y_pred)+self.eps ) )




    def _train_model(self, df: pd.DataFrame, config:TrainerConfig, models:List[Tuple]):
        self.logger.info("Starting model training...")
        feature_cols = config.num_features + config.label_encoder_cols
        self.logger.info(f"Training model on {len(feature_cols)} features...")
        self.logger.info(f"Feature list {list(feature_cols)}")

        X = df[feature_cols].copy() #df.drop(config.drop_cols, axis=1)
        y = df[config.target_col]
        oofs_dict = dict()

        for model_idx, (model, model_name) in enumerate(models):
            self.logger.info(f"Training model: {model_name}")
            oof = np.zeros(len(X))

            for fold in range(config.num_folds):
                train_idx = df.loc[df.fold != fold].index
                val_idx = df.loc[df.fold == fold].index

                X_train, y_train = X.loc[train_idx], y.loc[train_idx]
                X_val, y_val = X.loc[val_idx], y.loc[val_idx]

                try:
                    if model_name == "lgb":
                        model.fit(
                            X_train, y_train, 
                            eval_set=[(X_val, y_val)],
                            callbacks=[log_evaluation(config.log_eval_steps), early_stopping(300)]
                        )
                    elif model_name == "cat":
                        model.fit(X_train, y_train, cat_features=self.config.cat_features, verbose=config.log_eval_steps)
                    elif model_name == "xgb":
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=config.log_eval_steps)
                    
                    y_pred = model.predict(X_val)
                    oof[val_idx] = y_pred
                    fold_cv = self._compute_cv(config, y_val, y_pred)
                    self.logger.info(f"Fold {fold} training completed for {model_name}, CV: {fold_cv}")

                    model_save_pth = f"{config.output_dir}/{model_name}_fold{fold}_r0.model"
                    self._pickle_dump(model, model_save_pth)

                except Exception as e:
                    self.logger.error(f"Error training {model_name} on fold {fold}: {e}")
                    continue

            oofs_dict[model_name] = oof
            ### save oof_file
            oof_save_path = f"{config.output_dir}/{model_name}_oof.csv"
            self._pickle_dump(oof, oof_save_path)
            model_cv = self._compute_cv(config, y, oof)
            self.logger.info(f"Finished training model: {model_name}, OOF CV: {model_cv}")
        self.logger.info("Model training completed.")



    def _infer_model(self, df: pd.DataFrame, config: TrainerConfig, models: List[Tuple]) -> Dict[str, np.ndarray]:
        self.logger.info("Starting inference using saved models...")
        feature_cols = config.num_features + config.label_encoder_cols
        # X = df.drop(config.drop_cols, axis=1)
        X = df[feature_cols].copy() #df.drop(config.drop_cols, axis=1)
        self.logger.debug(f"Predicting model on {len(feature_cols)} features...")
        self.logger.debug(f"Feature list {list(feature_cols)}")

        preds_dict = {}

        for model_idx, (_, model_name) in enumerate(models):
            self.logger.info(f"Generating predictions with model: {model_name}")
            preds = np.zeros(len(X))

            for fold in range(config.num_folds):
                model_path = f"{config.output_dir}/{model_name}_fold{fold}_r0.model"
                try:
                    model = self._pickle_load(model_path)
                    self.logger.info(f"Loaded model: {model_path}")
                    fold_preds = model.predict(X)
                    preds += fold_preds / config.num_folds
                except Exception as e:
                    self.logger.error(f"Error loading/predicting with {model_name} fold {fold}: {e}")
                    continue

            preds_dict[model_name] = preds
            self.logger.info(f"Finished inference for model: {model_name}")

        self.logger.info("Inference completed for all models.")
        print("pred", preds_dict)
        return preds_dict

                

    def fit(self, 
        train_data: str | pd.DataFrame | pl.DataFrame = "train.csv",
        mode:str="train",
        save_models: bool = True,
       ):

        # self.datetime_features=datetime_features
        df = self._load_data(train_data=train_data, mode=mode) ### pandas dataframe
        df = self._basic_preprocessing(df, self.config, mode=mode)

        ### based on fold_strategy & num_folds create folds
        if 'fold' not in df.columns:
            df = self._create_folds(df, self.config)

        ### train model
        print(df.head())
        print("*"*100)
        self._train_model(df, config=self.config, models=self.config.models)


    def predict(self, 
            test_data: str | pd.DataFrame | pl.DataFrame = "train.csv",
            mode:str="test"
        ):
        df = self._load_data(train_data=test_data, mode=mode) ### pandas dataframe
        df = self._basic_preprocessing(df, self.config, mode=mode)
        preds_dict = self._infer_model(df, config=self.config, models=self.config.models)
        return preds_dict





    # def run(self):
    #     """Main execution function."""
    #     self._preprocess_data()
    #     # self._create_folds()
    #     self.fit()