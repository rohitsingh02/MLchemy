from sklearn.model_selection import KFold, StratifiedKFold

def get_kfold(n_splits=5, stratified=False):
    return StratifiedKFold(n_splits=n_splits) if stratified else KFold(n_splits=n_splits)
