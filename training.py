from sys import getsizeof

import cupy
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


class Result:
    def __init__(self, train_acc, train_acc_0, train_acc_1, train_auc, val_acc, val_acc_0, val_acc_1, val_auc):
        """ A struct to hold the accuracy and area under the curve scores for a particular training run """
        self.train_acc = train_acc
        self.train_acc_0 = train_acc_0
        self.train_acc_1 = train_acc_1
        self.train_auc = train_auc
        self.val_acc = val_acc
        self.val_acc_0 = val_acc_0
        self.val_acc_1 = val_acc_1
        self.val_auc = val_auc

    def __str__(self):
        return (
            f"\ntrain_acc: {self.train_acc:.4f}  train_auc: {self.train_auc:.4f}\n"
            f"val_acc:   {self.val_acc:.4f}  val_auc:   {self.val_auc:.4f}"
        )


def split_X_y(data):
    """ Splits the data into training (X) and labels (y) on the column "SwingAndMiss"

    Parameters
    ----------
    data (DataFrame) : the data to split 

    Returns
    -------
    DataFrame : the training data X

    Series : the labels y
    """
    return data.drop(columns="SwingAndMiss"), data.loc[:, "SwingAndMiss"]


def calc_result_stats(results):
    """ Calculates the mean and standard deviation of each metric of the given results 
    
    Parameters
    ----------
    results (List of Result Objects) : the results to get mean and standard deviation for

    Returns
    -------
    dict : a dictionary with the mean and standard deviation of each metric
    """
    train_accs = [result.train_acc for result in results]
    train_accs_0 = [result.train_acc_0 for result in results]
    train_accs_1 = [result.train_acc_1 for result in results]
    train_aucs = [result.train_auc for result in results]
    val_accs = [result.val_acc for result in results]
    val_accs_0 = [result.val_acc_0 for result in results]
    val_accs_1 = [result.val_acc_1 for result in results]
    val_aucs = [result.val_auc for result in results]
    return {
        "train_acc_mean": np.mean(train_accs),
        "train_acc_std": np.std(train_accs),
        "train_acc_0_mean": np.mean(train_accs_0),
        "train_acc_0_std": np.std(train_accs_0),
        "train_acc_1_mean": np.mean(train_accs_1),
        "train_acc_1_std": np.std(train_accs_1),
        "train_auc_mean": np.mean(train_aucs),
        "train_auc_std": np.std(train_aucs),
        "val_acc_mean": np.mean(val_accs),
        "val_acc_std": np.std(val_accs),
        "val_acc_0_mean": np.mean(val_accs_0),
        "val_acc_0_std": np.std(val_accs_0),
        "val_acc_1_mean": np.mean(val_accs_1),
        "val_acc_1_std": np.std(val_accs_1),
        "val_auc_mean": np.mean(val_aucs),
        "val_auc_std": np.std(val_aucs),
    }


def split_data_cv(data):
    """ Divides the data for a variation of k fold cross validation. Instead of choosing an arbitrary k, we replicate 
    the ratio of pitcher ids in the given training and holdout sets. Additionally, each pitcher is either in the 
    training or validation set, but not both. There are 873 pitchers in the training data and 78 pitchers in the holdout 
    data for a ratio of ~11. If we use k=12 on the training data that leaves us with partitions of size ~73. Therefore 
    each training set will be size 800 and each validation set will be size 73 for a ratio of ~11, which replicates the 
    final testing scenario.

    NOTE: This is a generator due to the high memory usage of storing many copies of the data
    
    Parameters
    ----------
    data (DataFrame) : the data to train and validate on

    Returns
    -------
    tuple of dataframes : the split data as a tuple where the first element is the training data and the second element 
    is the validation data
    """
    uq_pitchers = np.unique(data["PitcherID"])
    np.random.shuffle(uq_pitchers)
    fold_pitchers = np.array_split(uq_pitchers.to_array(), 12)

    for fold in fold_pitchers:
        val_data = data[data["PitcherID"].isin(fold)]
        train_data = data.drop(index=val_data.index, columns=["PitcherID"])
        yield train_data, val_data.drop(columns=["PitcherID"])


def train(data, model, feature_selection=None):
    """ Trains the given model on the given data with the given set of features. We can "cheat" and create features 
    outside of the cross validation because all of the features are created independently per pitcher. Since pitchers
    are completely contained in either training or validation this is okay.

    Parameters
    ----------
    data (DataFrame) : the data to train and validate with

    model (Object) : a model object with the methods fit() and predict()

    feature_selection (Object, default=None) : a feature_selection object with the methods fit_transform, transform, and 
    get_support

    Returns
    -------
    List of Results : a list of scores 

    """
    folds = split_data_cv(data)
    results = []
    while True:
        # Get next fold if there is one
        try:
            fold = next(folds)
        except StopIteration:
            return results
        train_data, val_data = fold

        # Split into training and validation sets
        train_X, train_y = split_X_y(train_data)
        val_X, val_y = split_X_y(val_data)

        # Select features with the provided feature selection algorithm and save the features used
        train_X = train_X.to_pandas()
        train_y = train_y.to_array()
        val_X = val_X.to_pandas()
        val_y = val_y.to_array()
        if feature_selection is not None:
            orig_features = np.array(train_X.columns)
            train_X = feature_selection.fit_transform(train_X, train_y)
            val_X = feature_selection.transform(val_X)
            features = list(orig_features[feature_selection.get_support()])

        # Fit model on training, predict on validation
        model.fit(train_X, train_y)
        train_preds = model.predict(train_X)
        val_preds = model.predict(val_X)
        model.get_booster().dump_model('xgb_model.txt', with_stats=True)

        # Get and save results
        train_data["prediction"] = train_preds
        val_data["prediction"] = val_preds
        train_data_0 = train_data[train_data["SwingAndMiss"] == 0]
        train_data_1 = train_data[train_data["SwingAndMiss"] == 1]
        val_data_0 = val_data[val_data["SwingAndMiss"] == 0]
        val_data_1 = val_data[val_data["SwingAndMiss"] == 1]

        train_acc = accuracy_score(train_y, train_preds)
        train_acc_0 = accuracy_score(train_data_0["SwingAndMiss"].to_array(), train_data_0["prediction"].to_array())
        train_acc_1 = accuracy_score(train_data_1["SwingAndMiss"].to_array(), train_data_1["prediction"].to_array())
        train_auc = roc_auc_score(train_y, train_preds)
        val_acc = accuracy_score(val_y, val_preds)
        val_acc_0 = accuracy_score(val_data_0["SwingAndMiss"].to_array(), val_data_0["prediction"].to_array())
        val_acc_1 = accuracy_score(val_data_1["SwingAndMiss"].to_array(), val_data_1["prediction"].to_array())
        val_auc = roc_auc_score(val_y, val_preds)
        r = Result(train_acc, train_acc_0, train_acc_1, train_auc, val_acc, val_acc_0, val_acc_1, val_auc)
        results.append(r)
