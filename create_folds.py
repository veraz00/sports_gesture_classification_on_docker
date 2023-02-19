import os
import pandas as pd
from sklearn import model_selection  
# refer KFold: https://towardsdatascience.com/how-to-train-test-split-kfold-vs-stratifiedkfold-281767b93869

if __name__ == "__main__":

    df = pd.read_csv("/home/linlin/dataset/sports_kaggle/sports.csv")
    df["kfold"] = -1
    # df = df.sample(frac=0.1).reset_index(drop=True)  # df.sample(frac = percentage): randomly select percentage of rows and return 
    y = df.labels.values
    kf = model_selection.StratifiedKFold(n_splits=10)  
    # from either train or test, it has the same percentage of class
    # return n times of split, in every split, there would be train_index, test_index, 
    # from the whole splits, every index would be experienced into test group
    for fold_num, (_, v_) in enumerate(kf.split(X=df, y=y)):    # train_index, test_index
        df.loc[v_, "kfold"] = fold_num
    print(df.kfold.value_counts())
    df.to_csv("/home/linlin/dataset/sports_kaggle/sports_with_fold.csv")
    #df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)
