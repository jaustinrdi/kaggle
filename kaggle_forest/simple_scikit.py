import pandas as pd
from sklearn import ensemble
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Competition: https://www.kaggle.com/c/forest-cover-type-prediction/

# This code is taken straight from
# https://www.kaggle.com/c/forest-cover-type-prediction/forums/t/8182/first-try-with-random-forests-scikit-learn
# and modified somewhat

if __name__ == "__main__":
    loc_train = "files\\train.csv"
    loc_test = "files\\test.csv"
    loc_submission = "files\\kaggle.forest.submission.csv"

    df_train = pd.read_csv(loc_train)
    df_test = pd.read_csv(loc_test)

    feature_cols = [col for col in df_train.columns if col not in ['Cover_Type', 'Id']]

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y = df_train['Cover_Type']
    test_ids = df_test['Id']

    clf = ensemble.RandomForestClassifier(n_estimators=500, n_jobs=-1)

    clf.fit(X_train, y)

    ##  with open(loc_submission, "wb") as outfile:
    ##    outfile.write(bytes("Id,Cover_Type\n", 'UTF-8'))
    ##    for e, val in enumerate(list(clf.predict(X_test))):
    ##      outfile.write(bytes("%s,%s\n"%(test_ids[e],val), 'UTF-8'))

# look at feature importances
    # http://matplotlib.org/examples/ticks_and_spines/ticklabels_demo_rotation.html
    y = clf.feature_importances_
    x = range(len(y))
    labels = feature_cols
    plt.plot(x, y, 'bo')
    plt.xticks(x, labels, rotation='vertical')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.grid()
    plt.show()

    print("done")
