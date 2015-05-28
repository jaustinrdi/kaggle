import pandas as pd
from sklearn import ensemble
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image

# Competition: https://www.kaggle.com/c/forest-cover-type-prediction/

# This code is taken straight from
# http://nbviewer.ipython.org/github/aguschin/kaggle/blob/master/forestCoverType_featuresEngineering.ipynb
# and modified somewhat


train = pd.read_csv('files/train.csv')
test = pd.read_csv('files/test.csv')

# Inspect data

train.ix[:0,:11]

train.ix[:,:11].hist(figsize=(16,12),bins=50)
#plt.show()


# As we know about that Aspect is in degrees azimuth, we can try to shift it at 180

def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180

train['Aspect2'] = train.Aspect.map(r)
test['Aspect2'] = test.Aspect.map(r)

# You can notice that Vertical_Distance_To_Hydrology have some negative values.
# It may be good idea to create variable which indicates positive or negative value here

train['Highwater'] = train.Vertical_Distance_To_Hydrology < 0
test['Highwater'] = test.Vertical_Distance_To_Hydrology < 0

# Take a look at Elevation and Vertical_Distance_To_Hydrology. Color each cover type in unique color to see some patterns


def plotc(c1,c2):

    fig = plt.figure(figsize=(16,8))
    sel = np.array(list(train.Cover_Type.values))

    plt.scatter(c1, c2, c=sel, s=100)
    plt.xlabel(c1.name)
    plt.ylabel(c2.name)
    plt.show()

# plotc(train.Elevation, train.Vertical_Distance_To_Hydrology)

# Now we can create some variable that will have plot a bit simplier (Besides understanding dependencies like this may lead you to some cool features!)
# plotc(train.Elevation-train.Vertical_Distance_To_Hydrology, train.Vertical_Distance_To_Hydrology)

# Fine! Now we can add this new feature to train and test (the same goes for Horizontal_Distance_To_Hydrology, you can check this by youself)
train['EVDtH'] = train.Elevation-train.Vertical_Distance_To_Hydrology
test['EVDtH'] = test.Elevation-test.Vertical_Distance_To_Hydrology

train['EHDtH'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2
test['EHDtH'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2


# Finally, we can create some features from those which have similar meanings, f.e. distances
train['Distanse_to_Hydrolody'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
test['Distanse_to_Hydrolody'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5

train['Hydro_Fire_1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
test['Hydro_Fire_1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']

train['Hydro_Fire_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
test['Hydro_Fire_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

train['Hydro_Road_1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])

train['Hydro_Road_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])


# Now let's make a submission using all those new features

feature_cols = [col for col in train.columns if col not in ['Cover_Type','Id']]

X_train = train[feature_cols]
X_test = test[feature_cols]
y = train['Cover_Type']
test_ids = test['Id']

# forest = ensemble.ExtraTreesClassifier(n_estimators=400, criterion='gini', max_depth=None,
#     min_samples_split=2, min_samples_leaf=1, max_features='auto',
#     bootstrap=False, oob_score=False, n_jobs=-1, random_state=None, verbose=0,
#     min_density=None)

forest = ensemble.RandomForestClassifier(n_estimators=500, n_jobs=-1)
# forest = ensemble.ExtraTreesClassifier(n_estimators=400, n_jobs=-1)

print("Fitting...")
forest.fit(X_train, y)

# with open('features_engineering_benchmark.csv', "wb") as outfile:
#     outfile.write("Id,Cover_Type\n")
#     for e, val in enumerate(list(forest.predict(X_test))):
#         outfile.write("%s,%s\n"%(test_ids[e],val))

# look at feature importances

pd.DataFrame(forest.feature_importances_,index=X_train.columns).sort([0], ascending=False) [:10]


print("done")