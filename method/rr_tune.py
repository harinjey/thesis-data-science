import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, randint


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor



frag = pd.read_csv("/mnt/project/Transpose/hidra/results/salmon-hidra/batch1/fragment_filtered_fasta/filtered_fragments.fasta", sep="\t", header= None,  names=["fragment_id", "sequence"])

deseq = pd.read_csv("/mnt/project/Transpose/hidra/results/salmon-hidra/batch1/DESeq_TSV/DESeq_res.tsv", sep="\t")

frag = frag.loc[~frag['sequence'].str.contains('n')].reset_index()
frag.drop("index", axis=1)

deseq = deseq.drop(labels = [2634852, 2634853, 2634854, 2634855, 2634856]).reset_index()
deseq = deseq.drop("index", axis=1)

not_sig = {"pvalue": 1, "padj":1}

deseq = deseq.fillna(value=not_sig)

df_all_cols = pd.concat([frag, deseq ], axis = 1)

chrom_21 = df_all_cols[df_all_cols['fragment'].str.contains('21:')]

chrom_25 = df_all_cols[df_all_cols['fragment'].str.contains('25:')]

test_set = pd.concat([chrom_21, chrom_25], axis = 0)

basemean = test_set.sort_values(by='baseMean',ascending=False).reset_index()

df_training = df_all_cols.loc[~df_all_cols["fragment"].str.contains('25:') & ~df_all_cols["fragment"].str.contains('21:')].reset_index()
df_training.drop("index", axis=1)

filtered_frag = df_training[df_training["baseMean"] >  62.609201]

filtered_test = test_set[test_set["baseMean"] >  65.333982]

y = (filtered_frag.log2FoldChange)

y_test = (filtered_test.log2FoldChange)

def getKmers(sequence, size):
    kmer_seq= [sequence[x:x+size].upper() for x in range(len(sequence) - size + 1)]
    return ' '.join(kmer_seq)
#from list to string

vectorizer = CountVectorizer()

#test set
filtered_test["kmer_2"] = filtered_test["sequence"].apply(lambda x: getKmers(x, size=2))
kmer2test = vectorizer.fit_transform(filtered_test['kmer_2']).astype('float32')
kmer_2test = np.asarray(kmer2test/kmer2test.sum(axis=1))
kmer2_testdf = pd.DataFrame(kmer_2test,columns=['2_mer'] * (4**2))

filtered_test["kmer_3"] = filtered_test["sequence"].apply(lambda x: getKmers(x, size=3))
kmer3test = vectorizer.fit_transform(filtered_test['kmer_3']).astype('float32')
kmer_3test = np.asarray(kmer3test/kmer3test.sum(axis=1))
kmer3_testdf = pd.DataFrame(kmer_3test,columns=['3_mer'] * (4**3))

filtered_test["kmer_4"] = filtered_test["sequence"].apply(lambda x: getKmers(x, size=4))
kmer4test = vectorizer.fit_transform(filtered_test['kmer_4']).astype('float32')
kmer_4test = np.asarray(kmer4test/kmer4test.sum(axis=1))
kmer4_testdf = pd.DataFrame(kmer_4test,columns=['4_mer'] * (4**4))

filtered_test["kmer_5"] = filtered_test["sequence"].apply(lambda x: getKmers(x, size=5))
kmer5test = vectorizer.fit_transform(filtered_test['kmer_5']).astype('float32')
kmer_5test = np.asarray(kmer5test/kmer5test.sum(axis=1))
kmer5_testdf = pd.DataFrame(kmer_5test,columns=['5_mer'] * (4**5))

filtered_test["kmer_6"] = filtered_test["sequence"].apply(lambda x: getKmers(x, size=6))
kmer6test = vectorizer.fit_transform(filtered_test['kmer_6']).astype('float32')
kmer_6test = np.asarray(kmer6test/kmer6test.sum(axis=1))
kmer6_testdf = pd.DataFrame(kmer_6test,columns=['6_mer'] * (4**6))

X_test = pd.concat([kmer2_testdf, kmer3_testdf, kmer4_testdf, kmer5_testdf, kmer6_testdf], axis = 1)

X_test= np.array(X_test)

vectorizer2 = CountVectorizer()
filtered_frag["kmer_2"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=2))
kmer2 = vectorizer2.fit_transform(filtered_frag['kmer_2']).astype('float32')
kmer_2 = np.asarray(kmer2/kmer2.sum(axis=1))
kmer2_df = pd.DataFrame(kmer_2,columns=['2_mer'] * (4**2))
feature_names1 = vectorizer2.get_feature_names_out()


vectorizer3 = CountVectorizer()
filtered_frag["kmer_3"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=3))
kmer3 = vectorizer3.fit_transform(filtered_frag['kmer_3']).astype('float32')
kmer_3 = np.asarray(kmer3/kmer3.sum(axis=1))
kmer3_df = pd.DataFrame(kmer_3,columns=['3_mer'] * (4**3))
feature_names2 = vectorizer3.get_feature_names_out()

vectorizer4 = CountVectorizer()
filtered_frag["kmer_4"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=4))
kmer4 = vectorizer4.fit_transform(filtered_frag['kmer_4']).astype('float32')
kmer_4 = np.asarray(kmer4/kmer4.sum(axis=1))
kmer4_df = pd.DataFrame(kmer_4,columns=['4_mer'] * (4**4))
feature_names3 = vectorizer4.get_feature_names_out()

vectorizer5 = CountVectorizer()
filtered_frag["kmer_5"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=5))
kmer5 = vectorizer5.fit_transform(filtered_frag['kmer_5']).astype('float32')
kmer_5 = np.asarray(kmer5/kmer5.sum(axis=1))
kmer5_df = pd.DataFrame(kmer_5,columns=['5_mer'] * (4**5))
feature_names4 = vectorizer5.get_feature_names_out()

vectorizer6 = CountVectorizer()
filtered_frag["kmer_6"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=6))
kmer6 = vectorizer6.fit_transform(filtered_frag['kmer_6']).astype('float32')
kmer_6 = np.asarray(kmer6/kmer6.sum(axis=1))
kmer6_df = pd.DataFrame(kmer_6,columns=['6_mer'] * (4**6))
feature_names5 = vectorizer6.get_feature_names_out()

X = pd.concat([kmer2_df, kmer3_df, kmer4_df, kmer5_df, kmer6_df], axis = 1)
feature_names = []
feature_names.extend(feature_names1)
feature_names.extend(feature_names2)
feature_names.extend(feature_names3)
feature_names.extend(feature_names4)
feature_names.extend(feature_names5)

# Feature names as list 
X_names = list(X.columns)

X = np.array(X)

X_train,X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state = 42)

n_estimators = [300, 500, 600,700, 900]

max_depth = [4, 6, 8, 12, 14, 16]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_search = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'bootstrap': bootstrap}

# Create a Gaussian Classifier
regressor = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation
clf_rr = RandomizedSearchCV(estimator = regressor, param_distributions = random_search, n_iter = 10,
                               cv = 3, verbose=2, random_state=42, n_jobs = -1)




# Train the model 
clf_rr.fit(X_train, y_train)

print(clf_rr.best_params_)


clf = clf_rr.best_estimator_

# Get predictions
y_pred = clf.predict(X_valid)


rmse_pred = mean_squared_error(y_valid, y_pred) 

print("Root Mean Squared Error:" , np.sqrt(rmse_pred))

print("R2 score:" , r2_score(y_valid, y_pred))



# Predictions on test set after validation
y_pred_test = clf.predict(X_test)

print(r2_score(y_test, y_pred_test))

plt.scatter(y_test, y_pred_test, alpha = 0.5)
plt.savefig("scat_rr.png")
 

print(np.corrcoef(y_test,y_pred_test))



# Get the feature importance scores
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
# Get the top feature indices and their importance scores
top_indices = indices[:500]

nucleotides= [feature_names[i] for i in indices]


# Write selected nucleotides to a FASTA file for the test data
with open("rr_selected_features.fasta", "w") as f:
    for i, seq in enumerate(nucleotides):
        f.write("{}\n".format(seq))
        



