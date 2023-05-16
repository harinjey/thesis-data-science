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

import xgboost as xgb


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

def getKmers(sequence, size):
    kmer_seq= [sequence[x:x+size].upper() for x in range(len(sequence) - size + 1)]
    return ' '.join(kmer_seq)
#from list to string

vectorizer = CountVectorizer()

filtered_frag["kmer_2"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=2))
kmer2 = vectorizer.fit_transform(filtered_frag['kmer_2']).astype('float32')
kmer_2 = np.asarray(kmer2/kmer2.sum(axis=1))
kmer2_df = pd.DataFrame(kmer_2,columns=['2_mer'] * (4**2))

filtered_frag["kmer_3"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=3))
kmer3 = vectorizer.fit_transform(filtered_frag['kmer_3']).astype('float32')
kmer_3 = np.asarray(kmer3/kmer3.sum(axis=1))
kmer3_df = pd.DataFrame(kmer_3,columns=['3_mer'] * (4**3))

filtered_frag["kmer_4"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=4))
kmer4 = vectorizer.fit_transform(filtered_frag['kmer_4']).astype('float32')
kmer_4 = np.asarray(kmer4/kmer4.sum(axis=1))
kmer4_df = pd.DataFrame(kmer_4,columns=['4_mer'] * (4**4))

filtered_frag["kmer_5"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=5))
kmer5 = vectorizer.fit_transform(filtered_frag['kmer_5']).astype('float32')
kmer_5 = np.asarray(kmer5/kmer5.sum(axis=1))
kmer5_df = pd.DataFrame(kmer_5,columns=['5_mer'] * (4**5))

filtered_frag["kmer_6"] = filtered_frag["sequence"].apply(lambda x: getKmers(x, size=6))
kmer6 = vectorizer.fit_transform(filtered_frag['kmer_6']).astype('float32')
kmer_6 = np.asarray(kmer6/kmer6.sum(axis=1))
kmer6_df = pd.DataFrame(kmer_6,columns=['6_mer'] * (4**6))

X = pd.concat([kmer2_df, kmer3_df, kmer4_df, kmer5_df, kmer6_df], axis = 1)

# Feature names as list 
X_names = list(X.columns)

X = np.array(X)

X_train,X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state = 42)

# Create a xgbregressor model
xgboost = xgb.XGBRegressor(objective="reg:squarederror",random_state=42)

#parameter for tuning
# Inspired by: https://github.com/yvasandvik/data-science-thesis/blob/master/method/xgb-tuning-model.ipynb
params = {
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150)
}

xgb_random = RandomizedSearchCV(xgboost, param_distributions=params, random_state=42, n_iter=10, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

# Train the model 
xgb_random.fit(X_train, y_train)

import joblib
filename = "xgb_check.joblib"
joblib.dump(xgb_random.best_estimator_, filename)

print(xgb_random.best_score_)
print(xgb_random.best_params_)

print(xgb_random.best_estimator_)

clf = xgb_random.best_estimator_


# Get predictions
y_pred = clf.predict(X_valid)


# Calculatig root mean absolute error
rmse_pred = mean_squared_error(y_valid, y_pred) 

print("Root Mean Squared Error:" , np.sqrt(rmse_pred))

print("R2 score:" , r2_score(y_valid, y_pred))

print(np.corrcoef(y_valid,y_pred))






