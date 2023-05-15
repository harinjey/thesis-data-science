import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

y_test = (filtered_test.log2FoldChange)

def getKmers(sequence, size):
    kmer_seq= [sequence[x:x+size].upper() for x in range(len(sequence) - size + 1)]
    return ' '.join(kmer_seq)


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

#training set
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


# Create a Gaussian Classifier
clf_xgboost = xgb.XGBRegressor(objective="reg:squarederror",random_state=42)



# Train the model 
clf_xgboost.fit(X_train, y_train)


import joblib
filename = "xgb_def.joblib"
joblib.dump(clf_xgboost, filename)


# Get predictions
y_pred = clf_xgboost.predict(X_valid)

# Calculatig root mean absolute error
rmse_pred = mean_squared_error(y_valid, y_pred) 

print("Root Mean Squared Error, validation:" , np.sqrt(rmse_pred))

print("R2 score validation:" , r2_score(y_valid, y_pred))

#plt.scatter(y_valid, y_pred, alpha = 0.5)
#plt.savefig("default_xgb_valid.png")



# Predictions on test set 
y_pred_test = clf_xgboost.predict(X_test)

# Calculatig root mean squared error
rmse_pred_test = mean_squared_error(y_test, y_pred_test) 

print("Root Mean Squared Error, test:" , np.sqrt(rmse_pred_test))

print("R2 score test:", r2_score(y_test, y_pred_test))

#plt.scatter(y_test, y_pred_test, alpha = 0.5)

 

# Calculating the correlation cofficient y_test and y_pred_test
print(np.corrcoef(y_test.T, y_pred_test.T)[0, 1])

r = np.corrcoef(y_test.T, y_pred_test.T)[0, 1]

# Plot the true vs predicted values with some customizations
plt.scatter(y_test, y_pred_test, alpha=0.5, color = "blue")
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Correlation coefficient: {:.2f}'.format(r))
plt.show()
plt.savefig("test1.png")

