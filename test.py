from pandas import read_csv

dataset = read_csv("dataset.csv", header=1, skiprows=147)
labels = dataset["koi_disposition"]
dataset = dataset[labels != "CANDIDATE"]
labels = labels[labels != "CANDIDATE"]
# labels.loc[labels == "CONFIRMED"] = "CANDIDATE"

# dataset = read_csv("tess_dataset.csv", header=1, skiprows=89)
# labels = dataset["tfopwg_disp"]
# dataset = dataset[~labels.isin(["APC", "PC"])]
# labels = labels[~labels.isin(["APC", "PC"])]
# labels.loc[labels.isin(["CP", "KP"])] = "CANDIDATE"
# labels.loc[labels.isin(["FP", "FA"])] = "FALSE POSITIVE"

# features = ["koi_period", "koi_eccen", "koi_impact", "koi_duration", "koi_depth", "koi_ror", "koi_srho", "koi_prad", "koi_sma", "koi_incl", "koi_teq", "koi_insol", "koi_dor", "koi_model_snr", "koi_count", "koi_num_transits", "koi_bin_oedp_sig", "koi_steff", "koi_slogg", "koi_srad", "koi_smass", "koi_kepmag"]

# features = ["pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade", "pl_eqt", "pl_insol", "st_teff", "st_logg", "st_rad", "st_tmag"]
features = [
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_num_transits",
    "koi_count",
    "koi_model_snr",
    "koi_srad"
]

dataset = dataset[features]


dataset = dataset.dropna()
labels = labels[dataset.index]

print(dataset)

# from xgboost import XGBClassifier
from SREDT import SREDTClassifier
# dataset.drop(columns=[""])
# clf = XGBClassifier(enable_categorical=True)
clf = SREDTClassifier(nb_processes=1, function_set={'add', 'mul', 'sub'}, verbose=1, max_depth=5, max_expression_height=3)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
from SREDT.utils import pickle
pickle(clf, "pkl/sredt_model_KOI_Alex.pkl")
preds = clf.predict(X_test)
clf.display("images/tree_less_KOI_Alex", features=features, labels=label_encoder.classes_)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
print(confusion_matrix(y_test, preds))
print("acc:", accuracy_score(y_test, preds))
print("f1:", f1_score(y_test, preds, average='weighted'))