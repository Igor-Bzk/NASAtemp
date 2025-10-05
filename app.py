import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Prevent running this script directly with `python app.py` which leads to missing Streamlit context.
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print("This app must be run with the Streamlit CLI: streamlit run app.py")
        import sys
        sys.exit(0)
except Exception:
    # If the import or check fails, allow execution; streamlit will raise appropriate errors.
    pass

st.set_page_config(page_title="CSV Classifier Explorer", layout="wide")

st.title("CSV Classifier Explorer")
from SREDT.utils import unpickle
from tree_to_html import tree_to_html

uploaded_file = st.file_uploader("Upload SREDT pkl file to visualize", type=["pkl"])
if uploaded_file is not None:
    clf = unpickle(uploaded_file)
    tree_html = tree_to_html(clf, None, None)
    st.components.v1.html(tree_html, height=600)

st.sidebar.header("Upload and settings")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"]) 

if uploaded_file is None:
    st.sidebar.info("Please upload a CSV file to begin")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# Simple column visibility control: multiselect that starts with all columns selected.
all_columns = df.columns.tolist()
visible_columns = st.multiselect("Select columns to show (deselect to hide)", options=all_columns, default=all_columns)

if not visible_columns:
    st.warning("No columns selected — selecting all columns by default.")
    visible_columns = all_columns.copy()


st.write("## Data Preview")
st.dataframe(df[visible_columns].head())

st.write("### Dataframe shape (visible columns)")
st.write((df.shape[0], len(visible_columns)))

with st.expander("Show full dataframe (visible columns)"):
    st.dataframe(df[visible_columns])

# Sidebar: target selection
st.sidebar.header("Modeling")
target = st.sidebar.selectbox("Select target column (classification)", options=[None] + visible_columns)

if target is None:
    st.sidebar.info("Select a target column to enable modeling")
    st.stop()

features = st.sidebar.multiselect("Select feature columns (empty = use all except target)", options=[c for c in visible_columns if c != target])
# Allow entering a Python-style list of features (e.g. ['a','b']) or comma-separated list
st.sidebar.markdown("**Feature list input (optional)**")
height = st.sidebar.slider("Feature input height (lines)", min_value=3, max_value=20, value=4)
feature_list_text = st.sidebar.text_area("Paste Python list or comma-separated names", value="", height=height)

# features multiselect already defined above using visible_columns when appropriate

if feature_list_text.strip():
    import ast
    parsed = None
    try:
        parsed = ast.literal_eval(feature_list_text)
    except Exception:
        # fallback: comma-separated
        parsed = [s.strip() for s in feature_list_text.split(',') if s.strip()]

    # ensure parsed is a list-like
    if isinstance(parsed, (list, tuple)):
        # validate against visible columns
        valid = [c for c in parsed if c in visible_columns and c != target]
        invalid = [c for c in parsed if c not in visible_columns]
        if invalid:
            st.sidebar.warning(f"Ignored unknown or hidden columns: {invalid}")
        if valid:
            features = valid
        else:
            st.sidebar.error("No valid feature names found in provided list; using multiselect/default.")

if not features:
    features = [c for c in visible_columns if c != target]

# Preprocessing options
st.sidebar.header("Preprocessing")
encode_categoricals = st.sidebar.checkbox("One-hot encode categorical features (pd.get_dummies)", value=False)
drop_na = st.sidebar.checkbox("Drop rows with NA in selected columns", value=True)

st.write("## Selected features and target")
st.write("Features:", features)
st.write("Target:", target)

# Simple preprocessing
data = df[features + [target]].copy()
if drop_na:
    data = data.dropna()

# Encode target if categorical
if data[target].dtype == object or data[target].dtype.name == 'category':
    data[target] = data[target].astype('category').cat.codes

# Optionally one-hot encode categorical features
if encode_categoricals:
    # only encode feature columns that are non-numeric
    cat_feats = data[features].select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_feats:
        data = pd.get_dummies(data, columns=cat_feats, drop_first=True)
        # update feature list
        features = [c for c in data.columns.tolist() if c != target]

X = data[features]
y = data[target]

# Sidebar model selection
model_name = st.sidebar.selectbox("Choose classifier", ["Random Forest", "Logistic Regression"]) 

# Train/test split
test_size = st.sidebar.slider("Test set proportion", 0.1, 0.5, 0.25)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

if st.sidebar.button("Train model"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

        # Simple scaling for logistic regression
        scaler = None
        if model_name == "Logistic Regression":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if model_name == "Random Forest":
            model = RandomForestClassifier(random_state=int(random_state))
        else:
            model = LogisticRegression(max_iter=1000, random_state=int(random_state))

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        st.write(f"### Test accuracy: {acc:.4f}")
        st.write("#### Classification report")
        st.text(classification_report(y_test, preds))

        st.write("#### Confusion matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        # Save model artifact for download
        artifact = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'target': target
        }
    except Exception as e:
        st.error(f"Training failed: {e}")
