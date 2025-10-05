import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from SREDT.utils import unpickle
from tree_to_html import tree_to_html
import ast

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
visible_columns = st.multiselect("Select columns to show (deselect to hide)", key="visible_cols", options=all_columns, default=all_columns)

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
st.sidebar.header("Preprocessing")
target = st.sidebar.selectbox("Select target column (classification)", options=[None] + visible_columns)
if target:
    st.code(f"Labels: {df[target].value_counts().keys().tolist()}")
    # Option to enter a list of allowed labels (label filtering)
    st.sidebar.markdown("**Label filtering (optional)**")
    label_filter_text = st.sidebar.text_area(
        "Enter a Python list of allowed labels. Only rows with these labels will be kept.\nExample: ['A', 'B', 'C']",
        value="['CONFIRMED', 'FALSE POSITIVE']", height=10
    )

    allowed_labels = None
    if label_filter_text.strip():
        try:
            allowed_labels = ast.literal_eval(label_filter_text)
            if not isinstance(allowed_labels, (list, tuple)):
                st.sidebar.error("Label filter must be a list or tuple.")
                allowed_labels = None
        except Exception as e:
            st.sidebar.error(f"Could not parse label filter: {e}")
            allowed_labels = None

        if allowed_labels:
            # Filter rows to only those with allowed labels
            df = df[df[target].isin(allowed_labels)]
            st.sidebar.success("Applied label filtering.")


if target is None:
    st.sidebar.info("Select a target column to enable modeling")
    st.stop()

# Allow entering a Python-style list of features (e.g. ['a','b']) or comma-separated list
st.sidebar.markdown("**Feature list input (optional)**")
feature_list_text = st.sidebar.text_area("Paste Python list or comma-separated names", value='["koi_period","koi_impact","koi_duration","koi_depth","koi_num_transits","koi_count","koi_model_snr","koi_srad"]', height=10)

# features multiselect already defined above using visible_columns when appropriate
features = None

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
        visible_columns = features + [target]
        st.dataframe(df[visible_columns].head())
        st.sidebar.success("Applied feature filtering.")


if features is None:
    features = [c for c in visible_columns if c != target]

# Preprocessing options
st.sidebar.header("Modeling")

st.write("## Selected features and target")
st.write("Features:", features)
st.write("Target:", target)

# Simple preprocessing
data = df[features + [target]].copy()

data = data.dropna()

X = data[features]
y = data[target]

uploaded_file = st.sidebar.file_uploader("Upload and test model", type=["pkl"])
if uploaded_file is not None:
    clf = unpickle(uploaded_file)
    label_encoder = LabelEncoder()
    
    y = label_encoder.fit_transform(y)
    tree_html = tree_to_html(clf, features, label_encoder.classes_)
    st.components.v1.html(tree_html, height=600)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)

    st.write(f"### Test accuracy: {acc:.4f}")
    st.write("#### Classification report")
    st.text(classification_report(y, preds))

    st.write("#### Confusion matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y, preds), annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

st.sidebar.write("or")

if st.sidebar.button("Train model"):
    pass