from datetime import datetime
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

GLOBAL = {"target_key": "HeartDisease", "feature_names": []}


@st.cache_data
def load_data():
    data = pd.read_csv("C:/Users/miss-/OneDrive/Документы/Nir2/heart.csv")
    data = data.dropna()

    NUM_COLUMNS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
    CAT_COLUMNS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

    mms = MinMaxScaler()
    le = LabelEncoder()

    for col in NUM_COLUMNS:
        data[col] = mms.fit_transform(data[[col]])

    for col in CAT_COLUMNS:
        data[col] = le.fit_transform(data[col])

    data = data.drop(columns=["RestingECG"])

    return data


@st.cache_data
def preprocess_data(_):
    X_data, y_data = (
        data.drop(columns=[GLOBAL["target_key"]]),
        data[[GLOBAL["target_key"]]],
    )
    X_test, X_train, y_test, y_train = train_test_split(X_data, y_data)

    GLOBAL["feature_names"] = [column for column in X_test.columns]
    return X_test, X_train, y_test, y_train


def class_accuracy_score(y_true: pd.DataFrame, y_pred: np.ndarray):
    unique, counts = np.unique(y_true[GLOBAL["target_key"]], return_counts=True)
    y_true_values = y_true[GLOBAL["target_key"]].values
    result = {}
    for value in unique:
        result[value] = 0
    for index in range(len(y_true_values)):
        if y_true_values[index] == y_pred[index]:
            result[y_true_values[index]] += 1
    for index in range(len(unique)):
        result[unique[index]] /= counts[index]
    result["all"] = accuracy_score(y_true, y_pred)
    return result


st.sidebar.header("Logistic Regression")
data = load_data()

C_range = [float(10 ** (power / 100)) for power in range(-200, 201)]
solver_range = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
class_weight_range = [{0: i, 1: 1000 - i} for i in range(1, 1000)]

C = None
solver = None
class_weight = None

C = st.sidebar.slider("C (10^x):", min_value=-2.0, max_value=2.0, step=0.1, value=1.0)

solver = st.sidebar.radio(
    "solver: ", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
)

class_weight = st.sidebar.slider(
    "class_weight, w(0) + w(1) = 1000, w(0):",
    min_value=0,
    max_value=1000,
    step=1,
    value=500,
)


my_cmap = sns.color_palette("vlag", as_cmap=True)
if st.checkbox("Show correlation matrix"):
    fig1, ax = plt.subplots()
    sns.heatmap(data.corr(), cmap=my_cmap, ax=ax, annot=True, fmt=".2f")
    st.pyplot(fig1)

data_len = data.shape[0]

X_train, X_test, y_train, y_test = preprocess_data(data)
lr = LogisticRegression(
    C=10**C,
    solver=solver,
    class_weight={0: class_weight, 1: 1000 - class_weight},
)
lr.fit(X_train, y_train)

st.header("Model quality")

y_pred = lr.predict(X_test)

st.subheader("Accuracy")
acc_df = pd.DataFrame.from_dict(
    class_accuracy_score(y_test, y_pred), orient="index", columns=["accuracy"]
)
st.dataframe(acc_df)

st.subheader("Confusion matrix")
fig2, ax = plt.subplots(figsize=(10, 5))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, normalize="true", ax=ax, cmap=my_cmap
)
fig2.suptitle("Confusion Matrix")
st.pyplot(fig2)


def draw_roc_curve(y_true, y_score, ax, pos_label=1, average="micro"):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    lw = 2
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc_value,
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right")


st.subheader("ROC curve")
fig3, ax = plt.subplots(figsize=(10, 5))
draw_roc_curve(y_test.values, y_pred, ax)
st.pyplot(fig3)
