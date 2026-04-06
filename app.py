import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv("reels_datasets.csv")

# -------------------------------
# TRAIN MODEL
# -------------------------------
X = data[["Hashtags","Likes","Comments","Shares","Followers","WatchTime","PostingTime","Genre"]]
y = data["Views"]

model = LinearRegression()
model.fit(X, y)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Reels Views Predictor")
st.title("Reels Views Predictor")

# -------------------------------
# INPUT
# -------------------------------
st.subheader("Enter Input Values")

col1, col2 = st.columns(2)

with col1:
    hashtags = st.number_input("Hashtags", min_value=0)
    likes = st.number_input("Likes", min_value=0)
    comments = st.number_input("Comments", min_value=0)
    shares = st.number_input("Shares", min_value=0)

with col2:
    followers = st.number_input("Followers", min_value=0)
    watch_time = st.number_input("Watch Time (%)", min_value=0)
    posting_time = st.selectbox("Posting Time", ["Morning","Afternoon","Evening","Night"])
    genre = st.selectbox("Genre", ["Entertainment","Education","Comedy","Tech","Lifestyle"])

time_map = {"Morning":1,"Afternoon":2,"Evening":3,"Night":4}
genre_map = {"Entertainment":1,"Education":2,"Comedy":3,"Tech":4,"Lifestyle":5}

posting_time_val = time_map[posting_time]
genre_val = genre_map[genre]

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Views"):

    input_data = np.array([[hashtags, likes, comments, shares,
                            followers, watch_time,
                            posting_time_val, genre_val]])

    pred = model.predict(input_data)[0]
    st.write("Predicted Views =", round(pred,2))

# -------------------------------
# CONCEPTS
# -------------------------------
st.subheader("Quantitative Analysis")

concept = st.selectbox("Select Concept",
                       ["Correlation","MLE","Hypothesis Testing"])

# -------------------------------
# CORRELATION
# -------------------------------
if concept == "Correlation":

    x = data["Likes"]
    y = data["Views"]

    r = x.corr(y)

    st.write("Formula:")
    st.latex(r"r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}")

    st.write("Mean of Likes =", round(x.mean(),2))
    st.write("Mean of Views =", round(y.mean(),2))

    st.write("Final Answer:")
    st.write("r =", round(r,2))

# -------------------------------
# MLE
# -------------------------------
if concept == "MLE":

    views = data["Views"]

    mean = views.mean()
    var = views.var()

    st.write("Formula:")
    st.latex(r"\mu = \frac{\sum x}{n}")
    st.latex(r"\sigma^2 = \frac{\sum (x - \mu)^2}{n}")

    st.write("Mean =", round(mean,2))
    st.write("Variance =", round(var,2))

# -------------------------------
# HYPOTHESIS TESTING
# -------------------------------
if concept == "Hypothesis Testing":

    x = data["Likes"]
    y = data["Views"]

    r, p = stats.pearsonr(x, y)

    alpha = 0.05

    st.write("H0: Likes do not affect Views")
    st.write("H1: Likes affect Views")

    st.write("Alpha =", alpha)

    # SIMPLE FORMULA
    st.write("Decision Rule:")
    st.write("If p-value < alpha → Reject H0")
    st.write("If p-value > alpha → Accept H0")

    # p-value display
    if p < 0.001:
        st.write("Calculated p-value < 0.001")
    else:
        st.write("Calculated p-value =", round(p,2))

    # decision
    if p < alpha:
        decision = "Reject H0"
    else:
        decision = "Accept H0"

    st.write("Decision:", decision)

    # -------------------------------
    # NORMAL CURVE
    # -------------------------------
    x_vals = np.linspace(-4, 4, 100)
    y_vals = stats.norm.pdf(x_vals)

    fig, ax = plt.subplots()

    ax.plot(x_vals, y_vals)

    # critical value
    z_crit = stats.norm.ppf(1 - alpha/2)

    ax.axvline(z_crit)
    ax.axvline(-z_crit)

    # rejection region
    ax.fill_between(x_vals, y_vals, where=(x_vals >= z_crit), color='red', alpha=0.5)
    ax.fill_between(x_vals, y_vals, where=(x_vals <= -z_crit), color='red', alpha=0.5)

    ax.set_title("Normal Distribution Curve")

    st.pyplot(fig)

    st.write("Critical Value =", round(z_crit,2))

    if p < alpha:
        st.write("p-value lies in rejection region → Reject H0")
    else:
        st.write("p-value lies in acceptance region → Accept H0")