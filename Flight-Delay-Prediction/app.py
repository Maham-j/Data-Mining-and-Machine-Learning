import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


st.set_page_config(page_title="Flight Delay Prediction", layout="wide")
# -----------------------
# CACHED: Load and preprocess data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("flights.csv")
    df.dropna(inplace=True)
    df['Delay'] = df['arr_delay'].apply(lambda x: 1 if x > 15 else 0)
    df.drop(['arr_delay', 'time_hour', 'tailnum'], axis=1, inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

# -----------------------
# CACHED: Train model
# -----------------------
@st.cache_resource
def train_model(df):
    X = df.drop('Delay', axis=1)
    y = df['Delay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, y_train

# Load data and model
df, label_encoders = load_data()
model, X_test, y_test, y_train = train_model(df)

# -----------------------
# STREAMLIT INTERFACE
# -----------------------
st.title("✈️ Flight Delay Dashboard")

# Define tabs
tab1, tab2, tab3 = st.tabs(["📄 Data & Model Evaluation", "📊 Flight Data Analysis", "🤖 Delay Prediction"])

# ===========================
# 📄 TAB 1: Data & Model Evaluation
# ===========================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50))

    st.subheader("Model Performance on Test Set")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.markdown(f"**Accuracy**: `{accuracy * 100:.2f}%`")

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

# ===========================
# 📊 TAB 2: Visualizations
# ===========================
with tab2:
    st.header("Flight Data Visualizations")

    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Delay', palette='Set2', ax=ax1)
    ax1.set_title("Overall Flight Status (1 = Delayed)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.countplot(data=df, x='carrier', hue='Delay', palette='Set1', ax=ax2)
    ax2.set_title("Flight Status by Airline")
    st.pyplot(fig2)

    df['route'] = df['origin'].astype(str) + " → " + df['dest'].astype(str)
    top_routes = df[df['Delay'] == 1]['route'].value_counts().nlargest(10)
    fig3, ax3 = plt.subplots()
    top_routes.plot(kind='barh', color='salmon', ax=ax3)
    ax3.set_title("Top 10 Routes with Most Delays")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(10,6))
    sns.histplot(data=df, x='hour', hue='Delay', multiple='stack', bins=24, palette='coolwarm', ax=ax4)
    ax4.set_title("Delays by Departure Hour")
    st.pyplot(fig4)

    fig5 = px.scatter(df, x='distance', y='air_time',
                      color=df['Delay'].map({1:'Delayed', 0:'On Time'}),
                      title="Flight Distance vs Air Time", labels={'color':'Flight Status'})
    st.plotly_chart(fig5)

    st.subheader("Average Departure Time by Carrier")
    st.bar_chart(df.groupby('carrier')['dep_time'].mean())

# ===========================
# 🤖 TAB 3: Delay Prediction
# ===========================
with tab3:
    st.header("Will Your Flight Be Delayed?")

    year = st.selectbox("Year", df['year'].unique())
    month = st.selectbox("Month", df['month'].unique())
    day = st.selectbox("Day", df['day'].unique())
    dep_time = st.slider("Departure Time (HHMM)", 0, 2359)
    sched_dep_time = st.slider("Scheduled Departure Time (HHMM)", 0, 2359)
    arr_time = st.slider("Arrival Time (HHMM)", 0, 2359)
    sched_arr_time = st.slider("Scheduled Arrival Time (HHMM)", 0, 2359)
    carrier = st.selectbox("Carrier", label_encoders['carrier'].classes_)
    flight = st.number_input("Flight Number", min_value=0)
    origin = st.selectbox("Origin", label_encoders['origin'].classes_)
    dest = st.selectbox("Destination", label_encoders['dest'].classes_)
    air_time = st.number_input("Air Time (minutes)", min_value=0.0)
    distance = st.number_input("Distance (miles)", min_value=0.0)
    hour = st.slider("Scheduled Hour", 0, 23)
    minute = st.slider("Scheduled Minute", 0, 59)

        # Get expected feature names from training data
    expected_features = X_test.columns.tolist()

    input_data = pd.DataFrame([{
        'year': year,
        'month': month,
        'day': day,
        'dep_time': dep_time,
        'sched_dep_time': sched_dep_time,
        'arr_time': arr_time,
        'sched_arr_time': sched_arr_time,
        'carrier': label_encoders['carrier'].transform([carrier])[0],
        'flight': flight,
        'origin': label_encoders['origin'].transform([origin])[0],
        'dest': label_encoders['dest'].transform([dest])[0],
        'air_time': air_time,
        'distance': distance,
        'hour': hour,
        'minute': minute
    }])

    # Align columns
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    if st.button("Predict Delay"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.error(f"⚠️ Flight will likely be **DELAYED** (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Flight will likely be **ON TIME** (Probability: {1 - prob:.2f})")
