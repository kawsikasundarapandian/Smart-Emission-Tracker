import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import datetime
st.set_page_config(page_title="Smart Emission Tracker", layout="wide")
conn = sqlite3.connect("climate_data.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS emissions (
    username TEXT,
    date TEXT,
    travel REAL,
    electricity REAL,
    total REAL,
    travel_mode TEXT
)
""")

conn.commit()
st.sidebar.title("⚙ Settings")
language = st.sidebar.selectbox("🌐 Language", ["English", "Tamil", "Hindi"])
theme = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {background-color: #0E1117; color: white;}
        </style>
    """, unsafe_allow_html=True)
translations = {
    "English": {
        "title": "Climate Impact Pro Dashboard",
        "travel": "Travel Emission",
        "electricity": "Electricity Emission",
        "total": "Total Emission",
        "history": "Emission History",
        "goal": "Monthly Goal"
    },
    "Tamil": {
        "title": "காலநிலை தாக்கம் டாஷ்போர்டு",
        "travel": "பயண வெளியீடு",
        "electricity": "மின்சார வெளியீடு",
        "total": "மொத்த வெளியீடு",
        "history": "வெளியீட்டு வரலாறு",
        "goal": "மாத இலக்கு"
    },
    "Hindi": {
        "title": "जलवायु प्रभाव डैशबोर्ड",
        "travel": "यात्रा उत्सर्जन",
        "electricity": "बिजली उत्सर्जन",
        "total": "कुल उत्सर्जन",
        "history": "उत्सर्जन इतिहास",
        "goal": "मासिक लक्ष्य"
    }
}

t = translations[language]
if "user" not in st.session_state:

    st.title("🔐 Smart Emission Tracker")

    menu = st.radio("Select Option", ["Login", "Register"], horizontal=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if menu == "Register":
            if st.button("Create Account"):
                try:
                    c.execute("INSERT INTO users VALUES (?, ?)", (username, password))
                    conn.commit()
                    st.success("Account Created Successfully!")
                except:
                    st.error("Username already exists")

        if menu == "Login":
            if st.button("Login"):
                c.execute("SELECT * FROM users WHERE username=? AND password=?",
                          (username, password))
                if c.fetchone():
                    st.session_state["user"] = username
                    st.success("Login Successful 🎉")
                else:
                    st.error("Invalid Credentials")
else:

    st.title(f"🌍 {t['title']}")
    st.markdown("## 📥 Enter Daily Emission Data")

    col1, col2 = st.columns(2)

    with col1:
        travel_mode = st.selectbox(
            "Travel Mode",
            ["Car", "Bike", "Bus", "Train", "Flight"]
        )
        distance = st.number_input("Distance (km)", min_value=0.0)

    with col2:
        units = st.number_input("Electricity (kWh)", min_value=0.0)

    emission_factors = {
        "Car": 0.21,
        "Bike": 0.05,
        "Bus": 0.10,
        "Train": 0.06,
        "Flight": 0.25
    }

    travel_emission = distance * emission_factors[travel_mode]
    electricity_emission = units * 0.82
    total_emission = travel_emission + electricity_emission

    col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
    with col_save2:
        if st.button("💾 Save Data"):
            today = str(datetime.date.today())
            c.execute("INSERT INTO emissions VALUES (?, ?, ?, ?, ?, ?)",
                      (st.session_state["user"], today,
                       travel_emission, electricity_emission,
                       total_emission, travel_mode))
            conn.commit()
            st.success("Saved Successfully!")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"🚗 {t['travel']}", f"{travel_emission:.2f} kg")
    col2.metric(f"💡 {t['electricity']}", f"{electricity_emission:.2f} kg")
    col3.metric(f"🌍 {t['total']}", f"{total_emission:.2f} kg")

    st.markdown("---")
    df = pd.read_sql_query(
        "SELECT * FROM emissions WHERE username=?",
        conn,
        params=(st.session_state["user"],)
    )

    if not df.empty:

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        st.subheader(f"📈 {t['history']}")
        st.plotly_chart(px.line(df, x="date", y="total", markers=True),
                        use_container_width=True)
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month")["total"].sum().reset_index()
        monthly["month"] = monthly["month"].astype(str)

        st.plotly_chart(px.bar(monthly, x="month", y="total",
                               title="Monthly Emission"),
                        use_container_width=True)
        mode_count = df["travel_mode"].value_counts().reset_index()
        mode_count.columns = ["Mode", "Count"]

        st.plotly_chart(px.pie(mode_count,
                               names="Mode",
                               values="Count",
                               title="Travel Mode Usage"),
                        use_container_width=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_emission,
            title={'text': "Carbon Level"},
            gauge={
                'axis': {'range': [0, 200]},
                'steps': [
                    {'range': [0, 70], 'color': "green"},
                    {'range': [70, 140], 'color': "yellow"},
                    {'range': [140, 200], 'color': "red"}
                ],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        score = max(0, 100 - (total_emission / 200) * 100)
        st.metric("🌟 Sustainability Score", f"{score:.0f}/100")
        if len(df) > 5:
            df["day"] = np.arange(len(df))
            model = RandomForestRegressor(n_estimators=100)
            model.fit(df[["day"]], df["total"])

            future = np.array([[len(df) + i] for i in range(1, 8)])
            pred = model.predict(future)

            future_dates = pd.date_range(
                start=df["date"].max() + pd.Timedelta(days=1),
                periods=7
            )

            forecast = pd.DataFrame({
                "date": future_dates,
                "Prediction": pred
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["total"],
                mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(
                x=forecast["date"],
                y=forecast["Prediction"],
                mode="lines+markers", name="Forecast"))
            fig.update_layout(title="7-Day AI Forecast")

            st.plotly_chart(fig, use_container_width=True)
        st.sidebar.subheader(f"🎯 {t['goal']}")
        goal = st.sidebar.number_input("Set Goal (kg)", min_value=0.0)

        if goal > 0:
            current_month = df["date"].dt.to_period("M").max()
            current_total = df[df["date"].dt.to_period("M")
                               == current_month]["total"].sum()

            if current_total <= goal:
                st.success("🎉 Goal Achieved!")
            else:
                st.warning("⚠️ Goal Exceeded!")

        st.download_button("📥 Download CSV",
                           df.to_csv(index=False),
                           "Emission_Data.csv",
                           "text/csv")

    else:
        st.info("No data available yet.")
    if st.button("🚪 Logout"):
        del st.session_state["user"]
        st.rerun()
    st.markdown("---")
    st.subheader("🤖 Eco Safety Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask me about emission safety or eco tips:")

    def eco_chatbot(query):
        query = query.lower()

        if "reduce" in query:
            return "Use public transport, cycle more, and reduce electricity usage."
        elif "car" in query:
            return "Try carpooling or switch to electric vehicles."
        elif "electricity" in query:
            return "Switch off unused devices and use LED bulbs."
        elif "flight" in query:
            return "Avoid flights for short distances and choose trains when possible."
        elif "climate" in query:
            return "Reducing greenhouse gas emissions helps fight climate change."
        elif "sustainable" in query:
            return "Adopt renewable energy and eco-friendly transportation."
        else:
            return "Try reducing travel emissions and saving electricity daily."

    if st.button("Ask"):
        if user_input.strip() != "":
            response = eco_chatbot(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑 {sender}:** {message}")
        else:
            st.markdown(f"**🌱 {sender}:** {message}")
