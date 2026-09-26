"""Entry point: `streamlit run app.py`. Defines the sidebar navigation."""
import streamlit as st

pages = [
    st.Page("board.py", title="Value board", icon="🎾", default=True),
    st.Page("pages/1_Match_Detail.py", title="Match detail", icon="🔎", url_path="match"),
    st.Page("pages/2_Manual_Lab.py", title="Manual lab", icon="🧪", url_path="lab"),
    st.Page("pages/3_Tracking.py", title="Tracking & CLV", icon="📈", url_path="tracking"),
    st.Page("pages/4_Data_Health.py", title="Model & data health", icon="🩺", url_path="health"),
]
st.navigation(pages).run()
