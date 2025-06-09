import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# === Load environment variables ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# === Load CSV files ===
ayurveda_df = pd.read_csv("ayurveda_knowledge.csv")
hospital_df = pd.read_csv("Odisha_Hospitals_BloodBanks.csv")
bloodbank_df = pd.read_csv("rew (1).csv")

# === Set up LLM with Groq ===
llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)

# === Streamlit UI Setup ===
st.set_page_config(page_title="BloodBank + Ayurveda ChatBot", page_icon="🩸", layout="centered")
st.title("🩸 Odisha BloodBank + Ayurveda ChatBot")
st.markdown("Ask anything about **Ayurvedic remedies** or **hospital/blood bank availability** in Odisha.")

# === Initialize chat session ===
if "chat" not in st.session_state:
    st.session_state.chat = []

# === Query Handler ===
def handle_query(query):
    query_lower = query.lower()

    # Check for Ayurvedic symptoms
    for _, row in ayurveda_df.iterrows():
        if row["symptom"].lower() in query_lower:
            return f"🧘 Ayurvedic remedy for *{row['symptom']}*: {row['ayurvedic_remedy']}"

    # Search hospitals or blood banks
    results = []

    for _, row in hospital_df.iterrows():
        if row["City"].lower() in query_lower or row["District"].lower() in query_lower:
            results.append(f"🏥 {row['Hospital Name']} - {row['City']} ({row['District']}) | Contact: {row['Contact Number']}")

    for _, row in bloodbank_df.iterrows():
        if (
            row["City"].lower() in query_lower or
            row["District"].lower() in query_lower or
            row["Blood Bank Name"].lower() in query_lower
        ):
            results.append(
                f"🩸 {row['Blood Bank Name']} - {row['City']} ({row['District']}) | Contact: {row['Contact Number']}, Affiliation: {row['Affiliation']}"
            )

    if results:
        return "\n\n".join(results)

    # Fallback to LLM if no match
    prompt_template = PromptTemplate(
        template="You are a helpful assistant. Try to answer user queries:\n\nQuestion: {question}\nAnswer:",
        input_variables=["question"]
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(question=query)

# === User input ===
user_query = st.text_input("Enter your question", key="user_input")

# === Button & Response ===
if st.button("Ask"):
    if user_query.strip():
        reply = handle_query(user_query)
        st.session_state.chat.append(("🧑 You", user_query))
        st.session_state.chat.append(("🤖 Bot", reply))
        # Do NOT stop or rerun — let it continue to render below

# === Chat History ===
if st.session_state.chat:
    st.subheader("Chat History")
    for speaker, text in reversed(st.session_state.chat):
        st.markdown(f"**{speaker}:** {text}")
