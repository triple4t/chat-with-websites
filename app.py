import streamlit as st
from chatbot import load_and_index_website, build_rag_agent, ask

st.set_page_config(page_title="Website Chatbot", page_icon="🤖")
st.title("🤖 Chat with ANY Website")
st.caption("Powered by LangChain + Gemini 2.0 | FREE")

url = st.text_input("🔗 Enter website URL:", placeholder="https://example.com")

if url and st.button("Load Website"):
    with st.spinner("Reading website..."):
        try:
            vector_store = load_and_index_website(url)
            st.session_state.agent = build_rag_agent(vector_store)
            st.session_state.messages = []
            st.success("✅ Done! Ask anything about this website.")
        except Exception as e:
            st.error(f"Error: {e}")

if "agent" in st.session_state:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(st.session_state.agent, user_input)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})