import uuid
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF RAG")

st.title("PDF RAG Chat")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

if (
    uploaded_file
    and uploaded_file.name != st.session_state.uploaded_file_name
):
    with st.spinner("Uploading PDF..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "application/pdf",
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
        )

        if response.ok:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success("PDF uploaded successfully.")
        else:
            st.error("Upload failed.")

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question")

if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "query": query,
                },
            )

            if response.ok:
                data = response.json()

                answer = data["answer"]

                st.markdown(answer)

                with st.expander("Sources"):
                    for source in data["retrieved_sources"]:
                        st.write(source["content"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )
            else:
                st.error("Request failed.")