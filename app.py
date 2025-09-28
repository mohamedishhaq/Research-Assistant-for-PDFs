import streamlit as st
from retriever import get_query_engine
import os

st.set_page_config(page_title="📚 Research Assistant for PDFs", layout="wide")

def main():
    st.title("📚 Research Assistant for PDFs")

    # Upload PDFs
    uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

    pdf_dir = "uploaded_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    # Save uploaded PDFs
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(pdf_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded.")

    # Load query engine
    query_engine = get_query_engine(pdf_dir if uploaded_files else "sample_papers")

    # Ask questions
    user_query = st.text_input("Ask a question about your papers:")
    if st.button("Search") and user_query:
        response = query_engine.query(user_query)

        st.subheader("🔍 Answer")
        st.write(response.response)

        with st.expander("📑 Sources"):
            for node in response.source_nodes:
                st.markdown(f"- {node.node.metadata.get('file_name', 'Unknown')}")

if __name__ == "__main__":
    main()
