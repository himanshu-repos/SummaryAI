import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit app configuration
st.set_page_config(page_title="Langchain: Summarize Text from Website", page_icon="🦜")
st.title("🦜 Langchain: Summarize Text From Website")
st.subheader('Summarize URL')

# Sidebar input for API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# URL input
generic_url = st.text_input("URL", label_visibility="collapsed")

# Prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarize button
if st.button("Summarize the Content from Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It may be a website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Initialize LLM here AFTER we have the API key
                llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

                # Load content based on URL type
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                          "Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                data = loader.load()

                # Run the summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(data)

                st.success(output_summary)

        except Exception as e:
            st.error(f"Exception: {e}")

