import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

def vector_db(urls):
    # Convert string of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]
    
    # PDF processing
    # loader = PyPDFLoader(pdf_path)
    # docs_list = loader.load()  # Load PDF as list of documents

    #split the text into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    #convert text chunks into embeddings and store in vector database
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text')
    )
    return vectorstore

def process_input(question,vectorstore):
    model_local = Ollama(model="mistral")
    retriever = vectorstore.as_retriever()
    
    #perform the RAG 
    
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

# Streamlit UI setup
st.title("Chat with Ollama 👋")

# Input fields
urls = st.text_area("Enter URLs separated by new lines", height=100)

# upload pdf
# uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
# if uploaded_file is not None:
#     st.session_state['vectordb'] = store_db(uploaded_file)

if st.button('Load Documents from Urls'):
    with st.spinner('Processing...'):
        st.session_state['vectordb'] = vector_db(urls)
        st.markdown("Done")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me👋"):
    # Display user message in chat message container
    with st.chat_message("user"):
        
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner('Processing...'):
            answer = process_input(prompt,st.session_state['vectordb'])
            response = st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": response})