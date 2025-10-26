import os
import streamlit as st
st.set_page_config(page_title="Research Tool", layout="wide")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
import requests
import re
from datetime import datetime, timedelta
import yake
from collections import Counter
import PyPDF2
import io
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY", "c8ad3a77b1aa46a68541fe8a9b56ac7f") 

if not openai_api_key:
    st.error("❌ OpenAI API key not found! Please add your OPENAI_API_KEY to the .env file")
    st.stop()

def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #76ABAE;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
set_background_color()

st.title("Research Tool")

tab1, tab2, tab3 = st.tabs(["🌐 URLs", "📄 PDF Upload", "📝 Direct Text"])

st.sidebar.header("📰 News Settings")
enable_news = st.sidebar.checkbox("Enable News Articles", value=True)
num_news_articles = st.sidebar.slider("Number of News Articles", 1, 10, 5)
show_keywords = st.sidebar.checkbox("Show Extracted Keywords", value=False)
debug_yake = st.sidebar.checkbox("Debug YAKE Output", value=False)

st.sidebar.header("🔍 Debug Settings")
enable_debug = st.sidebar.checkbox("Enable Debug Mode", value=False)
show_chunks = st.sidebar.checkbox("Show Retrieved Chunks", value=False)
show_processing_steps = st.sidebar.checkbox("Show Processing Steps", value=False)
show_storage_info = st.sidebar.checkbox("Show Storage Information", value=False)

start_processing = False
input_data = None
data_type = None

with tab1:
    st.header("🌐 URLs")
    st.write("Enter up to 3 URLs to process and ask questions about their content.")
    
    input_urls = [st.text_input(f"URL {i+1}", key=f"url_{i}") for i in range(3)]
    start_processing = st.button("🚀 Process URLs", key="process_urls")
    
    if start_processing:
        valid_urls = [url for url in input_urls if url.strip()]
        if valid_urls:
            input_data = valid_urls
            data_type = "urls"
        else:
            st.error("Please enter at least one valid URL")

with tab2:
    st.header("📄 Upload PDF")
    st.write("Upload a PDF file and ask questions about its content.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        key="pdf_uploader"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.write(f"📊 File size: {uploaded_file.size} bytes")
        
        if st.button("🚀 Process PDF", key="process_pdf"):
            input_data = uploaded_file
            data_type = "pdf"

with tab3:
    st.header("📝 Direct Text Input")
    st.write("Paste or type your text content directly and ask questions about it.")
    
    text_input = st.text_area(
        "Enter your text content:",
        height=300,
        placeholder="Paste your text here...",
        key="text_input"
    )
    
    if st.button("🚀 Process Text", key="process_text"):
        if text_input.strip():
            input_data = text_input
            data_type = "text"
        else:
            st.error("Please enter some text content")

status_container = st.container()

faiss_directory = "faiss_store"
language_model = ChatOpenAI(
    api_key=openai_api_key, 
    temperature=0.7, 
    max_tokens=1000,
    model="gpt-3.5-turbo"
)
llm_embeddings = OpenAIEmbeddings(api_key=openai_api_key)

expected_minimum_chunks = 10

def split_text(loaded_data):
    primary_delimiters = ['\n\n', '\n', '.', ',']
    fallback_delimiters = [';', ' ', '|']

    doc_splitter = RecursiveCharacterTextSplitter(separators=primary_delimiters, chunk_size=1000, chunk_overlap=200)
    split_documents = doc_splitter.split_documents(loaded_data)
    if not split_documents or len(split_documents) < expected_minimum_chunks:
        doc_splitter = RecursiveCharacterTextSplitter(separators=fallback_delimiters, chunk_size=1000, chunk_overlap=200)
        split_documents = doc_splitter.split_documents(loaded_data)

    return split_documents

def process_pdf_file(uploaded_file):
    try:
        pdf_content = uploaded_file.read()

        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_content = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text() + "\n"
        
        if not text_content.strip():
            return None, "No text content found in PDF"
        
        from langchain.schema import Document
        document = Document(
            page_content=text_content,
            metadata={"source": uploaded_file.name, "type": "pdf"}
        )
        
        return [document], None
        
    except Exception as e:
        return None, f"Error processing PDF: {str(e)}"

def process_text_input(text_content):
    try:
        if not text_content.strip():
            return None, "Please enter some text"
        
        from langchain.schema import Document
        document = Document(
            page_content=text_content,
            metadata={"source": "Direct Text Input", "type": "text"}
        )
        
        return [document], None
        
    except Exception as e:
        return None, f"Error processing text: {str(e)}"

def extract_keywords_with_yake(text, max_keywords=10):
    try:
        kw_extractor = yake.KeywordExtractor(
            lan="en",          
            n=3,                
            dedupLim=0.7,       
            top=20,             
            features=None       
        )
        
        keywords = kw_extractor.extract_keywords(text)
        if debug_yake:
            st.write("🔍 Raw YAKE output:", keywords[:5])
        
        extracted_keywords = []
        for kw_tuple in keywords[:max_keywords]:
            if isinstance(kw_tuple, tuple) and len(kw_tuple) >= 2:
                keyword = str(kw_tuple[1]).strip()
                if len(keyword) > 2 and keyword.replace(' ', '').isalpha():
                    extracted_keywords.append(keyword)
            elif isinstance(kw_tuple, str):
                keyword = kw_tuple.strip()
                if len(keyword) > 2 and keyword.replace(' ', '').isalpha():
                    extracted_keywords.append(keyword)
        
        return extracted_keywords
        
    except Exception as e:
        st.error(f"Error extracting keywords with YAKE: {str(e)}")
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who', 'this', 'that', 'these', 'those'}
        filtered_words = [word for word in words if word not in stop_words]
        return filtered_words[:max_keywords]

def extract_keywords_simple(text, max_keywords=10):
    try:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when',
            'where', 'why', 'who', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'from', 'into', 'during', 'including', 'until', 'against',
            'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'to', 'of', 'in',
            'for', 'on', 'with', 'as', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
            'will', 'just', 'don', 'should', 'now'
        }
        filtered_words = [word for word in words if word not in stop_words]
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(max_keywords)]
        
    except Exception as e:
        st.error(f"Error in simple keyword extraction: {str(e)}")
        return []

def extract_keywords_from_query(query):
    yake_keywords = extract_keywords_with_yake(query, max_keywords=8)
    
    valid_yake_keywords = []
    for kw in yake_keywords:
        if isinstance(kw, str) and kw.replace(' ', '').isalpha() and len(kw) > 2:
            valid_yake_keywords.append(kw)
    
    if valid_yake_keywords:
        return valid_yake_keywords
    else:
        return extract_keywords_simple(query, max_keywords=8)

def extract_keywords_from_documents(documents):
    all_text = ""
    for doc in documents:
        all_text += doc.page_content + " "
    
    yake_keywords = extract_keywords_with_yake(all_text, max_keywords=10)
    
    valid_yake_keywords = []
    for kw in yake_keywords:
        if isinstance(kw, str) and kw.replace(' ', '').isalpha() and len(kw) > 2:
            valid_yake_keywords.append(kw)
    
    if valid_yake_keywords:
        return valid_yake_keywords
    else:
        return extract_keywords_simple(all_text, max_keywords=10)

def fetch_news_articles(keywords, num_articles=5):
    if not keywords:
        return []
    
    try:
        search_query = " OR ".join(keywords)
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': search_query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': num_articles,
            'apiKey': news_api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'ok' and data['articles']:
            return data['articles']
        else:
            return []
            
    except Exception as e:
        st.error(f"❌ Error fetching news: {str(e)}")
        return []

def display_news_articles(articles):
    if not articles:
        return
    
    st.subheader("📰 Related News Articles")
    
    for i, article in enumerate(articles, 1):
        with st.expander(f"📄 {article['title'][:80]}..." if len(article['title']) > 80 else f"📄 {article['title']}"):
            st.write(f"**Source:** {article['source']['name']}")
            st.write(f"**Published:** {article['publishedAt'][:10]}")
            
            if article.get('description'):
                st.write(f"**Description:** {article['description']}")
            
            if article.get('url'):
                st.write(f"**Read more:** [Link]({article['url']})")
            
            if article.get('urlToImage'):
                st.image(article['urlToImage'], width=300)
if input_data is not None and data_type is not None:
    status_container.info(f"Processing {data_type}...")
    
    if show_processing_steps:
        st.info("🔍 **Debug: Starting Processing Pipeline**")
    
    try:
        loaded_data = None
        
        if data_type == "urls":
            if show_processing_steps:
                st.info(f"📥 **Step 1: Loading URLs** - Processing {len(input_data)} URL(s)")
                for i, url in enumerate(input_data, 1):
                    st.write(f"   {i}. {url}")
            
            url_loader = WebBaseLoader(input_data)
            loaded_data = url_loader.load()
            
            if show_processing_steps:
                st.info(f"✅ **Step 1 Complete** - Loaded {len(loaded_data)} raw documents")
            
        elif data_type == "pdf":
            if show_processing_steps:
                st.info(f"📄 **Step 1: Processing PDF** - {input_data.name}")
                st.write(f"   File size: {input_data.size} bytes")
            
            loaded_data, error = process_pdf_file(input_data)
            if error:
                status_container.error(f"❌ {error}")
                loaded_data = None
            else:
                if show_processing_steps:
                    st.info(f"✅ **Step 1 Complete** - Extracted text from PDF")
                
        elif data_type == "text":
            if show_processing_steps:
                st.info(f"📝 **Step 1: Processing Text** - {len(input_data)} characters")
                st.write(f"   Preview: {input_data[:100]}...")
            
            loaded_data, error = process_text_input(input_data)
            if error:
                status_container.error(f"❌ {error}")
                loaded_data = None
            else:
                if show_processing_steps:
                    st.info(f"✅ **Step 1 Complete** - Created document from text")
        
        if loaded_data:
            if show_processing_steps:
                st.info(f"✂️ **Step 2: Text Splitting** - Creating chunks from {len(loaded_data)} document(s)")
            
            split_documents = split_text(loaded_data)
            
            if not split_documents:
                status_container.error("Failed to split documents into chunks")
            else:
                if show_processing_steps:
                    st.info(f"✅ **Step 2 Complete** - Created {len(split_documents)} chunks")
                    st.write(f"   Average chunk size: {sum(len(chunk.page_content) for chunk in split_documents) // len(split_documents)} characters")
                
                if show_storage_info:
                    st.info(f"💾 **Step 3: Creating Vector Store** - Generating embeddings and saving to FAISS")
                
                vectorindex_openai = FAISS.from_documents(split_documents, llm_embeddings)
                
                if show_storage_info:
                    st.info(f"💾 **Step 4: Saving to Disk** - Storing in '{faiss_directory}' directory")
                
                vectorindex_openai.save_local("faiss_store")
                
                if show_storage_info:
                    st.info(f"📁 **Storage Location:** `{os.path.abspath(faiss_directory)}`")
                    st.write(f"   Files created:")
                    st.write(f"   - `{faiss_directory}/index.faiss` (vector index)")
                    st.write(f"   - `{faiss_directory}/index.pkl` (metadata)")
                
                source_info = ""
                if data_type == "urls":
                    source_info = f"from {len(input_data)} URL(s)"
                elif data_type == "pdf":
                    source_info = f"from PDF: {input_data.name}"
                elif data_type == "text":
                    source_info = "from direct text input"
                
                status_container.success(f"✅ {data_type.title()} processed successfully! Created {len(split_documents)} document chunks {source_info}.")
                
                if enable_debug:
                    st.success("🎉 **Processing Complete!** Ready to answer questions.")
                
    except Exception as e:
        status_container.error(f"❌ Error processing {data_type}: {str(e)}")
        if data_type == "urls":
            st.error("Make sure the URLs are accessible and contain readable content.")
        elif data_type == "pdf":
            st.error("Make sure the PDF file is valid and contains readable text.")
        elif data_type == "text":
            st.error("There was an error processing your text input.")

if os.path.isdir(faiss_directory):
    
    try:
        sample_index = FAISS.load_local(faiss_directory, llm_embeddings)
        
    except:
        pass

st.divider()
st.header("💬 Ask Questions")

user_query = st.text_input("🔍 Type your query here", placeholder="Ask anything about your processed content...")

if user_query:
    if os.path.isdir(faiss_directory):
        try:
            if show_processing_steps:
                st.info(f"🔍 **Query Processing: Loading Vector Store**")
                st.write(f"   Loading from: `{os.path.abspath(faiss_directory)}`")
            
            loaded_faiss_index = FAISS.load_local(faiss_directory, llm_embeddings)
            
            if show_processing_steps:
                st.info(f"🔍 **Query Processing: Setting up Retrieval**")
                st.write(f"   Retrieving top 4 most relevant chunks")
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=language_model, 
                chain_type="stuff", 
                retriever=loaded_faiss_index.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True
            )
            
            if show_processing_steps:
                st.info(f"🔍 **Query Processing: Searching and Generating Answer**")
            
            with st.spinner("Searching for relevant information..."):
                query_result = qa_chain({"query": user_query})
            
            if show_chunks:
                st.info(f"📄 **Retrieved Chunks (Top {len(query_result.get('source_documents', []))}):**")
                for i, doc in enumerate(query_result.get("source_documents", []), 1):
                    with st.expander(f"Chunk {i} (Score: Similarity-based)"):
                        st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"**Content Length:** {len(doc.page_content)} characters")
                        st.write(f"**Content Preview:**")
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            
            st.subheader("🤖 Answer")
            st.write(query_result["result"])
            
          
            if show_processing_steps:
                st.info(f"🔑 **Keyword Extraction: Starting**")
            
            query_keywords = extract_keywords_from_query(user_query)
            document_keywords = extract_keywords_from_documents(query_result.get("source_documents", []))
            all_keywords = list(set(query_keywords + document_keywords))
            
            if show_processing_steps:
                st.info(f"🔑 **Keyword Extraction: Complete**")
                st.write(f"   Query keywords: {len(query_keywords)}")
                st.write(f"   Document keywords: {len(document_keywords)}")
                st.write(f"   Combined unique keywords: {len(all_keywords)}")
            
            if all_keywords and enable_news:
                if show_keywords:
                    st.info(f"🔍 Query keywords: {', '.join(query_keywords[:5])}")
                    st.info(f"📄 Document keywords: {', '.join(document_keywords[:5])}")
                    st.info(f"🎯 Combined keywords: {', '.join(all_keywords[:8])}")
                
                if show_processing_steps:
                    st.info(f"📰 **News Search: Starting**")
                    st.write(f"   Searching for: {', '.join(all_keywords[:8])}")
                    st.write(f"   Number of articles requested: {num_news_articles}")
                
                st.info(f"🔍 Searching for news related to: {', '.join(all_keywords[:8])}")
                with st.spinner("Fetching related news articles..."):
                    news_articles = fetch_news_articles(all_keywords, num_articles=num_news_articles)
                    
                    if show_processing_steps:
                        st.info(f"📰 **News Search: Complete**")
                        st.write(f"   Articles found: {len(news_articles)}")
                    
                    if news_articles:
                        display_news_articles(news_articles)
                    else:
                        st.info("No recent news articles found for these keywords.")
            if query_result.get("source_documents"):
                st.subheader("📚 Sources")

                unique_sources = {}
                for i, doc in enumerate(query_result["source_documents"], 1):
                    source_url = doc.metadata.get('source', 'Unknown')
                    if source_url not in unique_sources:
                        unique_sources[source_url] = {
                            'chunks': [],
                            'chunk_count': 0
                        }
                    unique_sources[source_url]['chunks'].append((i, doc))
                    unique_sources[source_url]['chunk_count'] += 1
                

                st.write("**Referenced Sources:**")
                for source_url, data in unique_sources.items():
                    chunk_count = data['chunk_count']
                    if chunk_count > 1:
                        st.write(f"• **{source_url}** ({chunk_count} chunks)")
                    else:
                        st.write(f"• **{source_url}**")
                for source_url, data in unique_sources.items():
                    with st.expander(f"📄 {source_url}"):
                        for chunk_num, doc in data['chunks']:
                            st.write(f"**Chunk {chunk_num}:**")
                            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.write("---")
        except Exception as e:
            st.error(f"❌ Error loading FAISS index: {str(e)}")
            st.info("Try reprocessing the URLs to recreate the index.")
    else:
        status_container.error("❌ FAISS index not found. Please process the URLs first.")
    
    # Debug Summary
    if enable_debug and user_query:
        st.divider()
        st.subheader("🔍 Debug Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Processing Stats:**")
            st.write(f"- Query length: {len(user_query)} characters")
            if os.path.isdir(faiss_directory):
                st.write(f"- Vector store: ✅ Loaded")
                st.write(f"- Storage path: `{os.path.abspath(faiss_directory)}`")
            else:
                st.write(f"- Vector store: ❌ Not found")
        
        with col2:
            st.write("**🎯 Retrieval Stats:**")
            if 'query_result' in locals():
                st.write(f"- Chunks retrieved: {len(query_result.get('source_documents', []))}")
                st.write(f"- Answer generated: ✅")
                if 'all_keywords' in locals():
                    st.write(f"- Keywords extracted: {len(all_keywords)}")
                if 'news_articles' in locals():
                    st.write(f"- News articles found: {len(news_articles)}")
            else:
                st.write("- No query processed yet") 
