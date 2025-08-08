# UPDATED app.py for OpenAI v1.0+

import streamlit as st
from openai import OpenAI  # New import style
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure OpenAI with new client style
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "sk-your-actual-key-here":
    st.error("⚠️ OpenAI API key not found or not set properly!")
    st.info("Make sure you've added your real API key to the .env file")
    st.stop()

# Initialize the OpenAI client (NEW WAY)
client = OpenAI(api_key=api_key)

# Test the API key is valid
try:
    # New way to test connection
    client.models.list()
    st.sidebar.success("✅ OpenAI connected")
except Exception as e:
    st.sidebar.error(f"❌ OpenAI error: {str(e)}")
    st.stop()

# Configure the page
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Assistant")
st.markdown("Upload any research paper and I'll help you understand it in seconds.")

# Sidebar
with st.sidebar:
    st.header("📤 Upload Paper")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=['pdf'],
        help="Max file size: 200MB"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Loaded: {uploaded_file.name}")
        file_size = uploaded_file.size / 1024
        if file_size > 1024:
            st.info(f"📁 Size: {file_size/1024:.1f} MB")
        else:
            st.info(f"📁 Size: {file_size:.1f} KB")

# Main content
if uploaded_file is not None:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        
        # Extract text from all pages
        full_text = ""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num, page in enumerate(pdf_reader.pages):
            status_text.text(f"Reading page {page_num + 1} of {len(pdf_reader.pages)}...")
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            progress_bar.progress((page_num + 1) / len(pdf_reader.pages))
        
        progress_bar.empty()
        status_text.empty()
        
        if not full_text:
            st.error("❌ Could not extract any text from PDF")
            st.info("This might be a scanned PDF. Try a different file.")
            st.stop()
        
        # Store in session state
        st.session_state.paper_text = full_text
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Pages", len(pdf_reader.pages))
        with col2:
            st.metric("📝 Words", f"{len(full_text.split()):,}")
        with col3:
            st.metric("🔤 Characters", f"{len(full_text):,}")
        
        # Show preview
        with st.expander("👀 Preview extracted text (first 500 chars)"):
            st.text(full_text[:500])
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔍 Key Findings", "📚 Citations", "💬 Chat"])
        
        # Tab 1: Summary
        with tab1:
            st.markdown("### Generate AI Summary")
            
            text_amount = st.select_slider(
                "How much to analyze?",
                options=["First 1000 chars", "First 3000 chars", "First 5000 chars"],
                value="First 3000 chars"
            )
            
            if st.button("🤖 Analyze Paper", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        # Get text amount
                        char_limit = int(text_amount.split()[1])
                        text_sample = full_text[:char_limit]
                        
                        # Create prompt
                        prompt = f"""
                        Analyze this research paper and provide:
                        
                        1. **Main Topic**: What is this paper about? (2 sentences)
                        2. **Key Findings**: List the 3 most important findings
                        3. **Methods**: How was the research conducted? (2 sentences)
                        4. **Significance**: Why does this matter? (2 sentences)
                        
                        Paper text:
                        {text_sample}
                        """
                        
                        # Call OpenAI (NEW WAY)
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful research assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_tokens=500
                        )
                        
                        # Get the response (NEW WAY)
                        summary = response.choices[0].message.content
                        
                        # Display
                        st.markdown("### 📊 Analysis Results")
                        st.markdown(summary)
                        
                        # Show token usage
                        if response.usage:
                            st.caption(f"Tokens used: {response.usage.total_tokens}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        if "api_key" in str(e).lower():
                            st.info("Check your API key in the .env file")
        
        # Tab 2: Key Findings (Simple version)
        with tab2:
            st.markdown("### Extract Key Findings")
            
            if st.button("🔎 Find Key Sentences", type="primary"):
                # Simple keyword-based extraction
                sentences = full_text.split('.')
                keywords = ['found', 'discovered', 'showed', 'demonstrated', 
                          'concluded', 'revealed', 'suggest', 'indicate']
                
                key_sentences = []
                for sentence in sentences[:100]:  # Check first 100 sentences
                    if any(keyword in sentence.lower() for keyword in keywords):
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 20:  # Skip very short sentences
                            key_sentences.append(clean_sentence + '.')
                
                if key_sentences:
                    st.markdown("### 🎯 Potential Key Findings")
                    for i, finding in enumerate(key_sentences[:10], 1):
                        st.write(f"{i}. {finding}")
                else:
                    st.info("No explicit findings found. Try the AI Summary instead.")
        
        # Tab 3: Citations
        with tab3:
            st.markdown("### Extract Citations")
            
            if st.button("📖 Find Citations", type="primary"):
                # Look for common citation patterns
                patterns = [
                    r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s+\d{4}\)',  # (Author, 2024)
                    r'\[[0-9]+\]',  # [1], [2], etc.
                    r'\[[0-9]+-[0-9]+\]',  # [1-5]
                ]
                
                all_citations = []
                for pattern in patterns:
                    citations = re.findall(pattern, full_text)
                    all_citations.extend(citations)
                
                # Remove duplicates
                unique_citations = list(set(all_citations))
                unique_citations.sort()
                
                if unique_citations:
                    st.success(f"Found {len(unique_citations)} unique citations")
                    
                    # Display in columns
                    cols = st.columns(3)
                    for i, citation in enumerate(unique_citations[:60]):
                        with cols[i % 3]:
                            st.write(f"• {citation}")
                else:
                    st.warning("No citations found. The paper might use a different format.")
        
        # Tab 4: Chat
        with tab4:
            st.markdown("### 💬 Ask Questions About This Paper")
            
            user_question = st.text_input(
                "Your question:",
                placeholder="What methods did they use?"
            )
            
            if st.button("Get Answer", type="primary") and user_question:
                with st.spinner("Thinking..."):
                    try:
                        context = full_text[:3000]
                        
                        # NEW WAY to call OpenAI
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "Answer based on the paper excerpt provided."},
                                {"role": "user", "content": f"Question: {user_question}\n\nPaper excerpt: {context}"}
                            ],
                            temperature=0.5,
                            max_tokens=300
                        )
                        
                        answer = response.choices[0].message.content
                        st.markdown("### Answer:")
                        st.write(answer)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        st.info("Try a different PDF file")

else:
    st.info("👆 Upload a PDF in the sidebar to get started")
    
    with st.expander("ℹ️ Quick Start Guide"):
        st.markdown("""
        **How to use:**
        1. Upload any research paper (PDF)
        2. Click 'Analyze Paper' for AI summary
        3. Explore other tabs for citations and chat
        
        **Tips:**
        - Works best with text-based PDFs (not scanned)
        - Each analysis costs ~$0.002 in API credits
        - Analyzing less text saves money
        """)