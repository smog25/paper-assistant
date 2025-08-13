# app_v2.py
"""
Streamlit frontend for Research Assistant
Calls FastAPI backend for all logic
"""

import streamlit as st
import requests
import pandas as pd
import hashlib
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment
load_dotenv()

# Configuration
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Check if OpenAI is available
HAS_OPENAI = bool(OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-key-here-or-leave-blank")

if HAS_OPENAI:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Page config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'summary_cache' not in st.session_state:
    st.session_state.summary_cache = {}

# Header
st.title("📚 AI Research Assistant")
st.markdown("Free citation verification, statistics extraction, and paper analysis")

# Check backend health
try:
    health_response = requests.get(f"{API_BASE}/healthz", timeout=2)
    if health_response.status_code != 200:
        st.error("⚠️ Backend is not responding. Please ensure the API is running on port 8000.")
        st.stop()
except requests.exceptions.RequestException:
    st.error("⚠️ Cannot connect to backend. Run: `uvicorn backend:app --reload --port 8000`")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📤 Upload Paper")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Max 10MB, 100 pages"
    )
    
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📁 Size: {file_size_mb:.2f} MB")
        
        # Version info
        with st.expander("ℹ️ System Info"):
            try:
                version_response = requests.get(f"{API_BASE}/version")
                if version_response.status_code == 200:
                    version_data = version_response.json()
                    st.json(version_data)
            except:
                st.write("Version info unavailable")

# Main content
if uploaded_file:
    # Parse PDF
    with st.spinner("Parsing PDF..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(
                f"{API_BASE}/api/parse_pdf",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                pdf_data = response.json()
                st.session_state.pdf_data = pdf_data
                
                # Check if OCR is needed
                if pdf_data['extraction_method'] == 'failed_needs_ocr':
                    st.warning("⚠️ This appears to be a scanned PDF. Text extraction failed.")
                    st.info("OCR is required to read this document.")
                    
                    if st.button("🔍 Run OCR (may take 30-60 seconds)", type="primary"):
                        with st.spinner("Running OCR... Please wait..."):
                            try:
                                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                                ocr_response = requests.post(
                                    f"{API_BASE}/api/parse_pdf_ocr",
                                    files=files,
                                    timeout=120
                                )
                                
                                if ocr_response.status_code == 200:
                                    pdf_data = ocr_response.json()
                                    st.session_state.pdf_data = pdf_data
                                    st.success("✅ OCR completed successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"OCR failed: {ocr_response.text}")
                            except requests.exceptions.Timeout:
                                st.error("OCR timed out. Try a smaller PDF.")
                            except Exception as e:
                                st.error(f"OCR error: {str(e)}")
                    st.stop()
                
                # Display stats
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("📄 Pages", pdf_data['pages'])
                col2.metric("📝 Words", f"{pdf_data['word_count']:,}")
                col3.metric("📚 Sections", len(pdf_data.get('sections', {})))
                col4.metric("🔤 Characters", f"{len(pdf_data['text']):,}")
                col5.metric("🔎 Method", pdf_data['extraction_method'].upper())
                
                # Section navigation
                if pdf_data.get('sections'):
                    st.markdown("### 📍 Jump to Section")
                    section_cols = st.columns(len(pdf_data['sections']))
                    for i, section_name in enumerate(pdf_data['sections'].keys()):
                        with section_cols[i % len(section_cols)]:
                            if st.button(section_name, key=f"nav_{section_name}"):
                                st.info(f"Section: {section_name}")
                
            else:
                st.error(f"Failed to parse PDF: {response.text}")
                st.stop()
                
        except requests.exceptions.Timeout:
            st.error("PDF parsing timed out. Try a smaller file.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()
    
    # Create tabs
    tabs = st.tabs([
        "🔍 Verify Citations",
        "📊 Statistics Check",
        "🎯 Truthiness Score",
        "📚 In-Text Citations",
        "💬 AI Summary"
    ])
    
    # Tab 1: Citation Verification
    with tabs[0]:
        st.markdown("### Citation Verification (CrossRef)")
        st.caption("Checks if references are real and findable in academic databases")
        
        if st.button("🔍 Verify All References", type="primary", key="verify_btn"):
            with st.spinner("Verifying citations (20-30 seconds)..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/api/verify_references",
                        json={"text": pdf_data['text']},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        citations = response.json()
                        
                        if citations:
                            # Store in session state
                            st.session_state.citations = citations
                            
                            # Summary metrics
                            verified = [c for c in citations if c['status'] == 'verified']
                            suspicious = [c for c in citations if c['status'] == 'suspicious']
                            not_found = [c for c in citations if c['status'] == 'not_found']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("✅ Verified", len(verified))
                            col2.metric("⚠️ Suspicious", len(suspicious))
                            col3.metric("❌ Not Found", len(not_found))
                            col4.metric("📚 Total", len(citations))
                            
                            # Filter options
                            st.markdown("### Filter Results")
                            filter_option = st.radio(
                                "Show:",
                                ["All", "Verified Only", "Suspicious Only", "Not Found Only"],
                                horizontal=True,
                                key="filter_radio"
                            )
                            
                            # Apply filter
                            if filter_option == "Verified Only":
                                filtered = verified
                            elif filter_option == "Suspicious Only":
                                filtered = suspicious
                            elif filter_option == "Not Found Only":
                                filtered = not_found
                            else:
                                filtered = citations
                            
                            # Display table
                            if filtered:
                                st.markdown("### Results Table")
                                
                                # Create DataFrame
                                table_data = []
                                for cite in filtered:
                                    status_icon = "✅" if cite['status'] == 'verified' else "⚠️" if cite['status'] == 'suspicious' else "❌"
                                    
                                    table_data.append({
                                        "Status": status_icon,
                                        "Reference": cite['raw_text'][:60] + "...",
                                        "Confidence": f"{cite['confidence']:.0%}",
                                        "DOI": cite.get('doi', ''),
                                        "Explanation": cite.get('explanation', '')[:50]
                                    })
                                
                                df = pd.DataFrame(table_data)
                                
                                # Search box
                                search_term = st.text_input("🔎 Search references:", key="search_refs")
                                if search_term:
                                    df = df[df['Reference'].str.contains(search_term, case=False, na=False)]
                                
                                st.dataframe(df, use_container_width=True, height=400)
                                
                                # Export BibTeX
                                if verified:
                                    st.markdown("### Export Verified Citations")
                                    
                                    # Generate BibTeX
                                    bibtex_entries = []
                                    for i, cite in enumerate(verified, 1):
                                        if cite.get('doi'):
                                            entry = f"@article{{ref{i},\n"
                                            entry += f"  doi = {{{cite['doi']}}},\n"
                                            
                                            if cite.get('title'):
                                                # Fix: Build title line properly
                                                title = cite['title'].replace('{', '').replace('}', '')
                                                entry += "  title = {{" + title + "}},\n"
                                            
                                            if cite.get('authors'):
                                                authors = ' and '.join(cite['authors'][:5])
                                                entry += f"  author = {{{authors}}},\n"
                                            
                                            entry += "}"
                                            bibtex_entries.append(entry)
                                    
                                    bibtex_content = '\n\n'.join(bibtex_entries)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.download_button(
                                            label="📥 Download BibTeX",
                                            data=bibtex_content,
                                            file_name="verified_citations.bib",
                                            mime="text/plain",
                                            key="download_bibtex"
                                        )
                                    with col2:
                                        if st.button("📋 Copy DOIs", key="copy_dois"):
                                            dois = [c['doi'] for c in verified if c.get('doi')]
                                            doi_list = '\n'.join(dois)
                                            st.code(doi_list, language=None)
                        else:
                            st.warning("No references found in this paper")
                    else:
                        st.error(f"Verification failed: {response.text}")
                        
                except requests.exceptions.Timeout:
                    st.error("Verification timed out. Try again.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 2: Statistics Extraction
    with tabs[1]:
        st.markdown("### Statistical Analysis")
        st.caption("Extracts p-values, sample sizes, and identifies potential issues")
        
        if st.button("📊 Extract Statistics", type="primary", key="stats_btn"):
            with st.spinner("Analyzing statistics..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/api/extract_statistics",
                        json={"text": pdf_data['text']},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        stats = response.json()
                        
                        # Summary
                        st.info(stats['summary'])
                        
                        # Display in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### P-Values")
                            if stats['p_values']:
                                for p in stats['p_values'][:15]:
                                    if any(x in p for x in ['0.04', '0.05', '0.06']):
                                        st.warning(f"⚠️ {p}")
                                    else:
                                        st.write(f"• {p}")
                            else:
                                st.write("None found")
                            
                            st.markdown("#### Effect Sizes")
                            if stats['effect_sizes']:
                                for e in stats['effect_sizes'][:10]:
                                    st.write(f"• {e}")
                            else:
                                st.write("None found")
                        
                        with col2:
                            st.markdown("#### Sample Sizes")
                            if stats['sample_sizes']:
                                for n in stats['sample_sizes'][:15]:
                                    size = int(''.join(filter(str.isdigit, n)))
                                    if size < 30:
                                        st.warning(f"⚠️ {n} (small)")
                                    else:
                                        st.write(f"• {n}")
                            else:
                                st.write("None found")
                        
                        # Red flags
                        st.markdown("#### Analysis Flags")
                        for flag in stats['red_flags']:
                            if "✅" in flag:
                                st.success(flag)
                            elif "🚩" in flag or "❌" in flag:
                                st.error(flag)
                            else:
                                st.warning(flag)
                    else:
                        st.error("Statistics extraction failed")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 3: Truthiness Score
    with tabs[2]:
        st.markdown("### Paper Trustworthiness Analysis")
        st.warning("⚠️ **Disclaimer**: This is a heuristic analysis, not peer review. Use as a preliminary signal only.")
        
        # Field selection
        field = st.selectbox(
            "Select field for context-aware scoring:",
            ["general", "psychology", "clinical", "cs", "biology"],
            key="field_select"
        )
        
        if st.button("🎯 Calculate Truthiness Score", type="primary", key="truth_btn"):
            with st.spinner("Calculating..."):
                try:
                    # Only send field param if not "general"
                    params = {"field": field} if field != "general" else {}
                    
                    response = requests.post(
                        f"{API_BASE}/api/truthiness_score",
                        json={"text": pdf_data['text']},
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display score with color
                        score = result['score']
                        grade = result['grade']
                        
                        if score >= 80:
                            st.success(f"# Score: {score}/100 (Grade: {grade})")
                            st.balloons()
                        elif score >= 60:
                            st.warning(f"# Score: {score}/100 (Grade: {grade})")
                        else:
                            st.error(f"# Score: {score}/100 (Grade: {grade})")
                        
                        # Progress bar
                        st.progress(score / 100)
                        
                        # Disclaimer
                        st.caption(result['disclaimer'])
                        
                        # Factors
                        st.markdown("### Contributing Factors")
                        for reason in result['reasons']:
                            if "✅" in reason:
                                st.success(reason)
                            else:
                                st.warning(f"• {reason}")
                    else:
                        st.error("Score calculation failed")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 4: In-Text Citations
    with tabs[3]:
        st.markdown("### In-Text Citation Analysis")
        st.caption("Finds all citation markers within the paper text")
        
        if st.button("📚 Find In-Text Citations", type="primary", key="intext_btn"):
            with st.spinner("Extracting citations..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/api/find_intext_citations",
                        json={"text": pdf_data['text']},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        cites = response.json()
                        
                        # Summary
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Citations", cites['total_count'])
                        col2.metric("Numeric [1]", len(cites['numeric']))
                        col3.metric("Author (Year)", len(cites['author_year']))
                        
                        st.info(cites['explanation'])
                        
                        # Display citations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Numeric Citations")
                            if cites['numeric']:
                                for c in cites['numeric'][:50]:
                                    st.write(f"• [{c}]")
                            else:
                                st.write("None found")
                        
                        with col2:
                            st.markdown("#### Author-Year Citations")
                            if cites['author_year']:
                                for c in cites['author_year'][:50]:
                                    st.write(f"• ({c})")
                            else:
                                st.write("None found")
                    else:
                        st.error("Citation extraction failed")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 5: AI Summary (Optional)
    with tabs[4]:
        st.markdown("### AI-Powered Summary")
        
        if not HAS_OPENAI:
            st.info("ℹ️ OpenAI API key not configured. Add OPENAI_API_KEY to .env to enable AI summaries.")
            st.caption("Note: All other features work without OpenAI!")
            
            # Suggest Ollama as alternative
            with st.expander("💡 Want free local AI?"):
                st.markdown("""
                **Use Ollama for free, unlimited AI:**
                1. Install: `brew install ollama` (Mac) or see ollama.ai
                2. Pull model: `ollama pull llama2`
                3. We'll add Ollama support in the next version!
                """)
        else:
            st.caption("This feature uses OpenAI API (costs ~$0.002 per summary)")
            
            # Text selection
            text_length = st.select_slider(
                "How much text to analyze?",
                options=["First 1000 chars", "First 3000 chars", "First 5000 chars"],
                value="First 3000 chars",
                key="text_slider"
            )
            
            if st.button("📝 Generate Summary", type="primary", key="summary_btn"):
                # Extract text amount
                char_limit = int(text_length.split()[1])
                text_sample = pdf_data['text'][:char_limit]
                
                # Create cache key
                cache_key = hashlib.md5(text_sample.encode()).hexdigest()
                
                # Check cache
                if cache_key in st.session_state.summary_cache:
                    st.success("✅ Using cached summary")
                    st.markdown(st.session_state.summary_cache[cache_key])
                else:
                    with st.spinner("Generating summary with AI..."):
                        try:
                            prompt = f"""
                            Analyze this research paper and provide:
                            
                            1. **Main Topic** (2 sentences)
                            2. **Key Findings** (3 bullet points)
                            3. **Methods** (2 sentences)
                            4. **Significance** (2 sentences)
                            
                            Paper text:
                            {text_sample}
                            """
                            
                            response = openai_client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a research paper analysis expert."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.3,
                                max_tokens=500
                            )
                            
                            summary = response.choices[0].message.content
                            
                            # Cache it
                            st.session_state.summary_cache[cache_key] = summary
                            
                            # Display
                            st.markdown(summary)
                            
                            # Show cost
                            st.caption(f"Estimated cost: ~$0.002 | Cached for this session")
                            
                        except Exception as e:
                            st.error(f"AI generation failed: {str(e)}")

else:
    # No file uploaded
    st.info("👆 Upload a PDF in the sidebar to get started")
    
    # Features overview
    with st.expander("✨ Features Overview"):
        st.markdown("""
        ### Free Features (No AI Required)
        - 🔍 **Citation Verification**: Check if references exist in CrossRef
        - 📊 **Statistics Extraction**: Find p-values, sample sizes, effect sizes
        - 🎯 **Truthiness Score**: Heuristic paper quality assessment
        - 📚 **In-Text Citations**: Extract all citation markers
        - 📍 **Section Navigation**: Jump to Abstract, Methods, Results
        - 🔎 **OCR Support**: Process scanned PDFs
        
        ### Optional AI Features
        - 💬 **Smart Summary**: GPT-powered analysis (requires OpenAI key)
        
        ### Coming Soon
        - 🔗 Related papers via OpenAlex
        - 📈 Citation network visualization
        - 🗂️ Paper library with search
        """)
    
    # Quick test
    with st.expander("🧪 Quick Test"):
        st.markdown("""
        **Test the app with these papers:**
        1. [Download a CS paper](https://arxiv.org/pdf/1706.03762.pdf) (Attention Is All You Need)
        2. [Download a Biology paper](https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf)
        3. Any PDF from Google Scholar
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Works best with text-based PDFs
- OCR for scanned documents
- All features except AI summary are free
- Backend must be running on port 8000
""")

# Cost tracker
if 'pdf_data' in st.session_state:
    st.sidebar.markdown("### 💰 Cost Tracker")
    ai_calls = len(st.session_state.get('summary_cache', {}))
    estimated_cost = ai_calls * 0.002
    st.sidebar.metric("AI Summaries", ai_calls)
    st.sidebar.metric("Estimated Cost", f"${estimated_cost:.3f}")
    st.sidebar.caption("Citation verification is FREE!")