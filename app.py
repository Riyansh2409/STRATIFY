import streamlit as st
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import subprocess
import sys
import shutil
import os

# Ensure fpdf2 is installed in whatever environment is running Streamlit
try:
    import fpdf
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    import fpdf

from analysis.pdf_exporter import generate_pdf_report

st.set_page_config(page_title="Stratify Preprocessing Dashboard", layout="wide")

st.title("🚀 Stratify Preprocessing Pipeline Dashboard")
st.markdown("This dashboard lets you upload files (PDF, CSV, Excel, TXT, JSON), process them, and visualise the outputs.")

DATA_DIR = Path(__file__).parent / "processed_data"
REPORT_DIR = Path(__file__).parent / "reports_output"
UPLOAD_DIR = Path(__file__).parent / "uploads"

# ── File Upload Section ────────────────────────────────────────────────────────
st.header("📂 1. Upload Files")
st.markdown("Upload your raw documents here. Supported formats: `.txt`, `.pdf`, `.csv`, `.xlsx`, `.json`, `.docx`")

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)

if uploaded_files:
    # Clear existing uploads if new files are uploaded
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()
            
    # Save new files
    for uploaded_file in uploaded_files:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"✅ Successfully saved {len(uploaded_files)} files.")

# ── Run Pipeline Section ───────────────────────────────────────────────────────
st.header("⚙️ 2. Run Pipeline")
st.markdown("Click the button below to process the documents currently in your `uploads/` folder.")

if st.button("▶️ Start Preprocessing", type="primary"):
    has_files = any(UPLOAD_DIR.iterdir()) if UPLOAD_DIR.exists() else False
    
    if not has_files:
        st.error("❌ Please upload some files first.")
    else:
        with st.spinner("Processing documents... This might take a moment."):
            try:
                # Run the pipeline script
                result = subprocess.run(
                    [
                        sys.executable, "run_pipeline.py", 
                        "--input", "uploads", 
                        "--output", "processed_data", 
                        "--report", "reports_output",
                        "--chunk-size", "60", 
                        "--overlap", "10"
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent)
                )
                
                if result.returncode == 0:
                    st.success("✅ Pipeline executed successfully!")
                    
                    # Store chunks in Vector DB for RAG
                    with st.spinner("Building Knowledge Base for RAG..."):
                        try:
                            # Use subprocess to call their robust indexing script
                            data_path = DATA_DIR / "val.jsonl"
                            index_path = Path(__file__).parent / "rag_index"
                            subprocess.run(
                                [
                                    sys.executable, "-m", "rag.rag_pipeline", "index", 
                                    "--data", str(data_path), 
                                    "--index", str(index_path), 
                                    "--backend", "chroma",
                                    "--no-reranker"
                                ],
                                check=True,
                                cwd=str(Path(__file__).parent)
                            )
                            st.success("✅ Knowledge base built! You can now chat with your documents.")
                        except Exception as e:
                            st.error(f"Failed to build vector store: {e}")
                else:
                    st.error("❌ Pipeline execution failed.")
                    with st.expander("Show Logic Error Details"):
                        st.text(result.stderr)
            except Exception as e:
                st.error(f"Error running pipeline: {e}")

st.markdown("---")

# ── Output Dashboard ───────────────────────────────────────────────────────────
st.header("📊 3. Output Dashboard")

@st.cache_data(ttl=5) # Cache for 5 seconds to load newly generated files
def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

pipeline_stats = load_json(DATA_DIR / "pipeline_stats.json")
analysis_report = load_json(REPORT_DIR / "analysis_report.json")

if not pipeline_stats or not analysis_report:
    st.info("ℹ️ Output files not found. Please upload files and run the pipeline to see the dashboard.")
    st.stop()

# ── Pipeline Stats ──
st.subheader("Pipeline Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Files", pipeline_stats.get("total_files", 0))
col2.metric("Raw Characters", f"{pipeline_stats.get('total_raw_chars', 0):,}")
col3.metric("Chunks Before Filter", pipeline_stats.get("chunks_before_filter", 0))
col4.metric("Chunks After Filter", pipeline_stats.get("chunks_after_filter", 0))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filter Rate", f"{pipeline_stats.get('filter_rate_pct', 0)}%")
col2.metric("Dropped (Too Short)", pipeline_stats.get("dropped_too_short", 0))
col3.metric("Dropped (Low Quality)", pipeline_stats.get("dropped_low_quality", 0))
col4.metric("Dropped (Duplicates)", pipeline_stats.get("dropped_duplicates", 0))

# ── Dataset Analysis ──
st.subheader("Dataset Analysis (Tests & Charts)")

test_results = analysis_report.get("test_results", [])
if test_results:
    st.markdown("**Statistical Tests**")
    df_tests = pd.DataFrame(test_results)
    st.dataframe(df_tests, use_container_width=True)

chart_paths = analysis_report.get("chart_paths", {})
if chart_paths:
    st.markdown("**Data Distributions**")
    cols = st.columns(2)
    col_idx = 0
    for fig_id, path_str in chart_paths.items():
        img_path = Path(path_str)
        if img_path.exists():
            with cols[col_idx % 2]:
                image = Image.open(img_path)
                st.image(image, caption=fig_id, use_container_width=True)
            col_idx += 1

    st.markdown("---")
    
    # ── AI Statistical Assistant ──
    st.subheader("🧠 AI Statistical Assistant")
    st.markdown("Let the AI analyze your dataset properties and recommend the best statistical tests.")
    
    # Attempt to load RAG components globally for UI use
    rag_available = False
    ret = None
    gen = None
    try:
         from rag.rag_pipeline import load_retriever
         from rag.generator import RAGGenerator, GeneratorConfig
         from collections import namedtuple
         
         index_path = Path(__file__).parent / "rag_index"
         if index_path.exists():
              Args = namedtuple('Args', ['index', 'backend', 'no_reranker', 'no_mmr', 'top_k'])
              args = Args(index=str(index_path), backend="chroma", no_reranker=True, no_mmr=True, top_k=3)
              ret = load_retriever(args)
              gen = RAGGenerator(GeneratorConfig(max_new_tokens=256))
              rag_available = True
    except Exception as e:
         st.warning(f"AI Assistant initialization failed: {e}")

    if rag_available:
         with st.expander("Show AI Recommendations", expanded=True):
              if "ai_rec" not in st.session_state:
                   with st.spinner("AI is analyzing dataset properties..."):
                        rec_prompt = f"Analyze these dataset statistics: {json.dumps(pipeline_stats.get('language_distribution', {}))} and total chunks: {pipeline_stats.get('chunks_after_filter', 0)}. Recommend which of these tests should be run: Chi-Square, T-Test, or ANOVA. Explain briefly why. Limit to 3 sentences."
                        # Send directly to generator since we have the data
                        from rag.generator import RAGGenerator, GeneratorConfig
                        from rag.retriever import RAGResponse
                        dummy_resp = RAGResponse(query=rec_prompt, context=rec_prompt, passages=[], sources=[])
                        ans = gen.generate(dummy_resp)
                        st.session_state.ai_rec = ans.answer
              
              st.info(st.session_state.ai_rec)
              
         st.markdown("**Select Tests to run and include in the Full PDF Report:**")
         col1, col2, col3 = st.columns(3)
         with col1:
              run_chi = st.checkbox("Chi-Square (Distribution check)", value=True)
         with col2:
              run_ttest = st.checkbox("T-Test (Compare 2 models)", value=False)
         with col3:
              run_anova = st.checkbox("ANOVA (Compare all variants)", value=False)
    else:
         run_chi, run_ttest, run_anova = False, False, False

    st.markdown("---")
    st.subheader("Export Analysis Report")
    
    with st.spinner("Generating PDF (Including AI Executive Summary)..."):
        try:
            # Generate RAG Summary if pipeline is loaded
            ai_summary = None
            ai_stat_summary = None
            try:
                 from rag.rag_pipeline import load_retriever
                 from rag.generator import RAGGenerator, GeneratorConfig
                 from collections import namedtuple
                 
                 index_path = Path(__file__).parent / "rag_index"
                 if index_path.exists():
                      Args = namedtuple('Args', ['index', 'backend', 'no_reranker', 'no_mmr', 'top_k'])
                      args = Args(index=str(index_path), backend="chroma", no_reranker=True, no_mmr=True, top_k=3)
                      ret = load_retriever(args)
                      gen = RAGGenerator(GeneratorConfig(max_new_tokens=256))
                      
                      rag_req = ret.retrieve("Provide a comprehensive, high-level executive summary of exactly what dataset topics and domain focus areas are covered inside these documents. Limit your response to 2 detailed paragraphs.")
                      ans = gen.generate(rag_req)
                      ai_summary = ans.answer
                      
                      # Generate AI Analysis for selected tests
                      selected_tests = []
                      for t in test_results:
                           name = t.get("test", "")
                           if "Chi-Square" in name and run_chi: selected_tests.append(t)
                           elif "t-test" in name and run_ttest: selected_tests.append(t)
                           elif "ANOVA" in name and run_anova: selected_tests.append(t)
                      
                      if selected_tests:
                           test_summary_prompt = f"Analyze these statistical test results and explain them in plain English. Are they significant? What does it mean for the dataset? Results: {json.dumps(selected_tests)}. Limit to 2 paragraphs."
                           from rag.generator import RAGGenerator, GeneratorConfig
                           from rag.retriever import RAGResponse
                           dummy_resp2 = RAGResponse(query=test_summary_prompt, context=test_summary_prompt, passages=[], sources=[])
                           ai_stat_ans = gen.generate(dummy_resp2)
                           ai_stat_summary = ai_stat_ans.answer
                           
            except Exception as e:
                 print(f"Skipping AI Summary in PDF due to: {e}")

            pdf_bytes = generate_pdf_report(pipeline_stats, analysis_report, rag_summary_text=ai_summary, ai_stat_text=ai_stat_summary)
            st.download_button(
                label="📄 Download Full PDF Report",
                data=pdf_bytes,
                file_name="Stratify_Preprocessing_Report.pdf",
                mime="application/pdf",
                type="primary"
            )
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")

# ── Data Preview ──
st.subheader("Data Preview (val.jsonl)")

val_path = DATA_DIR / "val.jsonl"
if val_path.exists():
    preview_data = []
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10: # Preview top 10 rows
                break
            preview_data.append(json.loads(line))
            
    if preview_data:
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
    else:
        st.info("Validation dataset is empty.")
else:
    st.info("Validation dataset not found.")

st.markdown("---")

# ── RAG Chat Interface ─────────────────────────────────────────────────────────
st.header("💬 4. Ask Your Documents (RAG Chat)")
st.markdown("Ask natural language questions about the documents you uploaded. The system will retrieve the exact paragraphs to answer your query.")

try:
    from rag.rag_pipeline import load_retriever
    from rag.generator import RAGGenerator, GeneratorConfig
    from collections import namedtuple

    index_path = Path(__file__).parent / "rag_index"
    meta_path = index_path / "rag_meta.json"
    
    if not meta_path.exists():
        st.warning("⚠️ RAG knowledge base not found. Please upload files and click 'Start Preprocessing' to build the index.")
    else:
        # Load the RAG model pipeline each time for dev, so updates take immediate effect
        def load_rag_pipeline():
            Args = namedtuple('Args', ['index', 'backend', 'no_reranker', 'no_mmr', 'top_k'])
            # Set up arguments that match what the user's CLI expects
            args = Args(
                index=str(index_path), 
                backend="chroma", 
                no_reranker=True, 
                no_mmr=True, 
                top_k=3
            )
            
            retriever = load_retriever(args)
            generator = RAGGenerator(GeneratorConfig(max_new_tokens=256))
            return retriever, generator

        retriever, generator = load_rag_pipeline()
    
        # --- Advanced Search Filters ---
        st.subheader("Advanced Search Filters")
        
        # Build filter options from pipeline stats if available
        lang_options = ["All"]
        domain_options = ["All"]
        if pipeline_stats:
            lang_dist = pipeline_stats.get("language_distribution", {})
            if lang_dist:
                 lang_options.extend(list(lang_dist.keys()))
            domain_dist = pipeline_stats.get("domain_distribution", {})
            if domain_dist:
                 domain_options.extend(list(domain_dist.keys()))

        col1, col2 = st.columns(2)
        with col1:
             selected_lang = st.selectbox("Language Filter:", lang_options)
        with col2:
             selected_domain = st.selectbox("Domain Filter:", domain_options)
        st.markdown("---")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("E.g., What is machine learning?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Searching documents..."):
                # Compile filters
                filters = {}
                if selected_lang != "All":
                    filters["language"] = selected_lang
                if selected_domain != "All":
                    filters["domain"] = selected_domain

                rag_resp = retriever.retrieve(prompt, filters=filters if filters else None)
                answer = generator.generate(rag_resp)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer.answer)
                    if answer.sources:
                        with st.expander("View Source Metadata"):
                            st.json(answer.sources)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer.answer})

except Exception as e:
    st.warning(f"RAG system is currently unavailable or initializing. Error: {e}")

st.markdown("---")
st.markdown("Generated by Stratify Preprocessing Pipeline")
