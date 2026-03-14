import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const API_URL = 'http://localhost:9000';

export default function Home({ onAuthRequired }) {
    const { user } = useAuth();

    // Upload state
    const [isDragover, setIsDragover] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [uploadStatus, setUploadStatus] = useState(null); // null | 'uploading' | 'success' | 'error'
    const [uploadError, setUploadError] = useState('');
    const fileInputRef = useRef(null);

    // Pipeline state
    const [pipelineStatus, setPipelineStatus] = useState(null); // null | 'running' | 'success' | 'error'
    const [pipelineError, setPipelineError] = useState('');
    const [pipelineStats, setPipelineStats] = useState(null);

    // ── Feature card scroll animations ────────────────────────────────────────
    useEffect(() => {
        const observerOptions = { root: null, rootMargin: '0px', threshold: 0.15 };
        const featureObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate');
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);
        document.querySelectorAll('.feature-card').forEach(card => featureObserver.observe(card));
        return () => featureObserver.disconnect();
    }, []);

    // ── Drop / Click handlers ─────────────────────────────────────────────────
    const handleDropboxClick = (e) => {
        if (e.target.closest('.status-card')) return;
        fileInputRef.current.click();
    };

    const handleDragOver = (e) => { e.preventDefault(); setIsDragover(true); };
    const handleDragLeave = () => setIsDragover(false);

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragover(false);
        if (e.dataTransfer.files?.length > 0) sendFiles(e.dataTransfer.files);
    };

    const handleFileChange = (e) => {
        if (e.target.files?.length > 0) sendFiles(e.target.files);
    };

    // ── Upload to FastAPI ─────────────────────────────────────────────────────
    const sendFiles = async (files) => {
        setUploadStatus('uploading');
        setUploadError('');
        setPipelineStatus(null);
        setPipelineStats(null);

        const formData = new FormData();
        Array.from(files).forEach(f => formData.append('files', f));

        try {
            const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Upload failed');
            setUploadedFiles(data.files);
            setUploadStatus('success');
        } catch (err) {
            setUploadError(err.message);
            setUploadStatus('error');
        }
    };

    // ── Run Pipeline via FastAPI ──────────────────────────────────────────────
    const runPipeline = async () => {
        setPipelineStatus('running');
        setPipelineError('');
        setPipelineStats(null);

        try {
            const res = await fetch(`${API_URL}/run-pipeline`, { method: 'POST' });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail?.message || data.detail || 'Pipeline failed');

            const statsRes = await fetch(`${API_URL}/status`);
            const statsData = await statsRes.json();
            setPipelineStats(statsData.pipeline_stats);
            setPipelineStatus('success');
        } catch (err) {
            setPipelineError(err.message);
            setPipelineStatus('error');
        }
    };

    const resetAll = () => {
        setUploadStatus(null);
        setUploadedFiles([]);
        setPipelineStatus(null);
        setPipelineStats(null);
        setUploadError('');
        setPipelineError('');
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    return (
        <main>
            <div className="glow-wrapper">
                <div className="glow-1"></div>
                <div className="glow-2"></div>
            </div>

            <div className="hero-text">
                <h1>Data Intelligence, Simplified.</h1>
                <p>Upload your documents, run the pipeline, and explore deep insights — powered by AI.</p>
            </div>

            <div className="dropbox-wrapper">
                <div
                    className={`dropbox-container ${isDragover ? 'dragover' : ''} ${uploadStatus === 'success' ? 'uploaded' : ''}`}
                    onClick={handleDropboxClick}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                >
                    {uploadStatus === null && (
                        <>
                            <div className="dropbox-icon">📁</div>
                            <div className="dropbox-text">Drop your files here or click to browse</div>
                            <div className="dropbox-subtext">Supports PDF, CSV, Excel, TXT, JSON, DOCX</div>
                        </>
                    )}

                    {uploadStatus === 'uploading' && (
                        <div className="status-card uploading">
                            <div className="spinner"></div>
                            <span>Uploading files…</span>
                        </div>
                    )}

                    {uploadStatus === 'error' && (
                        <div className="status-card error">
                            <span className="status-icon">❌</span>
                            <span>{uploadError}</span>
                            <button className="retry-btn" onClick={(e) => { e.stopPropagation(); resetAll(); }}>Try Again</button>
                        </div>
                    )}

                    {uploadStatus === 'success' && (
                        <div className="status-card success" onClick={e => e.stopPropagation()}>
                            <span className="status-icon">✅</span>
                            <span className="status-title">{uploadedFiles.length} file{uploadedFiles.length > 1 ? 's' : ''} uploaded</span>

                            <ul className="file-list">
                                {uploadedFiles.map((f, i) => (
                                    <li key={i}><span className="file-name">{f.name}</span><span className="file-size">{f.size_kb} KB</span></li>
                                ))}
                            </ul>

                            <div className="pipeline-mini-panel">
                                {pipelineStatus === null && (
                                    <button className="run-pipeline-btn" onClick={runPipeline}>
                                        ⚡ Run Analysis & Insights
                                    </button>
                                )}
                                {pipelineStatus === 'running' && (
                                    <div className="pipeline-running">
                                        <div className="spinner"></div>
                                        <span>Analyzing Data...</span>
                                    </div>
                                )}
                                {pipelineStatus === 'error' && (
                                    <div className="pipeline-error">
                                        <span>❌ {pipelineError}</span>
                                        <button className="retry-btn" onClick={() => setPipelineStatus(null)}>Retry</button>
                                    </div>
                                )}
                                {pipelineStatus === 'success' && (
                                    <div className="discovery-cta">
                                        <div className="mini-stats">
                                            {pipelineStats ? (
                                                <span>{pipelineStats.chunks_after_filter} Chunks Analyzed</span>
                                            ) : (
                                                <span>Analysis Ready</span>
                                            )}
                                        </div>
                                        <Link className="open-dashboard-btn" to="/dashboard" style={{ width: '100%', textAlign: 'center' }}>
                                            💡 Go to Dashboard →
                                        </Link>
                                    </div>
                                )}
                            </div>
                            <button className="retry-btn-subtle" onClick={resetAll}>Upload Different Files</button>
                        </div>
                    )}

                    <input
                        type="file"
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        multiple
                        accept=".json,.csv,.txt,.pdf,.xlsx,.docx"
                        onChange={handleFileChange}
                    />
                </div>
            </div>

            <div className="features-section" id="features">
                <h2 className="features-title">Core Capabilities</h2>
                <div className="features-grid">
                    <div className="feature-card">
                        <h3 className="feature-title">Document Ingestion</h3>
                        <p className="feature-desc">Upload PDF, CSV, Excel, TXT, JSON and DOCX files. Our pipeline intelligently parses every format.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">Data Analysis</h3>
                        <p className="feature-desc">Perform statistical analysis — Chi-Square, T-Test, ANOVA — and generate rich visualizations.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">AI Assistant</h3>
                        <p className="feature-desc">Let the AI recommend which statistical tests to run and explain results in plain English.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">RAG Chat</h3>
                        <p className="feature-desc">Ask natural language questions about your uploaded documents. Our engine finds exact answers.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">Business Charts</h3>
                        <p className="feature-desc">Auto-generated KPI summaries, trend charts, and market share breakdowns from your data.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">PDF Export</h3>
                        <p className="feature-desc">Download a full AI-powered PDF report including executive summary and all analysis charts.</p>
                    </div>
                </div>
            </div>
        </main>
    );
}
