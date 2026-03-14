import React, { useState, useEffect, useRef } from 'react';

const API_URL = 'http://localhost:9000';

export default function Dashboard() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [aiRec, setAiRec] = useState('');
    const [chatMessages, setChatMessages] = useState([
        { role: 'assistant', content: 'Hello! I have indexed your data. You can ask me anything about the uploaded documents.' }
    ]);
    const [chatInput, setChatInput] = useState('');
    const [isChatLoading, setIsChatLoading] = useState(false);
    const chatEndRef = useRef(null);

    useEffect(() => {
        fetchStatus();
    }, []);

    const fetchStatus = async () => {
        try {
            const res = await fetch(`${API_URL}/status`);
            const json = await res.json();
            if (json.status === 'ready') {
                setData(json);
                fetchAiRecommend();
            }
        } catch (e) {
            console.error("Failed to fetch dashboard data", e);
        } finally {
            setLoading(false);
        }
    };

    const fetchAiRecommend = async () => {
        try {
            const res = await fetch(`${API_URL}/ai/recommend`, { method: 'POST' });
            const json = await res.json();
            setAiRec(json.recommendation);
        } catch (e) {
            console.error("AI Rec failed", e);
        }
    };

    const handleChat = async (e) => {
        e.preventDefault();
        if (!chatInput.trim()) return;

        const newMsg = { role: 'user', content: chatInput };
        setChatMessages(prev => [...prev, newMsg]);
        setChatInput('');
        setIsChatLoading(true);

        try {
            const res = await fetch(`${API_URL}/ai/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: chatInput })
            });
            const json = await res.json();
            setChatMessages(prev => [...prev, { role: 'assistant', content: json.answer }]);
        } catch (e) {
            setChatMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not reach AI assistant." }]);
        } finally {
            setIsChatLoading(false);
        }
    };

    const handleExport = async () => {
        try {
            // Since backend expects POST with body for /export-pdf, we use a form approach or fetch
            const res = await fetch(`${API_URL}/export-pdf`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ use_chi: true, use_ttest: true, use_anova: true })
            });
            if (!res.ok) throw new Error("PDF Gen failed");
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "Stratify_Intelligence_Report.pdf";
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (e) {
            alert("Export failed: " + e.message);
        }
    };

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatMessages]);

    if (loading) return (
        <div className="premium-loader">
            <div className="pulse-circle"></div>
            <p>Gathering Intelligence...</p>
        </div>
    );

    if (!data) return (
        <div className="empty-state-card">
            <h2>No Intelligence Available</h2>
            <p>Please upload your documents and run the processing pipeline first.</p>
            <a href="/" className="premium-btn primary">Initialize Pipeline</a>
        </div>
    );

    const { stats, charts } = data;

    return (
        <div className="premium-dashboard">
            {/* ── Top Bar ── */}
            <header className="db-top-bar">
                <div className="title-group">
                    <h1>Intelligence Center</h1>
                    <p>Real-time dataset analysis & document synthesis</p>
                </div>
                <div className="action-group">
                    <button className="premium-btn secondary" onClick={() => window.location.reload()}>
                        🔄 Refresh
                    </button>
                    <button className="premium-btn primary download-btn" onClick={handleExport}>
                        📥 Download PDF Report
                    </button>
                </div>
            </header>

            {/* ── KPI Grid ── */}
            <div className="kpi-grid">
                <div className="kpi-card">
                    <span className="kpi-label">Corpus Size</span>
                    <span className="kpi-value">{stats.total_files} Files</span>
                </div>
                <div className="kpi-card">
                    <span className="kpi-label">Text Volume</span>
                    <span className="kpi-value">{(stats.total_raw_chars || 0).toLocaleString()} <small>chars</small></span>
                </div>
                <div className="kpi-card">
                    <span className="kpi-label">Neural Chunks</span>
                    <span className="kpi-value">{stats.chunks_after_filter}</span>
                </div>
                <div className="kpi-card highlight">
                    <span className="kpi-label">Quality Index</span>
                    <span className="kpi-value">{stats.filter_rate_pct}%</span>
                </div>
            </div>

            <div className="dashboard-layout">
                {/* ── Main Content Area ── */}
                <div className="layout-left">
                    {/* ── PROMINENT CHAT SECTION AT TOP ── */}
                    <section className="neural-chat-section">
                        <div className="section-header">
                            <h2>💬 Neural Search & Ask</h2>
                            <span>RAG-powered conversational engine</span>
                        </div>
                        <div className="chat-interface">
                            <div className="chat-messages-container">
                                {chatMessages.map((m, i) => (
                                    <div key={i} className={`msg-bubble ${m.role}`}>
                                        <div className="avatar">{m.role === 'assistant' ? '🤖' : '👤'}</div>
                                        <div className="text-content">{m.content}</div>
                                    </div>
                                ))}
                                {isChatLoading && (
                                    <div className="msg-bubble assistant typing">
                                        <div className="avatar pulse">🤖</div>
                                        <div className="text-content">Synthesizing answer...</div>
                                    </div>
                                )}
                                <div ref={chatEndRef} />
                            </div>
                            <form className="chat-input-area" onSubmit={handleChat}>
                                <input
                                    type="text"
                                    placeholder="Ask anything about your documents..."
                                    value={chatInput}
                                    onChange={e => setChatInput(e.target.value)}
                                    disabled={isChatLoading}
                                />
                                <button type="submit" disabled={isChatLoading || !chatInput.trim()}>
                                    {isChatLoading ? '...' : 'Send'}
                                </button>
                            </form>
                        </div>
                    </section>

                    {/* ── CHARTS SECTION ── */}
                    <section className="visual-insights-section">
                        <div className="section-header">
                            <h2>📊 Data Visualizations</h2>
                            <p>Automated business metrics & distribution plots</p>
                        </div>
                        <div className="charts-masonry">
                            {Object.entries(charts || {}).map(([label, filename]) => (
                                <div key={filename} className="insight-card">
                                    <div className="card-media">
                                        <img src={`${API_URL}/charts/${filename}`} alt={label} />
                                    </div>
                                    <div className="card-info">
                                        <h3>{label}</h3>
                                        <button className="expand-overlay-btn" onClick={() => window.open(`${API_URL}/charts/${filename}`, '_blank')}>View Full</button>
                                    </div>
                                </div>
                            ))}
                            {(!charts || Object.keys(charts).length === 0) && (
                                <div className="no-visuals">
                                    <p>No tabular data found for business charting.</p>
                                </div>
                            )}
                        </div>
                    </section>
                </div>

                {/* ── Right Sidebar ── */}
                <aside className="layout-right">
                    <section className="ai-advisor-card">
                        <div className="header">
                            <span className="tag">AI ADVISOR</span>
                            <h3>Strategic Recommendations</h3>
                        </div>
                        <div className="advisor-content">
                            <p>{aiRec || "Identifying patterns in your dataset..."}</p>
                        </div>
                        <div className="advisor-footer">
                            <small>Based on statistical distribution analysis</small>
                        </div>
                    </section>

                    <section className="dataset-health">
                        <h3>Language Dist.</h3>
                        <div className="health-bar-container">
                            {Object.entries(stats.language_distribution || {}).map(([lang, count]) => (
                                <div key={lang} className="health-row">
                                    <span className="label">{lang.toUpperCase()}</span>
                                    <div className="bar-bg">
                                        <div className="bar-fill" style={{ width: `${(count / stats.total_files) * 100}%` }}></div>
                                    </div>
                                    <span className="count">{count}</span>
                                </div>
                            ))}
                        </div>
                    </section>
                </aside>
            </div>
        </div>
    );
}
