import React, { useState, useEffect, useRef } from 'react';

const API_URL = 'http://localhost:9000';

export default function Dashboard() {
    const [stats, setStats] = useState(null);
    const [charts, setCharts] = useState({});
    const [aiRec, setAiRec] = useState('');
    const [chatMessages, setChatMessages] = useState([
        { role: 'assistant', content: 'Hello! I have indexed your data. You can ask me anything about the uploaded documents.' }
    ]);
    const [chatInput, setChatInput] = useState('');
    const [isChatLoading, setIsChatLoading] = useState(false);
    const [isExporting, setIsExporting] = useState(false);

    // Scientific Controls Scale
    const [testOptions, setTestOptions] = useState({
        chi: true,
        ttest: false,
        anova: false
    });

    const chatEndRef = useRef(null);

    useEffect(() => {
        fetchStatus();
    }, []);

    // More robust animation - using a small timeout to ensure DOM is ready
    useEffect(() => {
        if (stats && window.gsap) {
            const timer = setTimeout(() => {
                window.gsap.from(".kpi-card", {
                    y: 20,
                    opacity: 0,
                    duration: 0.6,
                    stagger: 0.05,
                    ease: "power2.out"
                });
                window.gsap.from(".ragoo-section", {
                    y: 30,
                    opacity: 0,
                    duration: 0.8,
                    ease: "power2.out"
                });
                window.gsap.from(".insight-card", {
                    scale: 0.98,
                    opacity: 0,
                    duration: 0.6,
                    stagger: 0.05,
                    ease: "power2.out",
                    delay: 0.2
                });
            }, 100);
            return () => clearTimeout(timer);
        }
    }, [stats]);

    const fetchStatus = async () => {
        try {
            const res = await fetch(`${API_URL}/status`);
            const json = await res.json();
            if (json.status === 'ready') {
                setStats(json.stats);
                setCharts(json.charts);
                fetchAiRecommend();
            }
        } catch (e) {
            console.error("Failed to fetch dashboard data", e);
        }
    };

    const fetchAiRecommend = async () => {
        try {
            const res = await fetch(`${API_URL}/ai/recommend`, { method: 'POST' });
            const json = await res.json();
            if (json.recommendation) setAiRec(json.recommendation);
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

    const handleExport = () => {
        setIsExporting(true);
        const query = `chi=${testOptions.chi}&ttest=${testOptions.ttest}&anova=${testOptions.anova}`;
        window.open(`${API_URL}/download-report?${query}`, '_blank');
        setTimeout(() => setIsExporting(false), 2000);
    };

    const toggleTest = (key) => {
        setTestOptions(prev => ({ ...prev, [key]: !prev[key] }));
    };

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatMessages]);

    if (!stats) return (
        <div className="premium-loader">
            <div className="pulse-circle"></div>
            <p style={{ color: '#10b981', marginTop: '1rem' }}>Initiating Intelligence Protocols...</p>
        </div>
    );

    return (
        <div className="premium-dashboard">
            {/* ── Top Bar ── */}
            <div className="db-top-bar">
                <h1>Intelligence Center</h1>
                <div className="action-group">
                    <button className="premium-btn secondary" onClick={() => window.location.reload()}>
                        <span className="icon">🔄</span> Sync
                    </button>
                    <button
                        className={`premium-btn primary ${isExporting ? 'loading' : ''}`}
                        onClick={handleExport}
                        disabled={isExporting}
                    >
                        <span className="icon">{isExporting ? '⏳' : '📥'}</span>
                        {isExporting ? 'Generating...' : 'Export PDF'}
                    </button>
                </div>
            </div>

            {/* ── KPI Grid ── */}
            <div className="kpi-grid">
                <div className="kpi-card">
                    <span className="kpi-label">Corpus Size</span>
                    <span className="kpi-value">{stats.total_files || 0} <small style={{ fontSize: '0.9rem', color: '#64748b' }}>Files</small></span>
                </div>
                <div className="kpi-card">
                    <span className="kpi-label">Text Volume</span>
                    <span className="kpi-value">
                        {Number(stats.total_raw_chars || 0).toLocaleString()} <small style={{ fontSize: '0.9rem', color: '#64748b' }}>chars</small>
                    </span>
                </div>
                <div className="kpi-card">
                    <span className="kpi-label">Neural Chunks</span>
                    <span className="kpi-value">{stats.chunks_after_filter || 0}</span>
                </div>
                <div className="kpi-card highlight">
                    <span className="kpi-label">Quality Index</span>
                    <span className="kpi-value">{stats.filter_rate_pct || 0}%</span>
                </div>
            </div>

            <div className="dashboard-layout">
                {/* ── Main Content Area ── */}
                <div className="layout-left">
                    {/* ── RAGOO SECTION ── */}
                    <section className="ragoo-section">
                        <div className="section-header">
                            <h2><span className="icon">🚀</span> Ragoo</h2>
                            <span>Neural Dataset Interaction</span>
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
                                        <div className="text-content">Processing neural nodes...</div>
                                    </div>
                                )}
                                <div ref={chatEndRef} />
                            </div>
                            <form className="chat-input-area" onSubmit={handleChat}>
                                <input
                                    type="text"
                                    placeholder="Enter query for document synthesis..."
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
                            <h2><span className="icon">📊</span> Data Visualizations</h2>
                            <span>Real-time synthesized insights</span>
                        </div>
                        <div className="charts-masonry">
                            {Object.entries(charts || {}).map(([label, filename]) => (
                                <div key={filename} className="insight-card visible">
                                    <div className="card-media">
                                        <img
                                            src={`${API_URL}/charts/${filename}`}
                                            alt={label}
                                            onLoad={() => console.log(`✓ Loaded: ${filename}`)}
                                            onError={(e) => {
                                                console.error(`✗ Failed to load: ${filename}`);
                                                e.target.src = 'https://via.placeholder.com/400x300?text=Visualization+Loading...';
                                            }}
                                        />
                                    </div>
                                    <div className="card-info">
                                        <h3 style={{ color: '#fff', fontSize: '1.2rem', textShadow: '0 2px 4px rgba(0,0,0,0.5)' }}>{label}</h3>
                                        <div className="card-actions">
                                            <button className="expand-overlay-btn" onClick={() => window.open(`${API_URL}/charts/${filename}`, '_blank')}>
                                                View ↗
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            ))}
                            {(!charts || Object.keys(charts).length === 0) && (
                                <div className="insight-card empty-state">
                                    <div className="card-info">
                                        <p>No visualizations detected. Run the intelligence pipeline to generate insights.</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </section>
                </div>

                {/* ── Right Sidebar ── */}
                <aside className="layout-right">
                    {/* ── SCIENTIFIC CONTROLS ── */}
                    <section className="scientific-controls-card ai-advisor-card">
                        <div className="header">
                            <span className="tag">SCIENTIFIC CONTROLS</span>
                            <h3>Statistical Analysis</h3>
                        </div>
                        <div className="controls-content" style={{ padding: '0 1.5rem 1.5rem' }}>
                            <p style={{ fontSize: '0.85rem', color: '#888', marginBottom: '1rem' }}>Select tests for inclusion in global intelligence report:</p>
                            <div className="test-options-grid" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                                <label className="test-option-row" style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                    <input type="checkbox" checked={testOptions.chi} onChange={() => toggleTest('chi')} />
                                    <span>Chi-Square Analysis</span>
                                </label>
                                <label className="test-option-row" style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                    <input type="checkbox" checked={testOptions.ttest} onChange={() => toggleTest('ttest')} />
                                    <span>T-Test Distribution</span>
                                </label>
                                <label className="test-option-row" style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                    <input type="checkbox" checked={testOptions.anova} onChange={() => toggleTest('anova')} />
                                    <span>ANOVA Variance</span>
                                </label>
                            </div>
                        </div>
                    </section>

                    <section className="ai-advisor-card">
                        <div className="header">
                            <span className="tag">AI STRATEGIST</span>
                            <h3>Neural Insights</h3>
                        </div>
                        <div className="advisor-content">
                            <p>{aiRec || "Synthesizing cross-document recommendations..."}</p>
                        </div>
                        <div className="advisor-footer">
                            <small>Statistical Inference Active</small>
                        </div>
                    </section>

                    <section className="dataset-health">
                        <h3>Language Distribution</h3>
                        <div className="health-bar-container">
                            {Object.entries(stats.language_distribution || {}).map(([lang, count]) => (
                                <div key={lang} className="health-row">
                                    <div className="label">
                                        <span>{lang.toUpperCase()}</span>
                                        <span className="count">{count}</span>
                                    </div>
                                    <div className="bar-bg">
                                        <div className="bar-fill" style={{ width: `${(count / (stats.total_files || 1)) * 100}%` }}></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                </aside>
            </div>
        </div>
    );
}
