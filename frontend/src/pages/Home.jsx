import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';

export default function Home({ onAuthRequired }) {
    const { user } = useAuth();
    const [jsonOutput, setJsonOutput] = useState('');
    const [isDragover, setIsDragover] = useState(false);
    const fileInputRef = useRef(null);

    const handleDropboxClick = (e) => {
        // Prevent click if they clicked inside the json output itself
        if (e.target.id === 'json-output') return;

        if (!user) {
            onAuthRequired();
        } else {
            fileInputRef.current.click();
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragover(true);
    };

    const handleDragLeave = () => {
        setIsDragover(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragover(false);

        if (!user) {
            onAuthRequired();
            return;
        }

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            processFiles(e.dataTransfer.files);
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            processFiles(e.target.files);
        }
    };

    const processFiles = (files) => {
        const fileList = Array.from(files).map(f => ({
            name: f.name,
            size: (f.size / 1024).toFixed(2) + ' KB',
            type: f.type || 'unknown'
        }));

        const jsonStr = JSON.stringify({
            status: "success",
            message: "Files processed successfully",
            files: fileList
        }, null, 2);

        setJsonOutput(jsonStr);
    };

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

    return (
        <main>
            <div className="glow-wrapper">
                <div className="glow-1"></div>
                <div className="glow-2"></div>
            </div>

            <div className="hero-text">
                <h1>Data Intelligence, Simplified.</h1>
                <p>Upload your JSON or CSV data, extract insights from the web, and run powerful trained analysis models in seconds.</p>
            </div>

            <div className="dropbox-wrapper">
                <div 
                    className={`dropbox-container ${isDragover ? 'dragover' : ''}`} 
                    onClick={handleDropboxClick}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                >
                    <div className="dropbox-icon">📁</div>
                    <div className="dropbox-text">Drop your files here</div>
                    <div className="dropbox-subtext">Supports JSON, CSV, and TXT (Max 50MB)</div>
                    <input 
                        type="file" 
                        ref={fileInputRef} 
                        style={{ display: 'none' }} 
                        multiple 
                        accept=".json,.csv,.txt" 
                        onChange={handleFileChange}
                    />
                    {jsonOutput && (
                        <div id="json-output" style={{ display: 'block', marginTop: '1rem', padding: '1rem', background: '#f1f5f9', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.85rem', color: '#059669', textAlign: 'left', overflowX: 'auto' }}>
                            {jsonOutput}
                        </div>
                    )}
                </div>
            </div>

            <div className="features-section" id="features">
                <h2 className="features-title">Core Capabilities</h2>
                <div className="features-grid">
                    <div className="feature-card">
                        <h3 className="feature-title">Web Scraping</h3>
                        <p className="feature-desc">Automatically extract, parse, and structure data from any website. Turn unstructured web content into actionable JSON datasets instantly without writing code.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">Data Analysis</h3>
                        <p className="feature-desc">Perform complex statistical analysis, generate rich visualizations, and uncover hidden trends within your uploaded datasets with ease.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">Trained Analysis</h3>
                        <p className="feature-desc">Leverage custom AI models to classify, predict, and gain deep insights from your data using our advanced state-of-the-art machine learning pipeline.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">Cross-Platform</h3>
                        <p className="feature-desc">Access your data intelligence dashboard seamlessly across desktop, tablet, and mobile devices with zero loss of functionality.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">Actionable Insights</h3>
                        <p className="feature-desc">Convert raw data directly into clear, actionable business strategies with our automated insight generation engine.</p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">Real-Time Processing</h3>
                        <p className="feature-desc">Experience lightning-fast processing for huge datasets, ensuring you have the data you need exactly when you need it.</p>
                    </div>
                </div>
            </div>
        </main>
    );
}
