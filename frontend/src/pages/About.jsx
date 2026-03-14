import React from 'react';

export default function About() {
    return (
        <main>
            <div className="glow-wrapper">
                <div className="glow-2"></div>
            </div>

            <div className="hero-text">
                <h1>About Stratify</h1>
                <p>Empowering businesses with seamless, actionable data intelligence.</p>
            </div>

            <section className="about-section">
                <div className="about-card">
                    <h2>Our Mission</h2>
                    <p>
                        At Stratify, we believe that data intelligence shouldn't be confined to data scientists and engineers. 
                        Our mission is to democratize data analytics, making it accessible, understandable, and actionable for everyone.
                    </p>
                    
                    <h2>What We Do</h2>
                    <p>
                        We provide a comprehensive platform that bridges the gap between raw data and business strategy. 
                        Whether you are extracting data from the web, cleaning massive datasets, or running advanced machine learning models, 
                        Stratify orchestrates the complexity behind the scenes so you can focus on the results.
                    </p>

                    <h2>Our Core Values</h2>
                    <ul>
                        <li><strong style={{color: 'var(--text-color)'}}>Simplicity:</strong> Complex workflows turned into a single drag-and-drop interface.</li>
                        <li><strong style={{color: 'var(--text-color)'}}>Innovation:</strong> Always pushing the boundaries with state-of-the-art AI.</li>
                        <li><strong style={{color: 'var(--text-color)'}}>Security:</strong> Your data's privacy and security are our top priority.</li>
                    </ul>
                </div>
            </section>
        </main>
    );
}
