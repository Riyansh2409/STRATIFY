import React from 'react';

export default function Pricing() {
    return (
        <main>
            <div className="glow-wrapper">
                <div className="glow-1"></div>
            </div>

            <div className="hero-text">
                <h1>Simple, Transparent Pricing</h1>
                <p>Choose the plan that best fits your data intelligence needs. No hidden fees, ever.</p>
            </div>

            <div className="pricing-grid">
                
                {/* Basic Plan */}
                <div className="pricing-card">
                    <h3 className="plan-name">Basic</h3>
                    <div className="plan-price">$29<span>/mo</span></div>
                    <p className="plan-desc">Perfect for individuals and small teams getting started with data analysis.</p>
                    
                    <ul className="plan-features">
                        <li>5 GB Storage Space</li>
                        <li>100 Web Scrapes / month</li>
                        <li>Basic Visualizations</li>
                        <li>Community Support</li>
                    </ul>
                    <button className="pricing-btn">Get Started</button>
                </div>

                {/* Pro Plan */}
                <div className="pricing-card popular">
                    <div className="popular-badge">Popular</div>
                    <h3 className="plan-name">Pro</h3>
                    <div className="plan-price">$98<span>/mo</span></div>
                    <p className="plan-desc">Advanced features for professional data scientists and growing businesses.</p>
                    
                    <ul className="plan-features">
                        <li>50 GB Storage Space</li>
                        <li>Unlimited Web Scrapes</li>
                        <li>Advanced AI Models</li>
                        <li>Priority Email Support</li>
                    </ul>
                    <button className="pricing-btn">Get Started</button>
                </div>

                {/* Business Plan */}
                <div className="pricing-card">
                    <h3 className="plan-name">Business</h3>
                    <div className="plan-price">$198<span>/mo</span></div>
                    <p className="plan-desc">Enterprise-grade solution with unlimited capabilities and dedicated support.</p>
                    
                    <ul className="plan-features">
                        <li>Unlimited Storage Space</li>
                        <li>Custom AI Training</li>
                        <li>Real-time API Access</li>
                        <li>24/7 Dedicated Support</li>
                    </ul>
                    <button className="pricing-btn">Contact Sales</button>
                </div>

            </div>
        </main>
    );
}
