import React, { useState } from 'react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';

export default function AuthModal({ isOpen, onClose }) {
    const { user } = useAuth();
    const [isLoginView, setIsLoginView] = useState(true);
    const [errorMsg, setErrorMsg] = useState('');
    const [loading, setLoading] = useState(false);

    // Form inputs
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [contact, setContact] = useState('');
    const [company, setCompany] = useState('');
    const [niche, setNiche] = useState('');

    if (!isOpen) return null;

    const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        setErrorMsg('');
        
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        
        if (error) {
            setErrorMsg(error.message);
        } else {
            alert('Successfully Logged In!');
            onClose();
        }
        setLoading(false);
    };

    const handleSignup = async (e) => {
        e.preventDefault();
        setLoading(true);
        setErrorMsg('');

        try {
            const { data: authData, error: authError } = await supabase.auth.signUp({
                email,
                password,
                options: { data: { full_name: name, company_name: company, niche } }
            });

            if (authError) throw authError;

            if (authData.user) {
                const { error: dbError } = await supabase.from('user_profiles').insert([{ 
                    id: authData.user.id, 
                    full_name: name, 
                    email, 
                    company_name: company, 
                    niche_industry: niche, 
                    contact_details: contact
                }]);
                
                if (dbError) console.warn("Profile table insert failed:", dbError.message);
            }

            alert('Account Created Successfully! Check your email if verification is required.');
            onClose();
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="modal-overlay active">
            <div className="login-card">
                <h2>{isLoginView ? 'Welcome Back' : 'Create Profile'}</h2>
                <p>{isLoginView ? 'Please login to your profile to continue.' : 'Sign up to unleash data intelligence.'}</p>
                
                {errorMsg && <div style={{ color: '#ef4444', fontSize: '0.9rem', marginBottom: '1rem' }}>{errorMsg}</div>}

                {isLoginView ? (
                    <form onSubmit={handleLogin}>
                        <input type="email" placeholder="Email ID" className="input-field" required value={email} onChange={(e) => setEmail(e.target.value)} />
                        <input type="password" placeholder="Password" className="input-field" required value={password} onChange={(e) => setPassword(e.target.value)} />
                        <button type="submit" className="auth-btn" disabled={loading}>{loading ? 'Logging in...' : 'Login'}</button>
                        <p className="toggle-text">Don't have an account? <span className="toggle-link" onClick={() => { setIsLoginView(false); setErrorMsg(''); }}>Sign up</span></p>
                    </form>
                ) : (
                    <form onSubmit={handleSignup}>
                        <input type="text" placeholder="Full Name" className="input-field" required value={name} onChange={(e) => setName(e.target.value)} />
                        <input type="email" placeholder="Email Address" className="input-field" required value={email} onChange={(e) => setEmail(e.target.value)} />
                        <input type="text" placeholder="Phone / Alt Contact" className="input-field" value={contact} onChange={(e) => setContact(e.target.value)} />
                        <input type="text" placeholder="Company Name" className="input-field" required value={company} onChange={(e) => setCompany(e.target.value)} />
                        <input type="text" placeholder="Niche / Industry" className="input-field" required value={niche} onChange={(e) => setNiche(e.target.value)} />
                        <input type="password" placeholder="Password" className="input-field" required value={password} onChange={(e) => setPassword(e.target.value)} />
                        <button type="submit" className="auth-btn" disabled={loading}>{loading ? 'Creating...' : 'Create Account'}</button>
                        <p className="toggle-text">Already have an account? <span className="toggle-link" onClick={() => { setIsLoginView(true); setErrorMsg(''); }}>Login</span></p>
                    </form>
                )}

                <button type="button" className="close-modal" onClick={onClose}>Close</button>
            </div>
        </div>
    );
}
