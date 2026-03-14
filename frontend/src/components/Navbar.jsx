import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Navbar({ onAuthClick }) {
    const { user, logout } = useAuth();

    return (
        <header>
            <NavLink to="/" className="logo" style={{ textDecoration: 'none' }}>Stratify</NavLink>
            <nav>
                <ul>
                    <li><NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')}>Home</NavLink></li>
                    <li><NavLink to="/dashboard" className={({ isActive }) => (isActive ? 'active' : '')}>Dashboard</NavLink></li>
                    <li><NavLink to="/about" className={({ isActive }) => (isActive ? 'active' : '')}>About</NavLink></li>
                    <li><NavLink to="/pricing" className={({ isActive }) => (isActive ? 'active' : '')}>Pricing</NavLink></li>
                </ul>
            </nav>
        </header>
    );
}
