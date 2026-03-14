import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import AuthModal from './components/AuthModal';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import About from './pages/About';
import Pricing from './pages/Pricing';
import { useAuth } from './context/AuthContext';

function App() {
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const { loading } = useAuth();

  if (loading) {
      return <div style={{ minHeight: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>Loading...</div>;
  }

  return (
    <>
      <Navbar onAuthClick={() => setIsAuthModalOpen(true)} />
      
      <Routes>
        <Route path="/" element={<Home onAuthRequired={() => setIsAuthModalOpen(true)} />} />
        <Route path="/about" element={<About />} />
        <Route path="/pricing" element={<Pricing />} />
      </Routes>

      <Footer />
      
      <AuthModal isOpen={isAuthModalOpen} onClose={() => setIsAuthModalOpen(false)} />
    </>
  );
}

export default App;
