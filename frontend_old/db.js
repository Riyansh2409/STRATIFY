// db.js
// Database configuration and logic for Login and Signup
// This connects the frontend to the Supabase Database and handles saving records.

// 1. IMPORTANT: Replace this with your actual Supabase Project URL
const SUPABASE_URL = 'https://desgrqixwjkbplgjenyk.supabase.co'; 

// Using the Publishable Key from your configuration
const SUPABASE_PUBLISHABLE_KEY = 'sb_publishable_lmH8vSiBwWMe6yrZ2Xo1lQ_49cm8sU2';

// Initialize Supabase Client securely
let supabase;
try {
    if (window.supabase) {
        supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);
    } else {
        console.error("Supabase script failed to load from JS CDN. Check your internet or ad-blocker.");
    }
} catch (e) {
    console.error("Failed to initialize Supabase:", e);
}

/**
 * DATABASE STRUCTURE REQUIRED in Supabase:
 * Table Name: `user_profiles`
 * Columns:
 *  - id (uuid, primary key, linked to auth.users)
 *  - full_name (text)
 *  - email (text)
 *  - company_name (text)
 *  - niche_industry (text)
 *  - contact_details (text)
 */

// Function to collect signup details and save to database
async function saveUserToDatabase(fullName, companyName, niche, contact, email, password) {
    if (!supabase) return { success: false, message: "Database connection script failed to load. Please refresh the page or check internet connection." };
    if (SUPABASE_URL.includes('YOUR_PROJECT_ID')) {
        return { success: false, message: "CRITICAL: Please copy your Supabase Project URL and replace 'YOUR_PROJECT_ID.supabase.co' in db.js file!" };
    }
    
    try {
        // Step 1: Create the User Auth Profile in Supabase
        const { data: authData, error: authError } = await supabase.auth.signUp({
            email: email,
            password: password,
            options: {
                data: {
                    full_name: fullName,
                    company_name: companyName,
                    niche: niche
                }
            }
        });

        if (authError) {
            return { success: false, message: authError.message };
        }

        // Step 2: Save the extended details into our custom 'user_profiles' table with columns
        if (authData.user) {
            const { error: dbError } = await supabase
                .from('user_profiles')
                .insert([
                    { 
                        id: authData.user.id,           // Linking the auth UUID
                        full_name: fullName, 
                        email: email,
                        company_name: companyName, 
                        niche_industry: niche,
                        contact_details: contact
                    }
                ]);
            
            if (dbError) {
                console.warn("User signed up, but failed to insert into 'user_profiles' table. Ensure the table is created in Supabase with exact columns.", dbError.message);
                // Return success anyway since auth succeeded, but mention the warning in console
            }
        }

        return { success: true, data: authData };
    } catch (error) {
        return { success: false, message: error.message || 'An unexpected error occurred during signup' };
    }
}

// Function to Authenticate / Login User
async function authenticateUser(email, password) {
    if (!supabase) return { success: false, message: "Database connection script failed to load. Please refresh the page." };
    if (SUPABASE_URL.includes('YOUR_PROJECT_ID')) {
        return { success: false, message: "CRITICAL: Please copy your Supabase Project URL and replace 'YOUR_PROJECT_ID.supabase.co' in db.js file!" };
    }

    try {
        const { data, error } = await supabase.auth.signInWithPassword({
            email: email,
            password: password,
        });

        if (error) {
            return { success: false, message: error.message };
        }

        return { success: true, data: data };
    } catch (error) {
        return { success: false, message: error.message || 'An unexpected error occurred during login' };
    }
}

// Function to Logout User
async function logoutUser() {
    const { error } = await supabase.auth.signOut();
    return { success: !error, message: error?.message };
}

// Function to Check Active Session on Page Load
async function checkActiveSession() {
    if (!supabase) return null;
    if (SUPABASE_URL.includes('YOUR_PROJECT_ID')) return null;
    
    try {
        const { data: { session } } = await supabase.auth.getSession();
        return session;
    } catch(e) {
        return null;
    }
}
