import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = 'https://desgrqixwjkbplgjenyk.supabase.co'; 
const SUPABASE_PUBLISHABLE_KEY = 'sb_publishable_lmH8vSiBwWMe6yrZ2Xo1lQ_49cm8sU2';

export const supabase = createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);
