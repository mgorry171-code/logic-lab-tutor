import streamlit as st
import sympy
from sympy import symbols, solve, Eq, latex, simplify, I, pi, E, diff, integrate, limit, oo, Matrix, factorial, Function, Derivative, Integral, ImmutableDenseMatrix
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import datetime
import time
import pandas as pd
import re
import numpy as np
import plotly.graph_objects as go
import requests
import base64
import statistics

# --- CONFIG ---
st.set_page_config(page_title="The Logic Lab", page_icon="🧪", layout="centered")

# --- CUSTOM CSS (BRANDING & UI) ---
st.markdown("""
<style>
    /* 1. BRANDING & COLORS */
    :root {
        --regents-blue: #1a73e8;
        --dark-bg: #262730;
    }
    
    html, body, [class*="css"] { font-family: 'Segoe UI', Roboto, sans-serif; }
    
    .main-header {
        text-align: center;
        padding: 15px;
        background-color: var(--regents-blue);
        color: white;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    h1 { font-size: 24px !important; margin: 0 !important; }
    p { margin: 0 !important; }

    /* 2. REGENTS MODE DASHBOARD */
    .dashboard {
        display: flex;
        justify-content: space-around;
        background: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
    }
    .stat-item { text-align: center; font-weight: bold; color: #495057; font-size: 18px; }

    /* 3. BLACKBOARD BUTTONS */
    div.stButton > button {
        width: 100%; height: 50px; 
        border-radius: 10px; border: 1px solid #4a4a4a;
        background-color: #262730 !important; 
        -webkit-appearance: none !important;
        transition: all 0.1s;
    }
    div.stButton > button * { color: #ffffff !important; font-size: 22px !important; font-weight: 700 !important; }
    div.stButton > button:active { background-color: #000000 !important; transform: scale(0.98); }

    /* 4. INPUTS */
    [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] div:has(> div > div > input[aria-label="Previous Line"]) input {
        background-color: #f1f3f4 !important; color: #202124 !important; border: 1px solid #dadce0 !important;
    }
    [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] div:has(> div > div > input[aria-label="Current Line"]) input {
        background-color: #ffffff !important; border: 2px solid var(--regents-blue) !important; 
    }

    /* 5. FEEDBACK */
    .success-box { padding: 15px; background: #d1e7dd; color: #0f5132; border-radius: 10px; text-align: center; border: 1px solid #badbcc; }
    .warning-box { padding: 15px; background: #fff3cd; color: #664d03; border-radius: 10px; text-align: center; border: 1px solid #ffecb5; }
    .error-box { padding: 15px; background: #f8d7da; color: #842029; border-radius: 10px; text-align: center; border: 1px solid #f5c2c7; }
    
    /* 6. LEADERBOARD */
    .leaderboard { margin-top: 30px; padding: 15px; background: #fff; border-radius: 10px; border: 1px solid #e0e0e0; }
    .leaderboard h3 { color: #1a73e8; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'line_prev' not in st.session_state: st.session_state.line_prev = "" 
if 'line_curr' not in st.session_state: st.session_state.line_curr = ""
if 'step_verified' not in st.session_state: st.session_state.step_verified = False
if 'original_solution_set' not in st.session_state: st.session_state.original_solution_set = None

# REGENTS MODE & LEADERBOARD
if 'start_time' not in st.session_state: st.session_state.start_time = None
if 'hint_count' not in st.session_state: st.session_state.hint_count = 0
if 'problem_solved' not in st.session_state: st.session_state.problem_solved = False
if 'high_scores' not in st.session_state: st.session_state.high_scores = []

# --- HELPERS ---
def clear_all():
    st.session_state.line_prev = ""
    st.session_state.line_curr = ""
    st.session_state.step_verified = False
    st.session_state.original_solution_set = None
    st.session_state.start_time = None
    st.session_state.hint_count = 0
    st.session_state.problem_solved = False

def next_step():
    st.session_state.line_prev = st.session_state.line_curr
    st.session_state.line_curr = ""
    st.session_state.step_verified = False

def add_to_input(text_to_add):
    if st.session_state.start_time is None: st.session_state.start_time = time.time()
    if st.session_state.keypad_target == "Previous Line": st.session_state.line_prev += text_to_add
    else: st.session_state.line_curr += text_to_add

def clean_input(text):
    text = text.lower().replace("＋", "+").replace("－", "-")
    text = text.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "").replace("\\", "").replace("`", "")
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    text = text.replace(" and ", ",").replace(" or ", ",").replace("^", "**").replace("√", "sqrt")
    return text

def safe_parse_latex(text_str):
    """Robust parser for display only - handles = signs gracefully"""
    try:
        clean = clean_input(text_str)
        if "=" in clean:
            parts = clean.split("=")
            lhs = parse_expr(parts[0], transformations=standard_transformations)
            rhs = parse_expr(parts[1], transformations=standard_transformations)
            return latex(Eq(lhs, rhs))
        else:
            return latex(parse_expr(clean, transformations=standard_transformations))
    except:
        return text_str # Fallback to raw text if parsing fails

def parse_for_logic(text):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try:
        logic_dict = {'e': E, 'pi': pi, 'diff': diff, 'integrate': integrate, 'limit': limit, 'oo': oo, 'matrix': ImmutableDenseMatrix, 'factorial': factorial, 'mean': statistics.mean, 'median': statistics.median}
        if "=" in text:
            parts = text.split("=")
            return Eq(parse_expr(parts[0], transformations=transformations, evaluate=True, local_dict=logic_dict), parse_expr(parts[1], transformations=transformations, evaluate=True, local_dict=logic_dict))
        return parse_expr(text, transformations=transformations, evaluate=True, local_dict=logic_dict)
    except: return sympy.sympify(text, evaluate=True)

def get_solution_set(text_str):
    clean = clean_input(text_str)
    try:
        # Check if it's a simple final answer (e.g., "-1" or "2")
        if not any(c in clean for c in "+*/^=") and not clean.startswith("matrix"):
            return sympy.FiniteSet(parse_for_logic(clean))
            
        if "±" in clean:
            val = parse_for_logic(clean.split("±")[1].strip())
            return sympy.FiniteSet(val, -val)
        
        # Handle lists like "x = 2, -1"
        if "=" in clean and "," in clean:
            rhs = clean.split("=")[1]
            return sympy.FiniteSet(*[parse_for_logic(i.strip()) for i in rhs.split(",") if i.strip()])

        expr = parse_for_logic(clean)
        all_symbols = list(expr.free_symbols)
        if not all_symbols: return sympy.FiniteSet(expr)
        
        sol = solve(expr, all_symbols, set=True)
        return sympy.FiniteSet(*sol[1]) if isinstance(sol, tuple) else sol
    except: return None

def validate_step(line_a, line_b):
    try:
        set_A = get_solution_set(line_a)
        set_B = get_solution_set(line_b)
        
        if st.session_state.original_solution_set is None: st.session_state.original_solution_set = set_A
        
        # FINAL ANSWER DETECTION
        clean_b = clean_input(line_b)
        is_final = False
        # If it's just a number OR it starts with "var =" and has no ops on RHS
        if not any(c in clean_b for c in "+*^"):
            if "=" in clean_b:
                if not any(c.isalpha() for c in clean_b.split("=")[1]): is_final = True
            else: is_final = True

        if set_A == set_B: return True, ("Final" if is_final else "Valid"), ""
        
        if set_A.issubset(set_B) and set_A != set_B:
            if is_final and st.session_state.original_solution_set != set_B:
                return True, "Warning", "Check BOTH solutions in the original equation!"
            return True, "Valid", ""
            
        return False, "Invalid", "Values do not match."
    except: return False, "Error", ""

# --- UI START ---
st.markdown('<div class="main-header"><h1>🧪 THE LOGIC LAB</h1><p>NYC Regents Step-Checker</p></div>', unsafe_allow_html=True)

# SIDEBAR: SETTINGS & PARENT MODE
with st.sidebar:
    st.header("⚙️ Settings")
    parent_mode = st.toggle("👨‍👩‍👧 Parent Mode")
    if parent_mode:
        st.info("Parent Mode Active: Hints are simplified.")
    if st.button("🗑️ Clear Leaderboard"):
        st.session_state.high_scores = []
        st.rerun()

# DASHBOARD
col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    elapsed = 0
    if st.session_state.start_time and not st.session_state.problem_solved:
        elapsed = int(time.time() - st.session_state.start_time)
    st.markdown(f"<div class='stat-item'>⏱️ {elapsed}s</div>", unsafe_allow_html=True)
with col_d2:
    st.markdown(f"<div class='stat-item'>💡 {st.session_state.hint_count} Hints</div>", unsafe_allow_html=True)
with col_d3:
    if st.button("✨ NEW", key="new_btn"): clear_all(); st.rerun()

# WORKSPACE
st.text_input("Previous Line", key="line_prev", label_visibility="collapsed", placeholder="Problem...", help="Previous Line")
if st.session_state.line_prev: 
    # USE SAFE PARSER HERE
    st.latex(safe_parse_latex(st.session_state.line_prev))

st.markdown("---")

st.text_input("Current Line", key="line_curr", label_visibility="collapsed", placeholder="Your next step...", help="Current Line")
if st.session_state.line_curr: 
    st.latex(safe_parse_latex(st.session_state.line_curr))

# KEYPAD
with st.expander("⌨️ MATH KEYPAD", expanded=True):
    st.radio("Target:", ["Previous Line", "Current Line"], horizontal=True, key="keypad_target", label_visibility="collapsed")
    t1, t2, t3 = st.tabs(["Algebra", "Calculus", "Advanced"])
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.button("x", on_click=add_to_input, args=("x",), key="k_x"); c2.button("x²", on_click=add_to_input, args=("^2",), key="k_sq")
        c3.button("＋", on_click=add_to_input, args=("+",), key="k_p"); c4.button("－", on_click=add_to_input, args=("－",), key="k_m")
        c1.button("√", on_click=add_to_input, args=("sqrt(",), key="k_rt"); c2.button("÷", on_click=add_to_input, args=("/",), key="k_d")
        c3.button("(", on_click=add_to_input, args=("(",), key="k_o"); c4.button(")", on_click=add_to_input, args=(")",), key="k_c")
        c1.button("=", on_click=add_to_input, args=("=",), key="k_eq"); c2.button(",", on_click=add_to_input, args=(",",), key="k_cm")

    with t2:
        c1, c2, c3, c4 = st.columns(4)
        c1.button("d/dx", on_click=add_to_input, args=("diff(",), key="c_df"); c2.button("∫", on_click=add_to_input, args=("integrate(",), key="c_in")
        c3.button("lim", on_click=add_to_input, args=("limit(",), key="c_lm"); c4.button("∞", on_click=add_to_input, args=("oo",), key="c_oo")
    
    with t3:
        c1, c2, c3, c4 = st.columns(4)
        c1.button("Mean", on_click=add_to_input, args=("mean(",), key="s_mn"); c2.button("Med", on_click=add_to_input, args=("median(",), key="s_md")
        c3.button("Std", on_click=add_to_input, args=("stdev(",), key="s_sd"); c4.button("Mat", on_click=add_to_input, args=("Matrix([",), key="m_mx")


# ACTIONS
if not st.session_state.problem_solved:
    c_check, c_next = st.columns(2)
    with c_check:
        if st.button("CHECK LOGIC", type="primary"):
            if not st.session_state.start_time: st.session_state.start_time = time.time()
            ok, status, hint = validate_step(st.session_state.line_prev, st.session_state.line_curr)
            if ok:
                st.session_state.step_verified = True
                if status == "Final":
                    st.session_state.problem_solved = True
                    final_time = int(time.time() - st.session_state.start_time)
                    st.session_state.high_scores.append({"Time": f"{final_time}s", "Hints": st.session_state.hint_count, "Date": datetime.datetime.now().strftime("%H:%M")})
                    st.balloons()
                    st.success(f"🏆 Solved in {final_time}s!")
                elif status == "Warning":
                    st.markdown(f"<div class='warning-box'>⚠️ {hint}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='success-box'>✅ Correct! Keep going.</div>", unsafe_allow_html=True)
            else:
                st.session_state.hint_count += 1
                st.markdown(f"<div class='error-box'>❌ Logic Break. Hint: {hint}</div>", unsafe_allow_html=True)
    with c_next:
        if st.session_state.step_verified:
            st.button("NEXT STEP ⬇️", on_click=next_step)
else:
    st.success("✨ Problem Complete! Click NEW to start again.")

# LEADERBOARD
if st.session_state.high_scores:
    st.markdown("<div class='leaderboard'><h3>🏆 Session High Scores</h3>", unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.high_scores)
    st.table(df)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer-note'>Built for NYC Math Teachers • The Logic Lab v15.1</div>", unsafe_allow_html=True)
