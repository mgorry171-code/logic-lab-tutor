import streamlit as st
import sympy
from sympy import symbols, solve, Eq, latex, simplify, I, pi, E, diff, integrate, limit, oo, Matrix, factorial, Function, Derivative, Integral, ImmutableDenseMatrix, FiniteSet
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
st.set_page_config(page_title="The Logic Lab", page_icon="🦉", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    :root { --brand-color: #008080; --accent-color: #DAA520; }
    html, body, [class*="css"] { font-family: 'Segoe UI', Roboto, sans-serif; }
    .main-header { text-align: center; padding: 15px; background-color: var(--brand-color); color: white; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-bottom: 4px solid var(--accent-color); }
    h1 { font-size: 24px !important; margin: 0 !important; }
    p { margin: 0 !important; }
    .stat-item { text-align: center; font-weight: 800; color: #495057; font-size: 26px; margin-top: 5px; }
    div.stButton > button { width: 100%; height: 50px; border-radius: 10px; border: 1px solid #4a4a4a; background-color: #262730 !important; -webkit-appearance: none !important; transition: all 0.1s; }
    div.stButton > button * { color: #ffffff !important; font-size: 22px !important; font-weight: 700 !important; }
    div.stButton > button:active { background-color: #000000 !important; transform: scale(0.98); }
    [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] div:has(> div > div > input[aria-label="Previous Line"]) input { background-color: #f1f3f4 !important; color: #202124 !important; border: 1px solid #dadce0 !important; }
    [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] div:has(> div > div > input[aria-label="Current Line"]) input { background-color: #ffffff !important; border: 2px solid var(--accent-color) !important; }
    .success-box { padding: 15px; background: #d1e7dd; color: #0f5132; border-radius: 10px; text-align: center; border: 1px solid #badbcc; }
    .warning-box { padding: 15px; background: #fff3cd; color: #664d03; border-radius: 10px; text-align: center; border: 1px solid #ffecb5; }
    .error-box { padding: 15px; background: #f8d7da; color: #842029; border-radius: 10px; text-align: center; border: 1px solid #f5c2c7; }
    .leaderboard { margin-top: 30px; padding: 15px; background: #fff; border-radius: 10px; border: 1px solid #e0e0e0; }
    .footer-note { font-size: 13px; color: #70757a; text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'line_prev' not in st.session_state: st.session_state.line_prev = "" 
if 'line_curr' not in st.session_state: st.session_state.line_curr = ""
if 'step_verified' not in st.session_state: st.session_state.step_verified = False
if 'original_solution_set' not in st.session_state: st.session_state.original_solution_set = None
if 'start_time' not in st.session_state: st.session_state.start_time = None
if 'hint_count' not in st.session_state: st.session_state.hint_count = 0
if 'problem_solved' not in st.session_state: st.session_state.problem_solved = False
if 'high_scores' not in st.session_state: st.session_state.high_scores = []
if 'last_processed_buffer' not in st.session_state: st.session_state.last_processed_buffer = None # FIX FOR INFINITE LOOP

# --- HELPERS ---
def clear_all():
    st.session_state.line_prev = ""
    st.session_state.line_curr = ""
    st.session_state.step_verified = False
    st.session_state.original_solution_set = None
    st.session_state.start_time = None
    st.session_state.hint_count = 0
    st.session_state.problem_solved = False
    st.session_state.last_processed_buffer = None # Reset camera memory

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
    text = text.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
    text = text.replace("\\", "").replace("`", "")
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    text = text.replace(" and ", ",").replace(" or ", ",").replace("^", "**").replace("√", "sqrt")
    return text

def safe_parse_latex(text_str):
    try:
        clean = clean_input(text_str)
        if "=" in clean:
            parts = clean.split("=")
            lhs = parse_expr(parts[0], transformations=standard_transformations)
            rhs = parse_expr(parts[1], transformations=standard_transformations)
            return latex(Eq(lhs, rhs))
        return latex(parse_expr(clean, transformations=standard_transformations))
    except: return text_str

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
        if "," in clean:
            if "=" in clean:
                parts = clean.split(",")
                vals = []
                for p in parts:
                    if "=" in p: vals.append(parse_for_logic(p.split("=")[1].strip()))
                    else: vals.append(parse_for_logic(p.strip()))
                return FiniteSet(*vals)
            else:
                return FiniteSet(*[parse_for_logic(i.strip()) for i in clean.split(",") if i.strip()])
        if "±" in clean:
            val = parse_for_logic(clean.split("±")[1].strip())
            return FiniteSet(val, -val)
        expr = parse_for_logic(clean)
        all_symbols = list(expr.free_symbols)
        if not all_symbols: return FiniteSet(expr)
        sol = solve(expr, all_symbols, set=True)
        flat_results = []
        if isinstance(sol, tuple):
            for s in sol[1]:
                if isinstance(s, tuple) and len(s) == 1: flat_results.append(s[0])
                else: flat_results.append(s)
            return FiniteSet(*flat_results)
        return sol
    except: return None

# --- OCR ENGINE ---
def process_image_with_mathpix(image_file, app_id, app_key):
    try:
        image_bytes = image_file.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{image_base64}"
        url = "https://api.mathpix.com/v3/text"
        headers = {"app_id": app_id, "app_key": app_key, "Content-type": "application/json"}
        data = {"src": data_uri, "formats": ["asciimath", "text", "latex_simplified"], "data_options": {"include_asciimath": True}}
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        if 'latex_simplified' in result: return result['latex_simplified']
        elif 'asciimath' in result: return result['asciimath']
        elif 'text' in result: return result['text']
        else: return None
    except Exception as e: return None

def validate_step(line_a, line_b):
    try:
        set_A = get_solution_set(line_a)
        set_B = get_solution_set(line_b)
        if st.session_state.original_solution_set is None: st.session_state.original_solution_set = set_A
        clean_b = clean_input(line_b)
        is_final = False
        if not any(c in clean_b for c in "+*^") and not clean_b.startswith("matrix"): is_final = True
        if set_A == set_B: return True, ("Final" if is_final else "Valid"), ""
        if set_A and set_B and set_A.issubset(set_B):
            if is_final: return True, "Warning", "Wait! You found two potential solutions. Check BOTH in the **original** equation."
            return True, "Valid", ""
        return False, "Invalid", "Values do not match."
    except Exception as e: return False, "Error", str(e)

# --- UI START ---
st.markdown('<div class="main-header"><h1>🦉 THE LOGIC LAB</h1><p>AI Math Step-Checker</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    regents_mode = st.toggle("🏆 Challenge Mode", value=False)
    if regents_mode: st.caption("Timer & Hints Enabled")
    else: st.caption("Study Mode (Relaxed)")
    st.markdown("---")
    parent_mode = st.toggle("👨‍👩‍👧 Parent Mode")
    st.markdown("---")
    
    # STABLE CAMERA LOGIC
    use_camera = st.toggle("📷 Camera Mode")
    if use_camera:
        st.info("Snap a photo of a math problem.")
        img_file = st.camera_input("Scan Math")
        
        if img_file:
            # CHECK: Did we already process this exact image?
            current_buffer = img_file.getvalue()
            if st.session_state.last_processed_buffer != current_buffer:
                # NEW IMAGE DETECTED - PROCESS IT ONCE
                if "mathpix_app_id" in st.secrets:
                    with st.spinner("Analyzing with Mathpix..."):
                        scanned_math = process_image_with_mathpix(img_file, st.secrets["mathpix_app_id"], st.secrets["mathpix_app_key"])
                        if scanned_math:
                            # Clean and Store
                            clean_math = scanned_math.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
                            st.session_state.line_prev = clean_math
                            st.session_state.last_processed_buffer = current_buffer # LOCK IT
                            st.success(f"Math Detected: {clean_math}")
                            st.rerun() # Refresh once to show result
                        else: st.error("Could not read math.")
                else:
                    # DEMO MODE
                    st.warning("⚠️ No API Keys. Simulating scan...")
                    time.sleep(1)
                    st.session_state.line_prev = "4x + 2x = 12"
                    st.session_state.last_processed_buffer = current_buffer
                    st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Leaderboard"): st.session_state.high_scores = []; st.rerun()

col_d1, col_d2, col_d3 = st.columns(3)
if regents_mode:
    with col_d1:
        elapsed = int(time.time() - st.session_state.start_time) if st.session_state.start_time and not st.session_state.problem_solved else 0
        st.markdown(f"<div class='stat-item'>⏱️ {elapsed}s</div>", unsafe_allow_html=True)
    with col_d2: 
        st.markdown(f"<div class='stat-item'>💡 {st.session_state.hint_count}</div>", unsafe_allow_html=True)
with col_d3:
    if st.button("✨ NEW", key="new_btn"): clear_all(); st.rerun()

st.text_input("Previous Line", key="line_prev", label_visibility="collapsed", placeholder="Problem...", help="Previous Line")
if st.session_state.line_prev: st.latex(safe_parse_latex(st.session_state.line_prev))
st.markdown("---")
st.text_input("Current Line", key="line_curr", label_visibility="collapsed", placeholder="Your next step...", help="Current Line")
if st.session_state.line_curr: st.latex(safe_parse_latex(st.session_state.line_curr))

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
                    if regents_mode: st.session_state.high_scores.append({"Time": f"{final_time}s", "Hints": st.session_state.hint_count, "Date": datetime.datetime.now().strftime("%H:%M")})
                    st.balloons(); st.success(f"🏆 Solved in {final_time}s!")
                elif status == "Warning": st.markdown(f"<div class='warning-box'>⚠️ {hint}</div>", unsafe_allow_html=True)
                else: st.markdown("<div class='success-box'>✅ Correct! Keep going.</div>", unsafe_allow_html=True)
            else:
                st.session_state.hint_count += 1
                st.markdown(f"<div class='error-box'>❌ Logic Break. Hint: {hint}</div>", unsafe_allow_html=True)
    with c_next:
        if st.session_state.step_verified: st.button("NEXT STEP ⬇️", on_click=next_step)
else: st.success("✨ Problem Complete! Click NEW to start again.")

if st.session_state.high_scores:
    st.markdown("<div class='leaderboard'><h3>🏆 Session High Scores</h3>", unsafe_allow_html=True)
    st.table(pd.DataFrame(st.session_state.high_scores))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer-note'>Built by Teachers • Powered by Logic</div>", unsafe_allow_html=True)
