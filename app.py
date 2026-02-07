import streamlit as st
import sympy
from sympy import symbols, solve, Eq, latex, simplify, I, pi, E, diff, integrate, limit, oo, Matrix, factorial, Function, Derivative, Integral, ImmutableDenseMatrix
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import datetime
import pandas as pd
import re
import numpy as np
import plotly.graph_objects as go
import requests
import base64
import statistics

# --- CONFIG ---
st.set_page_config(page_title="The Logic Lab", page_icon="🧪", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* 1. GENERAL FONT */
    html, body, [class*="css"] { font-family: 'Segoe UI', Roboto, sans-serif; }
    
    /* 2. BUTTONS - UNIVERSAL SYMBOL FORCE */
    div.stButton > button {
        width: 100%; height: 50px; 
        border-radius: 12px; border: 1px solid #dfe1e5;
        background-color: #f8f9fa !important; 
        transition: all 0.2s;
    }
    div.stButton > button div, 
    div.stButton > button p, 
    div.stButton > button span,
    div.stButton > button {
        color: #000000 !important;
        font-size: 22px !important; 
        font-weight: 800 !important; 
        opacity: 1 !important;
    }
    div.stButton > button:active { background-color: #e2e6ea !important; transform: scale(0.98); }

    /* 3. STOP COLUMNS FROM STACKING ON MOBILE */
    [data-testid="column"] { min-width: 1px !important; }

    /* 4. INPUT BOX STYLING (Gray vs White) */
    [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] div:has(> div > div > input[aria-label="Previous Line"]) input {
        background-color: #f1f3f4 !important; 
        color: #202124 !important; 
        border: 1px solid #dadce0 !important;
    }
    [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] div:has(> div > div > input[aria-label="Current Line"]) input {
        background-color: #ffffff !important; 
        color: #000000 !important;
        border: 2px solid #1a73e8 !important; 
        box-shadow: 0 1px 6px rgba(32,33,36,0.12);
    }

    /* 5. FEEDBACK BOXES */
    .success-box { padding: 15px; background: #d1e7dd; color: #0f5132; border-radius: 10px; text-align: center; border: 1px solid #badbcc; margin-top: 10px; }
    .warning-box { padding: 15px; background: #fff3cd; color: #664d03; border-radius: 10px; text-align: center; border: 1px solid #ffecb5; margin-top: 10px; }
    .error-box { padding: 15px; background: #f8d7da; color: #842029; border-radius: 10px; text-align: center; border: 1px solid #f5c2c7; margin-top: 10px; }
    .hint-box { margin-top: 10px; padding: 10px; background: #e2e3e5; color: #41464b; border-radius: 5px; border-left: 5px solid #0d6efd; font-size: 15px; }
    
    /* 6. FOOTER NOTE STYLE */
    .footer-note { font-size: 13px; color: #70757a; text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# --- STATE ---
if 'line_prev' not in st.session_state: st.session_state.line_prev = "" 
if 'line_curr' not in st.session_state: st.session_state.line_curr = ""
if 'history' not in st.session_state: st.session_state.history = []
if 'keypad_target' not in st.session_state: st.session_state.keypad_target = "Current Line"
if 'step_verified' not in st.session_state: st.session_state.step_verified = False
if 'last_image_bytes' not in st.session_state: st.session_state.last_image_bytes = None
if 'original_solution_set' not in st.session_state: st.session_state.original_solution_set = None

# --- LOGIC ---
def clear_all():
    st.session_state.line_prev = ""
    st.session_state.line_curr = ""
    st.session_state.step_verified = False
    st.session_state.original_solution_set = None

def next_step():
    st.session_state.line_prev = st.session_state.line_curr
    st.session_state.line_curr = ""
    st.session_state.step_verified = False

def add_to_input(text_to_add):
    if st.session_state.keypad_target == "Previous Line": st.session_state.line_prev += text_to_add
    else: st.session_state.line_curr += text_to_add

def clean_input(text):
    text = text.lower()
    text = text.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
    text = text.replace("\\", "").replace("`", "")
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    text = text.replace(" and ", ",").replace(" or ", ",") 
    text = text.replace("^", "**").replace("√", "sqrt")
    text = text.replace("=<", "<=").replace("=>", ">=")
    return text

def parse_for_logic(text):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try:
        logic_dict = {'e': E, 'pi': pi, 'diff': diff, 'integrate': integrate, 'limit': limit, 'oo': oo, 'matrix': ImmutableDenseMatrix, 'factorial': factorial, 'mean': my_mean, 'avg': my_mean, 'median': my_median, 'med': my_median, 'stdev': my_stdev}
        if "<=" in text or ">=" in text or "<" in text or ">" in text: return parse_expr(text, transformations=transformations, evaluate=True, local_dict=logic_dict)
        elif "=" in text:
            parts = text.split("=")
            return Eq(parse_expr(parts[0], transformations=transformations, evaluate=True, local_dict=logic_dict), parse_expr(parts[1], transformations=transformations, evaluate=True, local_dict=logic_dict))
        else: return parse_expr(text, transformations=transformations, evaluate=True, local_dict=logic_dict)
    except: return sympy.sympify(text, evaluate=True)

# Stats Helpers
def sanitize_args(args):
    data = []
    for a in args:
        if isinstance(a, (list, tuple, sympy.FiniteSet)): data.extend(a)
        else: data.append(a)
    return [float(x) for x in data]
def to_sympy_number(val):
    try:
        val = round(val, 8) 
        if val == int(val): return sympy.Integer(int(val))
        return sympy.Float(val)
    except: return sympy.nan
def my_mean(*args): return to_sympy_number(statistics.mean(sanitize_args(args)))
def my_median(*args): return to_sympy_number(statistics.median(sanitize_args(args)))
def my_stdev(*args): return to_sympy_number(statistics.stdev(sanitize_args(args)))

def flatten_set(s):
    if s is None: return set()
    flat_items = []
    for item in s:
        if isinstance(item, (tuple, sympy.Tuple)): flat_items.append(item[0])
        else: flat_items.append(item)
    return sympy.FiniteSet(*flat_items)

def get_solution_set(text_str):
    clean = clean_input(text_str)
    try:
        if "±" in clean:
            parts = clean.split("±")
            val = parse_for_logic(parts[1].strip())
            return flatten_set(sympy.FiniteSet(val, -val))
        elif clean.count("=") == 1 and "," in clean and "(" not in clean:
             rhs = clean.split("=")[1]; items = rhs.split(",")
             vals = [parse_for_logic(i.strip()) for i in items if i.strip()]
             return flatten_set(sympy.FiniteSet(*vals))
        elif "," in clean and "=" not in clean and "(" not in clean:
            items = clean.split(",")
            vals = [parse_for_logic(i.strip()) for i in items if i.strip()]
            return flatten_set(sympy.FiniteSet(*vals))
        equations = []
        if ";" in clean: raw_eqs = clean.split(";")
        elif clean.count("=") > 1 and "," in clean: raw_eqs = clean.split(",")
        else: raw_eqs = [clean]
        for r in raw_eqs:
            if r.strip(): equations.append(parse_for_logic(r))
        all_symbols = set()
        for eq in equations: all_symbols.update(eq.free_symbols)
        solve_vars = list(all_symbols)
        if len(equations) > 1: return flatten_set(solve(equations, solve_vars, set=True)[1])
        else:
            expr = equations[0]
            if expr.is_Number: return flatten_set(sympy.FiniteSet(expr))
            if isinstance(expr, tuple) or isinstance(expr, ImmutableDenseMatrix): return flatten_set(sympy.FiniteSet(expr))
            if isinstance(expr, Eq) or not (expr.is_Relational):
                 if not solve_vars: return flatten_set(sympy.FiniteSet(expr))
                 return flatten_set(solve(expr, solve_vars, set=True)[1])
            else:
                if not solve_vars: return flatten_set(sympy.FiniteSet(expr))
                return flatten_set(solve(expr, solve_vars[0], set=True)[1])
    except: return None

def check_numerical_equivalence(set_a, set_b, tolerance=1e-8):
    try:
        l_a, l_b = list(set_a), list(set_b)
        if len(l_a) != len(l_b): return False
        if l_a and (isinstance(l_a[0], ImmutableDenseMatrix) or isinstance(l_b[0], ImmutableDenseMatrix)):
            if l_a[0] == l_b[0] or l_a[0] == l_b[0].T: return True
            return False
        l_a = [complex(i.evalf()) for i in set_a]; l_b = [complex(i.evalf()) for i in set_b]
        l_a.sort(key=lambda z: (z.real, z.imag)); l_b.sort(key=lambda z: (z.real, z.imag))
        for a, b in zip(l_a, l_b):
            if not np.isclose(a, b, atol=tolerance): return False
        return True
    except: return False

def check_common_errors(text_a, text_b):
    hint = ""
    try:
        clean_a, clean_b = clean_input(text_a), clean_input(text_b)
        if ("<" in clean_a or ">" in clean_a) and ("-" in clean_a):
            if ("<" in clean_a and "<" in clean_b) or (">" in clean_a and ">" in clean_b): hint = "⚠️ Trap Detected: Did you divide by a negative? Remember to flip the sign!"
        if "(" in clean_a and ")" in clean_a and "(" not in clean_b: hint = "⚠️ Check Distribution: Did you multiply the term outside to EVERY term inside?"
    except: pass
    return hint

def validate_step(line_prev_str, line_curr_str):
    debug_info = {}
    try:
        if not line_prev_str or not line_curr_str: return False, "Empty", "", {}
        set_A, set_B = get_solution_set(line_prev_str), get_solution_set(line_curr_str)
        if st.session_state.original_solution_set is None and set_A is not None: st.session_state.original_solution_set = set_A
        is_final_answer = False
        if "=" in line_curr_str:
            if not any(c.isalpha() for c in line_curr_str.split("=")[1].strip()): is_final_answer = True
        elif not any(c in line_curr_str for c in "+-*/^"): is_final_answer = True
        if set_A is None or set_B is None: return False, "Parsing Error", "", debug_info
        if set_A == set_B: 
            return True, ("Final" if is_final_answer else "Valid"), "", debug_info
        if set_A.issubset(set_B) and set_A != set_B:
            if is_final_answer and st.session_state.original_solution_set != set_B:
                return True, "Valid (Warning)", "Wait! You found two potential solutions. Check BOTH in the **original** equation.", debug_info
            return True, "Valid", "", debug_info
        if check_numerical_equivalence(set_A, set_B, tolerance=1e-8): return True, ("Final" if is_final_answer else "Valid"), "", debug_info
        if check_numerical_equivalence(set_A, set_B, tolerance=0.01): return True, "Valid (Rounded)", "Approximation accepted.", debug_info
        trap_hint = check_common_errors(line_prev_str, line_curr_str)
        return False, "Invalid", (trap_hint if trap_hint else "Values do not match."), debug_info
    except: return False, "Error", "", debug_info

def pretty_print(math_str):
    try:
        if not math_str: return ""
        clean = clean_input(math_str)
        return latex(sympy.sympify(clean, evaluate=False))
    except: return math_str

# --- WEB INTERFACE ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1: st.title("🧪 The Logic Lab")
with col_head2: 
    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    if st.button("✨ New", key="top_reset"): clear_all(); st.rerun()

if st.session_state.original_solution_set: st.markdown("<div style='background:#e3f2fd;padding:8px;border-radius:5px;font-size:14px;color:#0d47a1;margin-bottom:15px'>🧠 Memory Active: Tracking Original Equation</div>", unsafe_allow_html=True)

st.caption("PREVIOUS STEP (Problem)")
st.text_input("Line A", key="line_prev", label_visibility="collapsed", placeholder="Enter Step 1...", help="Previous Line")
if st.session_state.line_prev: 
    st.latex(pretty_print(st.session_state.line_prev))

st.markdown("---")
st.caption("CURRENT STEP (Your Work)")
st.text_input("Line B", key="line_curr", label_visibility="collapsed", placeholder="Enter Step 2...", help="Current Line")
if st.session_state.line_curr: st.latex(pretty_print(st.session_state.line_curr))

with st.expander("⌨️ Keypad", expanded=True):
    st.radio("Target:", ["Previous Line", "Current Line"], horizontal=True, key="keypad_target", label_visibility="collapsed")
    t1, t2, t3, t4 = st.tabs(["Alg", "Calc", "Stat", "Mat"])
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.button("x²", on_click=add_to_input, args=("^2",), key="a_sq"); c2.button("√", on_click=add_to_input, args=("sqrt(",), key="a_rt")
        c3.button("(", on_click=add_to_input, args=("(",), key="a_op"); c4.button(")", on_click=add_to_input, args=(")",), key="a_cp")
        c1.button("x", on_click=add_to_input, args=("x",), key="a_x"); c2.button("÷", on_click=add_to_input, args=("/",), key="a_d")
        c3.button("+", on_click=add_to_input, args=("+",), key="a_p"); c4.button("-", on_click=add_to_input, args=("-",), key="a_m")
    with t2:
        c1, c2, c3, c4 = st.columns(4)
        c1.button("d/dx", on_click=add_to_input, args=("diff(",), key="c_df"); c2.button("∫", on_click=add_to_input, args=("integrate(",), key="c_in")
        c3.button("lim", on_click=add_to_input, args=("limit(",), key="c_lm"); c4.button("∞", on_click=add_to_input, args=("oo",), key="c_oo")
        c1.button(",", on_click=add_to_input, args=(", ",), key="c_cm"); c2.button("dx", on_click=add_to_input, args=(", x",), key="c_dx")
    with t3:
        c1, c2, c3, c4 = st.columns(4)
        c1.button("Mean", on_click=add_to_input, args=("mean(",), key="s_mn"); c2.button("Med", on_click=add_to_input, args=("median(",), key="s_md")
        c3.button("Std", on_click=add_to_input, args=("stdev(",), key="s_sd"); c4.button(",", on_click=add_to_input, args=(", ",), key="s_cm")
    with t4:
        c1, c2, c3, c4 = st.columns(4)
        c1.button("Mat", on_click=add_to_input, args=("Matrix([",), key="m_mx"); c2.button("[ ]", on_click=add_to_input, args=("[",), key="m_br")
        c3.button("]", on_click=add_to_input, args=("])",), key="m_cl"); c4.button("!", on_click=add_to_input, args=("factorial(",), key="m_fa")

c_chk, c_nxt = st.columns([1, 1])
with c_chk:
    if st.button("Check Logic", type="primary", key="btn_check"):
        res, status, hint, _ = validate_step(st.session_state.line_prev, st.session_state.line_curr)
        if res:
            st.session_state.step_verified = True
            if status == "Final": st.balloons(); st.markdown("<div class='success-box'><b>🎉 Problem Solved!</b></div>", unsafe_allow_html=True)
            elif "Warning" in status: st.markdown(f"<div class='warning-box'><b>⚠️ Valid, but...</b><br><small>{hint}</small></div>", unsafe_allow_html=True)
            else: st.markdown("<div class='success-box'><b>✅ Good Step. Keep going.</b></div>", unsafe_allow_html=True)
        else:
            st.session_state.step_verified = False
            st.markdown("<div class='error-box'><b>❌ Logic Break</b></div>", unsafe_allow_html=True)
            if hint: st.markdown(f"<div class='hint-box'><b>💡 {hint}</b></div>", unsafe_allow_html=True)
with c_nxt:
    if st.session_state.step_verified: st.button("⬇️ Next Step", on_click=next_step, key="btn_nxt")

# --- FOOTER NOTE ---
st.markdown("""
<div class='footer-note'>
    <b>💡 Classroom Tips:</b><br>
    Click <b>✨ New</b> before starting a different problem to clear the memory.<br>
    Use the <b>⬇️ Next Step</b> button to move your work down and keep going!
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    if st.toggle("📷 Camera"):
        st.warning("Needs API Keys in Secrets")
