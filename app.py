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

# --- SETUP SESSION STATE ---
if 'line_prev' not in st.session_state:
    st.session_state.line_prev = "" 
if 'line_curr' not in st.session_state:
    st.session_state.line_curr = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'keypad_target' not in st.session_state:
    st.session_state.keypad_target = "Current Line"
if 'step_verified' not in st.session_state:
    st.session_state.step_verified = False
if 'last_image_bytes' not in st.session_state:
    st.session_state.last_image_bytes = None
if 'original_solution_set' not in st.session_state:
    st.session_state.original_solution_set = None

# --- HELPER FUNCTIONS ---
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
    if st.session_state.keypad_target == "Previous Line":
        st.session_state.line_prev += text_to_add
    else:
        st.session_state.line_curr += text_to_add

def clean_input(text):
    text = text.lower()
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace("\\", "")
    text = text.replace("`", "")
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    text = text.replace(" and ", ",") 
    text = text.replace(" or ", ",") 
    text = text.replace("^", "**")
    text = re.sub(r'(?<![a-z])i(?![a-z])', 'I', text) 
    text = text.replace("+/-", "±")
    text = text.replace("√", "sqrt")
    text = text.replace("%", "/100")
    text = text.replace(" of ", "*")
    text = text.replace("=<", "<=").replace("=>", ">=")
    return text

# --- CUSTOM STATS FUNCTIONS ---
def sanitize_args(args):
    data = []
    for a in args:
        if isinstance(a, (list, tuple, sympy.FiniteSet)):
            data.extend(a)
        else:
            data.append(a)
    return [float(x) for x in data]

def to_sympy_number(val):
    try:
        val = round(val, 8) 
        if val == int(val):
            return sympy.Integer(int(val))
        return sympy.Float(val)
    except:
        return sympy.nan

def my_mean(*args):
    try:
        data = sanitize_args(args)
        if not data: return sympy.nan
        val = statistics.mean(data)
        return to_sympy_number(val)
    except:
        return sympy.nan

def my_median(*args):
    try:
        data = sanitize_args(args)
        if not data: return sympy.nan
        val = statistics.median(data)
        return to_sympy_number(val)
    except:
        return sympy.nan

def my_stdev(*args):
    try:
        data = sanitize_args(args)
        if len(data) < 2: return sympy.nan
        val = statistics.stdev(data)
        return to_sympy_number(val)
    except:
        return sympy.nan

# --- PARSING ENGINES ---
def parse_for_display(text):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try:
        display_dict = {
            'e': E, 'pi': pi, 'oo': oo,
            'diff': Derivative,      
            'integrate': Integral,   
            'limit': limit,
            'matrix': Function('Matrix'), 
            'factorial': factorial,
            'mean': Function('Mean'),       
            'avg': Function('Mean'),         
            'median': Function('Median'),   
            'med': Function('Median'),       
            'stdev': Function('StDev')      
        }
        clean_text = clean_input(text)
        if "=" in clean_text:
            parts = clean_text.split("=")
            lhs = parse_expr(parts[0], transformations=transformations, evaluate=False, local_dict=display_dict)
            rhs = parse_expr(parts[1], transformations=transformations, evaluate=False, local_dict=display_dict)
            return Eq(lhs, rhs)
        else:
            return parse_expr(clean_text, transformations=transformations, evaluate=False, local_dict=display_dict)
    except:
        return text

def parse_for_logic(text):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try:
        logic_dict = {
            'e': E, 'pi': pi, 'diff': diff, 'integrate': integrate, 'limit': limit, 'oo': oo,
            'matrix': ImmutableDenseMatrix, 
            'factorial': factorial,
            'mean': my_mean, 
            'avg': my_mean,        
            'median': my_median, 
            'med': my_median,      
            'stdev': my_stdev
        }
        if "<=" in text or ">=" in text or "<" in text or ">" in text:
            return parse_expr(text, transformations=transformations, evaluate=True, local_dict=logic_dict)
        elif "=" in text:
            parts = text.split("=")
            lhs = parse_expr(parts[0], transformations=transformations, evaluate=True, local_dict=logic_dict)
            rhs = parse_expr(parts[1], transformations=transformations, evaluate=True, local_dict=logic_dict)
            return Eq(lhs, rhs)
        else:
            return parse_expr(text, transformations=transformations, evaluate=True, local_dict=logic_dict)
    except:
        return sympy.sympify(text, evaluate=True)

def pretty_print(math_str):
    try:
        if not math_str: return ""
        expr = parse_for_display(math_str)
        if isinstance(expr, str): return expr
        return latex(expr)
    except:
        return math_str

# --- LOGIC CORE ---
def flatten_set(s):
    if s is None: return set()
    flat_items = []
    for item in s:
        if isinstance(item, (tuple, sympy.Tuple)):
            if len(item) == 1: flat_items.append(item[0])
            else: flat_items.append(item) 
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
             rhs = clean.split("=")[1]
             items = rhs.split(",")
             vals = []
             for i in items:
                if i.strip(): vals.append(parse_for_logic(i.strip()))
             return flatten_set(sympy.FiniteSet(*vals))
        elif "," in clean and "=" not in clean and "(" not in clean:
            items = clean.split(",")
            vals = []
            for i in items:
                if i.strip(): vals.append(parse_for_logic(i.strip()))
            return flatten_set(sympy.FiniteSet(*vals))

        equations = []
        if ";" in clean:
            raw_eqs = clean.split(";")
            for r in raw_eqs:
                if r.strip(): equations.append(parse_for_logic(r))
        elif clean.count("=") > 1 and "," in clean:
            raw_eqs = clean.split(",")
            for r in raw_eqs:
                if r.strip(): equations.append(parse_for_logic(r))
        else:
             equations.append(parse_for_logic(clean))

        all_symbols = set()
        for eq in equations:
            all_symbols.update(eq.free_symbols)
        solve_vars = list(all_symbols)

        if len(equations) > 1:
            sol = solve(equations, solve_vars, set=True)
            return flatten_set(sol[1])
        else:
            expr = equations[0]
            if expr.is_Number: return flatten_set(sympy.FiniteSet(expr))
            if isinstance(expr, tuple): return flatten_set(sympy.FiniteSet(expr))
            if isinstance(expr, ImmutableDenseMatrix): return flatten_set(sympy.FiniteSet(expr))
            if isinstance(expr, Eq) or not (expr.is_Relational):
                 if not solve_vars: return flatten_set(sympy.FiniteSet(expr))
                 sol = solve(expr, solve_vars, set=True)
                 return flatten_set(sol[1])
            else:
                try:
                    if not solve_vars: return flatten_set(sympy.FiniteSet(expr))
                    sol = solve(expr, solve_vars[0], set=True)
                    return flatten_set(sol[1])
                except:
                    return flatten_set(sympy.FiniteSet(expr))
    except Exception as e:
        return None

def check_numerical_equivalence(set_a, set_b, tolerance=1e-8):
    try:
        list_a, list_b = list(set_a), list(set_b)
        if len(list_a) != len(list_b): return False
        if list_a and (isinstance(list_a[0], ImmutableDenseMatrix) or isinstance(list_b[0], ImmutableDenseMatrix)):
            mat_a, mat_b = list_a[0], list_b[0]
            if mat_a == mat_b or mat_a == mat_b.T: return True
            return False
        l_a = [complex(i.evalf()) for i in set_a]
        l_b = [complex(i.evalf()) for i in set_b]
        l_a.sort(key=lambda z: (z.real, z.imag))
        l_b.sort(key=lambda z: (z.real, z.imag))
        for a, b in zip(l_a, l_b):
            if not np.isclose(a, b, atol=tolerance): return False
        return True
    except:
        return False

# --- HINT LOGIC ---
def check_common_errors(text_a, text_b):
    hint = ""
    try:
        clean_a, clean_b = clean_input(text_a), clean_input(text_b)
        if ("<" in clean_a or ">" in clean_a) and ("-" in clean_a):
            if ("<" in clean_a and "<" in clean_b) or (">" in clean_a and ">" in clean_b):
                hint = "⚠️ Trap Detected: Did you divide by a negative? Remember to flip the sign!"
        if "(" in clean_a and ")" in clean_a and "(" not in clean_b:
             hint = "⚠️ Check Distribution: Did you multiply the term outside to EVERY term inside?"
    except:
        pass
    return hint

def validate_step(line_prev_str, line_curr_str):
    debug_info = {}
    try:
        if not line_prev_str or not line_curr_str: return False, "Empty", "", {}
        set_A, set_B = get_solution_set(line_prev_str), get_solution_set(line_curr_str)
        if st.session_state.original_solution_set is None and set_A is not None:
            st.session_state.original_solution_set = set_A
        
        is_final_answer = False
        if "=" in line_curr_str:
            rhs = line_curr_str.split("=")[1].strip()
            if not any(c.isalpha() for c in rhs): is_final_answer = True
        elif not any(c in line_curr_str for c in "+-*/^"): is_final_answer = True

        if set_A is None or set_B is None: return False, "Parsing Error", "", debug_info

        if set_A == set_B: 
            return True, ("Final" if is_final_answer else "Valid"), "", debug_info
        
        if set_A.issubset(set_B) and set_A != set_B:
            if is_final_answer and st.session_state.original_solution_set != set_B:
                return True, "Valid (Warning)", "Wait! You found two potential solutions. Check BOTH in the **original** equation.", debug_info
            return True, "Valid", "", debug_info

        if check_numerical_equivalence(set_A, set_B, tolerance=1e-8): 
            return True, ("Final" if is_final_answer else "Valid"), "", debug_info
        if check_numerical_equivalence(set_A, set_B, tolerance=0.01): 
            return True, "Valid (Rounded)", "Approximation accepted.", debug_info

        trap_hint = check_common_errors(line_prev_str, line_curr_str)
        return False, "Invalid", (trap_hint if trap_hint else "Values do not match."), debug_info
    except:
        return False, "Error", "", debug_info

# --- UI ---
st.set_page_config(page_title="The Logic Lab v12.3", page_icon="🧪")
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    .success-box { padding: 15px; background-color: #d4edda; color: #155724; border-radius: 10px; text-align: center; font-size: 18px; margin-bottom: 10px; }
    .warning-box { padding: 15px; background-color: #fff3cd; color: #856404; border-radius: 10px; text-align: center; font-size: 18px; margin-bottom: 10px; }
    .error-box { padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 10px; text-align: center; font-size: 18px; margin-bottom: 10px; }
    .hint-box { padding: 10px; background-color: #e2e3e5; color: #383d41; border-radius: 5px; border-left: 5px solid #007bff; }
</style>
""", unsafe_allow_html=True)

col_title, col_reset = st.columns([3, 1])
with col_title: st.title("🧪 The Logic Lab")
with col_reset:
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    if st.button("✨ New Problem", type="primary"): clear_all(); st.rerun()

if st.session_state.original_solution_set is not None:
    st.info("🧠 Memory Active: Tracking Original Equation for Extraneous Solutions.")

with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Reset All"): clear_all(); st.rerun()
    st.markdown("---")
    parent_mode = st.toggle("👨‍👩‍👧 Parent Mode", value=False)

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Previous Line")
    st.text_input("Line A", key="line_prev", label_visibility="collapsed")
    if st.session_state.line_prev: st.latex(pretty_print(st.session_state.line_prev))
with col2:
    st.markdown("### Current Line")
    st.text_input("Line B", key="line_curr", label_visibility="collapsed")
    if st.session_state.line_curr: st.latex(pretty_print(st.session_state.line_curr))

st.markdown("---")
with st.expander("⌨️ Show Keypad", expanded=False):
    st.radio("Target:", ["Previous Line", "Current Line"], horizontal=True, key="keypad_target", label_visibility="collapsed")
    t1, t2, t3, t4 = st.tabs(["Algebra", "Calculus", "Statistics", "Pre-Calc"])
    with t1:
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.button("x²", on_click=add_to_input, args=("^2",))
        b2.button("√", on_click=add_to_input, args=("sqrt(",))
        b3.button("(", on_click=add_to_input, args=("(",))
        b4.button(")", on_click=add_to_input, args=(")",))
        b5.button("x", on_click=add_to_input, args=("x",))
        b6.button("÷", on_click=add_to_input, args=("/",))
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.button("i", on_click=add_to_input, args=("i",))
        b2.button("π", on_click=add_to_input, args=("pi",))
        b3.button("e", on_click=add_to_input, args=("e",))
        b4.button("log", on_click=add_to_input, args=("log(",))
        b5.button("sin", on_click=add_to_input, args=("sin(",))
        b6.button("cos", on_click=add_to_input, args=("cos(",))
    with t2:
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.button("d/dx", on_click=add_to_input, args=("diff(",))
        b2.button("∫", on_click=add_to_input, args=("integrate(",))
        b3.button("lim", on_click=add_to_input, args=("limit(",))
        b4.button("∞", on_click=add_to_input, args=("oo",))
        b5.button(",", on_click=add_to_input, args=(", ",), key="c_c")
        b6.button("dx", on_click=add_to_input, args=(", x",))
    with t3:
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.button("Mean", on_click=add_to_input, args=("mean(",))
        b2.button("Med", on_click=add_to_input, args=("median(",))
        b3.button("StDev", on_click=add_to_input, args=("stdev(",))
        b4.button(",", on_click=add_to_input, args=(", ",), key="s_c")
    with t4:
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.button("Matrix", on_click=add_to_input, args=("Matrix([",))
        b2.button("[ ]", on_click=add_to_input, args=("[",))
        b3.button("]", on_click=add_to_input, args=("])",))
        b4.button("n!", on_click=add_to_input, args=("factorial(",))

st.markdown("---")
c_check, c_next = st.columns([1, 1])
with c_check:
    if st.button("Check Logic", type="primary"):
        res, status, hint, _ = validate_step(st.session_state.line_prev, st.session_state.line_curr)
        if res:
            st.session_state.step_verified = True
            if status == "Final": st.balloons(); st.markdown("<div class='success-box'><b>✅ Victory! Problem Solved.</b></div>", unsafe_allow_html=True)
            elif "Warning" in status: st.markdown(f"<div class='warning-box'><b>⚠️ Valid Step, but...</b><br><small>{hint}</small></div>", unsafe_allow_html=True)
            else: st.markdown("<div class='success-box'><b>✅ Correct Step! Keep going.</b></div>", unsafe_allow_html=True)
        else:
            st.session_state.step_verified = False
            st.markdown("<div class='error-box'><b>❌ Logic Break</b></div>", unsafe_allow_html=True)
            if hint: st.markdown(f"<div class='hint-box'><b>💡 Hint:</b> {hint}</div>", unsafe_allow_html=True)
with c_next:
    if st.session_state.step_verified: st.button("⬇️ Next Step (Move Down)", on_click=next_step)
