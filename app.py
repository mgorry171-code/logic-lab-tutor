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

# --- HELPER FUNCTIONS ---
def clear_all():
    st.session_state.line_prev = ""
    st.session_state.line_curr = ""
    st.session_state.step_verified = False

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

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

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
        list_a = list(set_a)
        list_b = list(set_b)
        if len(list_a) != len(list_b): return False
        if list_a and (isinstance(list_a[0], ImmutableDenseMatrix) or isinstance(list_b[0], ImmutableDenseMatrix)):
            mat_a = list_a[0]
            mat_b = list_b[0]
            if mat_a == mat_b: return True
            if mat_a == mat_b.T: return True
            return False

        list_a = [complex(i.evalf()) for i in set_a]
        list_b = [complex(i.evalf()) for i in set_b]
        list_a.sort(key=lambda z: (z.real, z.imag))
        list_b.sort(key=lambda z: (z.real, z.imag))
        for a, b in zip(list_a, list_b):
                if not np.isclose(a, b, atol=tolerance): return False
        return True
    except:
        return False

# --- NEW: REGENTS "TRAP" DETECTOR ---
def check_common_errors(text_a, text_b):
    """Detects specific student errors and returns a custom hint."""
    hint = ""
    try:
        clean_a = clean_input(text_a)
        clean_b = clean_input(text_b)
        
        # 1. Inequality Sign Error (Regents Trap #2)
        if ("<" in clean_a or ">" in clean_a) and ("-" in clean_a):
            # Check if they divided by negative but kept sign same
            if "<" in clean_a and "<" in clean_b:
                hint = "⚠️ Trap Detected: Did you divide by a negative? Remember to flip the sign!"
            elif ">" in clean_a and ">" in clean_b:
                hint = "⚠️ Trap Detected: Did you divide by a negative? Remember to flip the sign!"

        # 2. Distribution Error (Regents Trap #1 & #3)
        # Very basic check: did they lose a term?
        if "(" in clean_a and ")" in clean_a and "(" not in clean_b:
             # This is hard to detect perfectly without parsing, but we can offer a gentle nudge
             hint = "⚠️ Check Distribution: Did you multiply the term outside to EVERY term inside?"

    except:
        pass
    return hint

def validate_step(line_prev_str, line_curr_str):
    debug_info = {}
    try:
        if not line_prev_str or not line_curr_str: return False, "Empty", "", {}
        set_A = get_solution_set(line_prev_str)
        set_B = get_solution_set(line_curr_str)
        
        debug_info['Set A'] = str(set_A)
        debug_info['Set B'] = str(set_B)
        
        if set_A is None and line_prev_str: return False, "Could not solve Line A", "", debug_info
        if set_B is None: return False, "Could not parse Line B", "", debug_info

        # 1. Exact Match
        if set_A == set_B: return True, "Valid", "", debug_info
        try:
            if sorted([str(s) for s in set_A]) == sorted([str(s) for s in set_B]):
                 return True, "Valid", "", debug_info
        except: pass
        
        # 2. Numerical Match
        if check_numerical_equivalence(set_A, set_B, tolerance=1e-8):
             return True, "Valid", "", debug_info
        
        # 3. Rounding Match
        if check_numerical_equivalence(set_A, set_B, tolerance=0.01):
             return True, "Valid (Rounded)", "Approximation accepted.", debug_info

        # 4. IF INVALID -> CHECK FOR REGENTS TRAPS
        trap_hint = check_common_errors(line_prev_str, line_curr_str)
        if trap_hint:
             return False, "Invalid", trap_hint, debug_info

        return False, "Invalid", "Values do not match.", debug_info

    except Exception as e:
        return False, f"Error: {e}", "", debug_info

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
        if 'asciimath' in result: return result['asciimath']
        elif 'text' in result: return result['text']
        elif 'latex_simplified' in result: return result['latex_simplified']
        else: return None
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return None

# --- WEB INTERFACE ---

st.set_page_config(page_title="The Logic Lab v10.0", page_icon="🧪")

st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .stButton>button { width: 100%; border-radius: 5px; }
    .success-box { padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🧪 The Logic Lab")

with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Reset All Inputs", type="primary"):
        clear_all()
        st.rerun()
    
    with st.expander("📚 Demo Cheat Sheet"):
        st.markdown("**Calculus**")
        st.code("diff(x^2, x)")
        st.caption("Answer: 2x")
        st.markdown("**Statistics**")
        st.code("median(1, 3, 5)")
        st.caption("Answer: 3")
        st.markdown("**Pre-Calc (Matrices)**")
        st.code("Matrix([[1,2],[3,4]])")
    
    st.markdown("---")
    
    mp_id = None
    mp_key = None
    if "mathpix_app_id" in st.secrets and "mathpix_app_key" in st.secrets:
        mp_id = st.secrets["mathpix_app_id"]
        mp_key = st.secrets["mathpix_app_key"]
        st.success("✅ Camera Active (Licensed)")
    else:
        st.subheader("📷 Camera Settings")
        mp_id = st.text_input("Mathpix App ID", type="password")
        mp_key = st.text_input("Mathpix App Key", type="password")

    st.markdown("---")
    parent_mode = st.toggle("👨‍👩‍👧 Parent Mode", value=False)
    st.markdown("---")
    
    if st.session_state.history:
        st.write(f"Problems Checked: **{len(st.session_state.history)}**")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# --- CAMERA INPUT ---
with st.expander("📷 Scan Handwritten Math", expanded=False):
    enable_cam = st.checkbox("Enable Camera Stream")
    if enable_cam:
        cam_col1, cam_col2 = st.columns([1, 3])
        with cam_col1:
            img_file = st.camera_input("Take a photo", label_visibility="collapsed")
        with cam_col2:
            if img_file is not None:
                current_bytes = img_file.getvalue()
                if current_bytes != st.session_state.last_image_bytes:
                    st.session_state.last_image_bytes = current_bytes 
                    st.write("Processing...")
                    if mp_id and mp_key:
                        extracted_text = process_image_with_mathpix(img_file, mp_id, mp_key)
                        if extracted_text:
                            st.session_state.line_prev = clean_input(extracted_text)
                            st.rerun()
                    else:
                        st.warning("⚠️ No API Keys found. Running Simulation.")
                        st.session_state.line_prev = "diff(x^2, x)"
                        st.rerun()

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Previous Line")
    st.text_input("Line A", key="line_prev", label_visibility="collapsed", placeholder="e.g. diff(x^2, x)")
    
    if st.session_state.line_prev:
        latex_str = pretty_print(st.session_state.line_prev)
        if latex_str: st.latex(latex_str)
        
        if parent_mode:
            if st.button("👁️ Reveal Answer"):
                sol_set = get_solution_set(st.session_state.line_prev)
                if sol_set:
                    st.success("**Key:**")
                    st.latex(latex(sol_set))
                else:
                    st.error("Could not solve.")
        
        if st.checkbox("📈 Graph"):
            fig, table_list = plot_system_interactive(st.session_state.line_prev)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Current Line")
    st.text_input("Line B", key="line_curr", label_visibility="collapsed", placeholder="Type your next step...")
    if st.session_state.line_curr:
        latex_str = pretty_print(st.session_state.line_curr)
        if latex_str: st.latex(latex_str)

st.markdown("---")

# --- EXPANDED KEYPAD ---
with st.expander("⌨️ Show Keypad", expanded=False):
    st.radio("Target:", ["Previous Line", "Current Line"], horizontal=True, key="keypad_target", label_visibility="collapsed")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Algebra", "Calculus", "Statistics", "Pre-Calc"])
    
    with tab1: # Algebra
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

    with tab2: # Calculus
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.button("d/dx", on_click=add_to_input, args=("diff(",))
        b2.button("∫", on_click=add_to_input, args=("integrate(",))
        b3.button("lim", on_click=add_to_input, args=("limit(",))
        b4.button("∞", on_click=add_to_input, args=("oo",))
        b5.button(",", on_click=add_to_input, args=(", ",), key="calc_comma") 
        b6.button("dx", on_click=add_to_input, args=(", x",))

    with tab3: # Stats
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.button("Mean", on_click=add_to_input, args=("mean(",))
        b2.button("Median", on_click=add_to_input, args=("median(",))
        b3.button("StDev", on_click=add_to_input, args=("stdev(",))
        b4.button(",", on_click=add_to_input, args=(", ",), key="stats_comma") 
        b5.button("Mode (soon)", disabled=True) 
        b6.button("Norm (soon)", disabled=True) 

    with tab4: # Pre-Calc
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.button("Matrix", on_click=add_to_input, args=("Matrix([",))
        b2.button("[ ]", on_click=add_to_input, args=("[",))
        b3.button("]", on_click=add_to_input, args=("])",))
        b4.button("n!", on_click=add_to_input, args=("factorial(",))
        b5.button("Σ (soon)", disabled=True) 
        b6.button("∏ (soon)", disabled=True) 

st.markdown("---")

# --- ACTION AREA ---
c_check, c_next = st.columns([1, 1])
with c_check:
    if st.button("Check Logic", type="primary"):
        line_a = st.session_state.line_prev
        line_b = st.session_state.line_curr
        is_valid, status, hint, debug_data = validate_step(line_a, line_b)
        
        now = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.history.append({"Time": now, "Input A": line_a, "Input B": line_b, "Result": status})
        
        if is_valid:
            st.session_state.step_verified = True 
            st.balloons()
            if status == "Valid (Rounded)":
                st.markdown(f"<div class='success-box'><b>✅ Correct!</b> <small>(Rounded)</small></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='success-box'><b>✅ Perfect Logic!</b></div>", unsafe_allow_html=True)
        else:
            st.session_state.step_verified = False
            st.error("❌ Logic Break")
            # KEEP X-RAY FOR SAFETY
            st.markdown("#### 🛠️ X-Ray Debugger:")
            st.code(f"I calculated Set A as: {debug_data.get('Set A')}")
            st.code(f"I calculated Set B as: {debug_data.get('Set B')}")
            if hint: st.info(f"💡 Hint: {hint}")

with c_next:
    if st.session_state.step_verified:
        st.button("⬇️ Next Step (Move Down)", on_click=next_step)
