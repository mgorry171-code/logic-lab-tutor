import streamlit as st
import sympy
from sympy import symbols, solve, Eq, latex, simplify, I, pi, E, diff, integrate, limit, oo, Matrix, factorial
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

def add_to_input(text_to_add):
    if st.session_state.keypad_target == "Previous Line":
        st.session_state.line_prev += text_to_add
    else:
        st.session_state.line_curr += text_to_add

def clean_input(text):
    text = text.lower()
    # Strip LaTeX wrappers
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace("\\", "")
    text = text.replace("`", "")
    
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    text = text.replace(" and ", ",") 
    text = text.replace("^", "**")
    text = re.sub(r'(?<![a-z])i(?![a-z])', 'I', text) 
    text = text.replace("+/-", "±")
    text = text.replace("√", "sqrt")
    text = text.replace("%", "/100")
    text = text.replace(" of ", "*")
    text = text.replace("=<", "<=").replace("=>", ">=")
    return text

# --- CUSTOM STATS FUNCTIONS FOR PARSER ---
def my_mean(*args):
    # Handle both mean(1,2,3) and mean([1,2,3])
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return sympy.sympify(statistics.mean(args[0]))
    return sympy.sympify(statistics.mean(args))

def my_median(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return sympy.sympify(statistics.median(args[0]))
    return sympy.sympify(statistics.median(args))

def my_stdev(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return sympy.sympify(statistics.stdev(args[0]))
    return sympy.sympify(statistics.stdev(args))

def smart_parse(text, evaluate=True):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try:
        # EXPANDED DICTIONARY FOR CALCULUS & STATS
        local_dict = {
            'e': E, 
            'pi': pi, 
            'diff': diff, 
            'integrate': integrate, 
            'limit': limit, 
            'oo': oo,
            'matrix': Matrix,
            'factorial': factorial,
            'mean': my_mean,
            'median': my_median,
            'stdev': my_stdev
        }
        
        if "<=" in text or ">=" in text or "<" in text or ">" in text:
            return parse_expr(text, transformations=transformations, evaluate=evaluate, local_dict=local_dict)
        elif "=" in text:
            parts = text.split("=")
            lhs = parse_expr(parts[0], transformations=transformations, evaluate=evaluate, local_dict=local_dict)
            rhs = parse_expr(parts[1], transformations=transformations, evaluate=evaluate, local_dict=local_dict)
            return Eq(lhs, rhs)
        else:
            return parse_expr(text, transformations=transformations, evaluate=evaluate, local_dict=local_dict)
    except:
        return sympy.sympify(text, evaluate=evaluate)

def pretty_print(math_str):
    try:
        if not math_str: return ""
        clean_str = clean_input(math_str)
        clean_str = clean_str.replace("±", "±")
        if ";" in clean_str:
             parts = clean_str.split(";")
             latex_parts = [latex(smart_parse(p, evaluate=False)) for p in parts if p.strip()]
             return ", \\quad ".join(latex_parts)
        expr = smart_parse(clean_str, evaluate=False)
        return latex(expr)
    except:
        return math_str

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- LOGIC BRAIN ---
def flatten_set(s):
    if s is None: return set()
    flat_items = []
    for item in s:
        if isinstance(item, (tuple, sympy.Tuple)):
            if len(item) == 1:
                flat_items.append(item[0])
            else:
                flat_items.append(item) 
        else:
            flat_items.append(item)
    return sympy.FiniteSet(*flat_items)

def get_solution_set(text_str):
    x, y = symbols('x y')
    clean = clean_input(text_str)
    try:
        if "±" in clean:
            parts = clean.split("±")
            val = smart_parse(parts[1].strip(), evaluate=True)
            return flatten_set(sympy.FiniteSet(val, -val))
        elif "," in clean and "=" not in clean:
            items = clean.split(",")
            vals = []
            for i in items:
                if i.strip(): vals.append(smart_parse(i.strip(), evaluate=True))
            return flatten_set(sympy.FiniteSet(*vals))

        equations = []
        if ";" in clean:
            raw_eqs = clean.split(";")
            for r in raw_eqs:
                if r.strip(): equations.append(smart_parse(r, evaluate=True))
        elif clean.count("=") > 1 and "," in clean:
            raw_eqs = clean.split(",")
            for r in raw_eqs:
                if r.strip(): equations.append(smart_parse(r, evaluate=True))
        else:
             equations.append(smart_parse(clean, evaluate=True))

        if len(equations) > 1:
            sol = solve(equations, (x, y), set=True)
            return flatten_set(sol[1])
        else:
            expr = equations[0]
            if isinstance(expr, tuple): return flatten_set(sympy.FiniteSet(expr))
            # Handle Matrix Logic
            if isinstance(expr, Matrix):
                 return flatten_set(sympy.FiniteSet(expr))
            
            if isinstance(expr, Eq) or not (expr.is_Relational):
                 if 'y' in str(expr) and 'x' in str(expr): return flatten_set(sympy.FiniteSet(expr))
                 else:
                     if 'x' not in str(expr) and 'y' not in str(expr): return flatten_set(sympy.FiniteSet(expr))
                     sol = solve(expr, x, set=True)
                     return flatten_set(sol[1])
            else:
                solution = reduce_inequalities(expr, x)
                return flatten_set(solution.as_set())
    except Exception as e:
        return None

def check_numerical_equivalence(set_a, set_b):
    try:
        list_a = [complex(i.evalf()) for i in set_a]
        list_b = [complex(i.evalf()) for i in set_b]
        list_a.sort(key=lambda z: (z.real, z.imag))
        list_b.sort(key=lambda z: (z.real, z.imag))
        if len(list_a) != len(list_b): return False
        for a, b in zip(list_a, list_b):
                if not np.isclose(a, b, atol=1e-8): return False
        return True
    except:
        return False

def diagnose_error(set_correct, set_user):
    return "Check your math logic.", ""

def next_step():
    st.session_state.line_prev = st.session_state.line_curr
    st.session_state.line_curr = ""
    st.session_state.step_verified = False

def plot_system_interactive(text_str):
    try:
        x, y = symbols('x y')
        clean = clean_input(text_str)
        equations = []
        if ";" in clean:
            raw_eqs = clean.split(";")
            for r in raw_eqs:
                if r.strip(): equations.append(smart_parse(r, evaluate=True))
        else:
            if clean.count("=") > 1 and "," in clean:
                 raw_eqs = clean.split(",")
                 for r in raw_eqs:
                    if r.strip(): equations.append(smart_parse(r, evaluate=True))
            else:
                 equations.append(smart_parse(clean, evaluate=True))
        
        fig = go.Figure()
        x_vals = np.linspace(-10, 10, 100)
        colors = ['blue', 'orange', 'green']
        i = 0
        table_data_list = [] 
        has_plotted = False
        
        for eq in equations:
            try:
                if eq.has(I): continue
                if 'y' in str(eq):
                    y_expr = solve(eq, y)
                    if y_expr:
                        f_y = sympy.lambdify(x, y_expr[0], "numpy") 
                        y_vals = f_y(x_vals)
                        if np.iscomplexobj(y_vals): y_vals = y_vals.real 
                        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f"Eq {i+1}", line=dict(color=colors[i % 3])))
                        t_x = []
                        t_y = []
                        for val in [-4, -2, 0, 2, 4]:
                            try:
                                res_y = y_expr[0].subs(x, val)
                                if res_y.is_real: 
                                    t_x.append(val)
                                    t_y.append(round(float(res_y), 2))
                            except: pass
                        if t_x:
                            df_table = pd.DataFrame({"x": t_x, "y": t_y})
                            table_data_list.append({"label": f"Equation {i+1}: ${latex(eq)}$", "df": df_table})
                        has_plotted = True
                        i += 1
                elif 'x' in str(eq):
                    x_sol = solve(eq, x)
                    if x_sol:
                        val = float(x_sol[0])
                        fig.add_vline(x=val, line_dash="dash", line_color=colors[i%3], annotation_text=f"x={val}")
                        t_x = [val]*5
                        t_y = [-4, -2, 0, 2, 4]
                        df_table = pd.DataFrame({"x": t_x, "y": t_y})
                        table_data_list.append({"label": f"Equation {i+1}: ${latex(eq)}$", "df": df_table})
                        has_plotted = True
                        i += 1
            except: pass
        if not has_plotted: return None, None
        fig.update_layout(xaxis_title="X Axis", yaxis_title="Y Axis", xaxis=dict(range=[-10, 10], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black'), yaxis=dict(range=[-10, 10], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black'), height=500, showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
        return fig, table_data_list
    except Exception as e:
        return None, None

def validate_step(line_prev_str, line_curr_str):
    debug_info = {}
    try:
        if not line_prev_str or not line_curr_str: return False, "Empty", "", {}
        set_A = get_solution_set(line_prev_str)
        set_B = get_solution_set(line_curr_str)
        debug_info['Raw Set A'] = str(set_A)
        debug_info['Raw Set B'] = str(set_B)
        
        if set_A is None and line_prev_str: return False, "Could not solve Line A", "", debug_info
        if set_B is None: return False, "Could not parse Line B", "", debug_info

        if set_A == set_B: return True, "Valid", "", debug_info
        try:
            list_A = sorted([str(s) for s in set_A])
            list_B = sorted([str(s) for s in set_B])
            if list_A == list_B:
                 return True, "Valid", "", debug_info
        except: pass
        
        if check_numerical_equivalence(set_A, set_B):
             return True, "Valid", "", debug_info

        hint, internal_debug = diagnose_error(set_A, set_B)
        return False, "Invalid", hint, debug_info

    except Exception as e:
        return False, f"Syntax Error: {e}", "", debug_info

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

st.set_page_config(page_title="The Logic Lab v8.2", page_icon="🧪")

# --- CUSTOM CSS ---
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
        st.code("mean(1, 3, 5)")
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

# --- EXPANDED KEYPAD (FIXED ARGS ERROR) ---
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
        b5.button("Mode (soon)", disabled=True) # FIXED
        b6.button("Norm (soon)", disabled=True) # FIXED

    with tab4: # Pre-Calc
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.button("Matrix", on_click=add_to_input, args=("Matrix([",))
        b2.button("[ ]", on_click=add_to_input, args=("[",))
        b3.button("]", on_click=add_to_input, args=("])",))
        b4.button("n!", on_click=add_to_input, args=("factorial(",))
        b5.button("Σ (soon)", disabled=True) # FIXED
        b6.button("∏ (soon)", disabled=True) # FIXED

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
            st.markdown(f"<div class='success-box'><b>✅ Perfect Logic!</b></div>", unsafe_allow_html=True)
        else:
            st.session_state.step_verified = False
            st.error("❌ Logic Break")
            if hint: st.info(f"💡 Hint: {hint}")

with c_next:
    if st.session_state.step_verified:
        st.button("⬇️ Next Step (Move Down)", on_click=next_step)
