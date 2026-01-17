import streamlit as st
import sympy
from sympy import symbols, solve, Eq, latex, simplify, I, pi, E
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import datetime
import pandas as pd
import re
import numpy as np
import plotly.graph_objects as go
import requests
import base64
from PIL import Image
from io import BytesIO

# --- SETUP SESSION STATE ---
if 'line_prev' not in st.session_state:
    st.session_state.line_prev = "x^2 + 4 = 0" 
if 'line_curr' not in st.session_state:
    st.session_state.line_curr = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'keypad_target' not in st.session_state:
    st.session_state.keypad_target = "Current Line"
if 'step_verified' not in st.session_state:
    st.session_state.step_verified = False
# NEW: Track the last processed image to prevent loops
if 'last_image_bytes' not in st.session_state:
    st.session_state.last_image_bytes = None

# --- HELPER FUNCTIONS ---
def add_to_input(text_to_add):
    if st.session_state.keypad_target == "Previous Line":
        st.session_state.line_prev += text_to_add
    else:
        st.session_state.line_curr += text_to_add

def clean_input(text):
    text = text.lower()
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

def smart_parse(text, evaluate=True):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try:
        local_dict = {'e': E, 'pi': pi}
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
        return sympify(text, evaluate=evaluate)

def pretty_print(math_str):
    try:
        clean_str = clean_input(math_str)
        clean_str = clean_str.replace("±", "±")
        if ";" in clean_str:
             parts = clean_str.split(";")
             latex_parts = [latex(smart_parse(p, evaluate=False)) for p in parts if p.strip()]
             return ", \\quad ".join(latex_parts)
        expr = smart_parse(clean_str, evaluate=False)
        return latex(expr)
    except:
        return None

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- LOGIC BRAIN 5.3 ---
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

def check_simplification(text):
    try:
        clean = clean_input(text)
        expr = smart_parse(clean, evaluate=False)
        if isinstance(expr, Eq): rhs = expr.rhs
        else: rhs = expr
        if rhs.is_Number or rhs.is_Symbol: return True
        if rhs.is_Mul and len(rhs.args) == 2 and rhs.args[0] == -1 and rhs.args[1].is_Number: return True
        if rhs.has(I): return True
        return False
    except:
        return True

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
        
        hint, internal_debug = diagnose_error(set_A, set_B)
        return False, "Invalid", hint, debug_info

    except Exception as e:
        return False, f"Syntax Error: {e}", "", debug_info

# --- NEW: MATHPIX INTEGRATION ---
def process_image_with_mathpix(image_file, app_id, app_key):
    try:
        image_bytes = image_file.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{image_base64}"
        url = "https://api.mathpix.com/v3/text"
        headers = {
            "app_id": app_id,
            "app_key": app_key,
            "Content-type": "application/json"
        }
        data = {
            "src": data_uri,
            "formats": ["text", "latex_simplified"],
            "data_options": {"include_asciimath": True}
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        if 'latex_simplified' in result:
            return result['latex_simplified']
        elif 'text' in result:
            return result['text']
        else:
            return None
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return None

# --- WEB INTERFACE ---

st.set_page_config(page_title="The Logic Lab v6.1", page_icon="🧪")
st.title("🧪 The Logic Lab")

with st.sidebar:
    st.header("Settings")
    if st.session_state.history:
        st.write(f"Problems Checked: **{len(st.session_state.history)}**")
        df = pd.DataFrame(st.session_state.history)
        csv = convert_df_to_csv(df)
        st.download_button("📊 Download Session Data", csv, "Math_Session.csv", "text/csv")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    st.markdown("---")
    st.subheader("📷 Camera Settings")
    st.caption("Enter Keys to enable real scanning. Empty keys = Demo Mode.")
    mp_id = st.text_input("Mathpix App ID", type="password")
    mp_key = st.text_input("Mathpix App Key", type="password")
    st.markdown("---")
    parent_mode = st.toggle("👨‍👩‍👧 Parent Mode", value=False)
    st.markdown("---")
    show_debug = st.checkbox("🛠️ Engineer Mode", value=False)

# --- CAMERA INPUT (FIXED NO-LOOP) ---
with st.expander("📷 Scan Handwritten Math", expanded=True):
    cam_col1, cam_col2 = st.columns([1, 3])
    with cam_col1:
        img_file = st.camera_input("Take a photo")
    with cam_col2:
        if img_file is not None:
            # CHECK: Only process if it's a NEW photo
            current_bytes = img_file.getvalue()
            if current_bytes != st.session_state.last_image_bytes:
                st.session_state.last_image_bytes = current_bytes # Remember this photo
                
                st.write("Processing...")
                if mp_id and mp_key:
                    extracted_text = process_image_with_mathpix(img_file, mp_id, mp_key)
                    if extracted_text:
                        st.session_state.line_prev = extracted_text
                        st.success(f"Scanned: {extracted_text}")
                        # No st.rerun() needed - Streamlit updates automatically
                else:
                    st.warning("⚠️ No API Keys found. Running Simulation.")
                    st.session_state.line_prev = "3x^2 + 5x - 2 = 0"
                    st.success("Simulated Scan: 3x^2 + 5x - 2 = 0")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Previous Line")
    st.text_input("Line A", key="line_prev", label_visibility="collapsed")
    if st.session_state.line_prev: 
        st.latex(pretty_print(st.session_state.line_prev))
        
        if parent_mode:
            if st.button("👁️ Reveal Answer for Line A"):
                sol_set = get_solution_set(st.session_state.line_prev)
                if sol_set:
                    st.success("**Answer Key:**")
                    st.latex(latex(sol_set))
                else:
                    st.error("Could not solve this expression.")
        
        if st.checkbox("📈 Visualize Graph"):
            fig, table_list = plot_system_interactive(st.session_state.line_prev)
            if fig:
                tab1, tab2 = st.tabs(["📉 Interactive Graph", "🔢 Table of Values"])
                with tab1:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Hover to see points.")
                with tab2:
                    st.write("Use these T-Charts to plot the lines:")
                    if table_list:
                        if len(table_list) == 2:
                            t1, t2 = st.columns(2)
                            with t1:
                                st.write(table_list[0]["label"])
                                st.dataframe(table_list[0]["df"], hide_index=True)
                            with t2:
                                st.write(table_list[1]["label"])
                                st.dataframe(table_list[1]["df"], hide_index=True)
                        else:
                            for item in table_list:
                                st.write(item["label"])
                                st.dataframe(item["df"], hide_index=True)
            else:
                st.caption("Could not graph this expression.")

with col2:
    st.markdown("### Current Line")
    st.text_input("Line B", key="line_curr", label_visibility="collapsed")
    if st.session_state.line_curr: st.latex(pretty_print(st.session_state.line_curr))

st.markdown("---")

with st.expander("⌨️ Show Math Keypad", expanded=False):
    st.write("Click a button to add it to the **" + st.session_state.keypad_target + "**.")
    st.radio("Target:", ["Previous Line", "Current Line"], horizontal=True, key="keypad_target", label_visibility="collapsed")
    st.write("") 
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.button("x²", on_click=add_to_input, args=("^2",))
    c2.button("√", on_click=add_to_input, args=("sqrt(",))
    c3.button("(", on_click=add_to_input, args=("(",))
    c4.button(")", on_click=add_to_input, args=(")",))
    c5.button(";", on_click=add_to_input, args=("; ",))
    c6.button("÷", on_click=add_to_input, args=("/",))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.button(" < ", on_click=add_to_input, args=("<",))
    c2.button("\>", on_click=add_to_input, args=(">",)) 
    c3.button(" ≤ ", on_click=add_to_input, args=("<=",))
    c4.button(" ≥ ", on_click=add_to_input, args=(">=",))
    c5.button("x", on_click=add_to_input, args=("x",))
    c6.button("y", on_click=add_to_input, args=("y",))
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.button("i", on_click=add_to_input, args=("i",)) 
    c2.button("π", on_click=add_to_input, args=("pi",))
    c3.button("e", on_click=add_to_input, args=("e",))
    c4.button("log", on_click=add_to_input, args=("log(",))
    c5.button("sin", on_click=add_to_input, args=("sin(",))
    c6.button("cos", on_click=add_to_input, args=("cos(",))

st.markdown("---")

c_check, c_next = st.columns([1, 1])
with c_check:
    if st.button("Check Logic", type="primary"):
        line_a = st.session_state.line_prev
        line_b = st.session_state.line_curr
        is_valid, status, hint, debug_data = validate_step(line_a, line_b)
        now = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.history.append({
            "Time": now, "Input A": line_a, "Input B": line_b, "Result": status, "Hint": hint
        })
        if is_valid:
            st.session_state.step_verified = True 
            if status == "Valid":
                st.success("✅ **Perfect Logic!**")
                st.balloons()
            elif status == "Unsimplified":
                st.warning("⚠️ **Correct, but not fully simplified.**")
                st.info("💡 **Hint:** Perform the arithmetic.")
            elif status == "Partial":
                st.warning("⚠️ **Technically Correct, but Incomplete.**")
        else:
            st.session_state.step_verified = False
            st.error("❌ **Logic Break**")
            if hint and hint != "Logic error.":
                st.info(f"💡 **Hint:** {hint}")
        if not is_valid and show_debug:
            st.markdown("---")
            st.write("🛠️ **Debug X-Ray:**")
            st.write(f"**Raw Set A:** `{debug_data.get('Raw Set A')}`")
            st.write(f"**Raw Set B:** `{debug_data.get('Raw Set B')}`")

with c_next:
    if st.session_state.step_verified:
        st.button("⬇️ Next Step (Move Down)", on_click=next_step)

st.markdown("---")
st.markdown(
    """<div style='text-align: center; color: #666;'><small>Built by The Logic Lab 🧪 | © 2026 Step-Checker</small></div>""",
    unsafe_allow_html=True
)
