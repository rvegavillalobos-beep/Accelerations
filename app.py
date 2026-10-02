import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import io

# ---------------------------------------------------------
# CONFIGURACIÓN Y ESTILOS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Measurement Processor",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #0066cc;
        margin-bottom: 10px;
    }
    .stApp {
        background-color: #fafbfe;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CARGA Y PREPROCESAMIENTO DE DATOS
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data(file_bytes_or_path):
    if isinstance(file_bytes_or_path, str):
        df = pd.read_csv(file_bytes_or_path, sep="\t")
    else:
        content = file_bytes_or_path.getvalue().decode("utf-8")
        sep = "\t" if "\t" in content[:1000] else ","
        df = pd.read_csv(io.StringIO(content), sep=sep)

    df.columns = df.columns.str.strip()

    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], errors='coerce')
        df['elapsed_sec'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()
    else:
        df['elapsed_sec'] = np.arange(len(df)) * 0.05
        df['time'] = df['elapsed_sec'].astype(str) + " s"

    acc_cols = [c for c in df.columns if 'Acc' in c and '(' in c]
    if len(acc_cols) >= 3:
        df['Acc_Mag'] = np.sqrt(df[acc_cols[0]]**2 + df[acc_cols[1]]**2 + df[acc_cols[2]]**2)
    
    gyro_cols = [c for c in df.columns if 'As' in c and '(' in c]
    if len(gyro_cols) >= 3:
        df['Gyro_Mag'] = np.sqrt(df[gyro_cols[0]]**2 + df[gyro_cols[1]]**2 + df[gyro_cols[2]]**2)

    return df

# ---------------------------------------------------------
# SIDEBAR (CONTROLES Y CONFIGURACIÓN)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration & Filters")

uploaded_file = st.sidebar.file_uploader(
    "Upload Data File (.txt, .csv, .tsv)", 
    type=["txt", "csv", "tsv"]
)

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)
else:
    try:
        df = load_and_preprocess_data("20260910103112.txt")
        st.sidebar.info("📂 Using default trace file (`20260910103112.txt`).")
    except Exception:
        st.warning("Please upload a trace file to begin.")
        st.stop()

target_group = st.sidebar.radio(
    "Variable Group to Analyze:",
    ["Acceleration (g)", "Angular Velocity (°/s)", "Inclination Angles (°)", "Magnetometer (uT)"]
)

if target_group == "Acceleration (g)":
    selected_axes = [c for c in df.columns if 'Acc' in c]
elif target_group == "Angular Velocity (°/s)":
    selected_axes = [c for c in df.columns if 'As' in c or 'Gyro' in c]
elif target_group == "Inclination Angles (°)":
    selected_axes = [c for c in df.columns if 'Angle' in c]
else:
    selected_axes = [c for c in df.columns if 'H' in c and 'uT' in c]

st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Hide / Show Axes")
default_axes = [c for c in selected_axes if 'mag' not in c.lower()]

visible_axes = st.sidebar.multiselect(
    "Visible axes on plot:",
    options=selected_axes,
    default=default_axes
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Signal Filtering")
apply_smoothing = st.sidebar.checkbox("Apply Moving Average (Smoothing)", value=False)
window_size = st.sidebar.slider("Smoothing Window (samples):", 3, 51, 9, step=2) if apply_smoothing else 1

remove_gravity = st.sidebar.checkbox("Remove Gravity from Z (Dynamic Acceleration)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🧲 Friction & Slip Parameters")
enable_physics_model = st.sidebar.checkbox("Enable Dynamic Friction Model", value=True)
mu_s = st.sidebar.number_input("Static Friction Coefficient (µs):", value=0.28, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🔨 Jerk Detection")
enable_jerk_detection = st.sidebar.checkbox("Detect High Jerk Events", value=True)
jerk_threshold = st.sidebar.number_input("Jerk Threshold (g/s):", value=3.0, step=0.5) if enable_jerk_detection else None

st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Conventional Thresholds")
enable_upper = st.sidebar.checkbox("Enable Upper Threshold", value=True)
upper_thresh = st.sidebar.number_input("Upper Threshold (|g|):", value=0.16, step=0.01) if enable_upper else None

enable_lower = st.sidebar.checkbox("Enable Lower Threshold", value=False)
lower_thresh = st.sidebar.number_input("Lower Threshold (|g|):", value=-0.16, step=0.01) if enable_lower else None

min_distance_sec = st.sidebar.slider("Minimum Event Separation (s):", 0.1, 10.0, 1.0, 0.1)

# ---------------------------------------------------------
# CÁLCULOS FÍSICOS Y JERK
# ---------------------------------------------------------
plot_df = df.copy()

if apply_smoothing:
    for col in selected_axes:
        plot_df[col] = plot_df[col].rolling(window=window_size, center=True).mean().bfill().ffill()

acc_x_col = next((c for c in plot_df.columns if 'acc' in c.lower() and 'x' in c.lower()), None)
acc_y_col = next((c for c in plot_df.columns if 'acc' in c.lower() and 'y' in c.lower()), None)
acc_z_col = next((c for c in plot_df.columns if 'acc' in c.lower() and 'z' in c.lower()), None)
has_3d_acc = all([acc_x_col, acc_y_col, acc_z_col])

if remove_gravity and acc_z_col:
    plot_df[acc_z_col] = plot_df[acc_z_col] - plot_df[acc_z_col].mean()

dt_sample = plot_df['elapsed_sec'].diff().median()
if pd.isna(dt_sample) or dt_sample <= 0:
    dt_sample = 0.05
fs = 1.0 / dt_sample if dt_sample > 0 else 0.0

if has_3d_acc:
    plot_df['Acc_Horiz_XY'] = np.sqrt(plot_df[acc_x_col]**2 + plot_df[acc_y_col]**2)
    # Cálculo de Jerk en g/s
    plot_df['Jerk_XY'] = plot_df['Acc_Horiz_XY'].diff().abs().fillna(0) / dt_sample

if enable_physics_model and has_3d_acc:
    acc_z_vals = plot_df[acc_z_col].values
    mean_z = np.mean(acc_z_vals)
    normal_g = acc_z_vals if mean_z > 0.5 else (1.0 + acc_z_vals)
    normal_g = np.maximum(0.01, normal_g)
    plot_df['Normal_Force_g'] = normal_g

    plot_df['Slip_Risk_Ratio'] = plot_df['Acc_Horiz_XY'] / plot_df['Normal_Force_g']
    plot_df['Acc_Max_Allowed'] = mu_s * plot_df['Normal_Force_g']

    mu_k = mu_s * 0.9
    acc_net_g = np.maximum(0.0, plot_df['Acc_Horiz_XY'] - (mu_k * plot_df['Normal_Force_g']))
    plot_df['Acc_Net_m_s2'] = acc_net_g * 9.81

# ---------------------------------------------------------
# DETECCIÓN DE EVENTOS MULTI-CRITERIO
# ---------------------------------------------------------
def detect_comprehensive_events(data_df, channels, upper=None, lower=None, min_dist_s=1.0, check_slip=False, mu_stat=0.28, jerk_lim=None):
    dt = data_df['elapsed_sec'].diff().median() or 0.05
    dist_samples = int(max(1, min_dist_s / dt))
    events = []
    
    # 1. Umbrales por eje
    for col in channels:
        if col not in data_df.columns:
            continue
        if upper is not None:
            peaks, _ = find_peaks(data_df[col].values, height=upper, distance=dist_samples)
            for p in peaks:
                events.append({
                    'Index': p,
                    'Timestamp (ISO)': data_df['time'].iloc[p],
                    'Elapsed Time (s)': round(data_df['elapsed_sec'].iloc[p], 3),
                    'Axis / Criterion': col,
                    'Event Type': 'Upper Exceeded ↑',
                    'Measured Value': round(data_df[col].iloc[p], 4),
                    'Set Limit': upper,
                    'Jerk (g/s)': round(data_df['Jerk_XY'].iloc[p], 2) if 'Jerk_XY' in data_df.columns else 0.0,
                    'Duration (ms)': "-",
                    'Est. Displacement (mm)': "-",
                    'Root Cause Diagnosis': "Threshold Breach",
                    'Estimated Effect': "Axis acceleration limit exceeded"
                })

    # 2. Eventos específicos de Jerk
    if jerk_lim is not None and 'Jerk_XY' in data_df.columns:
        jerk_peaks, _ = find_peaks(data_df['Jerk_XY'].values, height=jerk_lim, distance=dist_samples)
        for p in jerk_peaks:
            events.append({
                'Index': p,
                'Timestamp (ISO)': data_df['time'].iloc[p],
                'Elapsed Time (s)': round(data_df['elapsed_sec'].iloc[p], 3),
                'Axis / Criterion': '⚡ Jerk (XY)',
                'Event Type': f'Jerk >= {jerk_lim} g/s',
                'Measured Value': round(data_df['Jerk_XY'].iloc[p], 2),
                'Set Limit': jerk_lim,
                'Jerk (g/s)': round(data_df['Jerk_XY'].iloc[p], 2),
                'Duration (ms)': "-",
                'Est. Displacement (mm)': "-",
                'Root Cause Diagnosis': "💥 Mechanical Impact / Sudden Jerk",
                'Estimated Effect': "High instantaneous force gradient (Potential slip initiation)"
            })

    # 3. Eventos de deslizamiento físico
    if check_slip and 'Slip_Risk_Ratio' in data_df.columns:
        slip_peaks, _ = find_peaks(data_df['Slip_Risk_Ratio'].values, height=mu_stat, distance=dist_samples)
        for p in slip_peaks:
            start_p = p
            while start_p > 0 and data_df['Slip_Risk_Ratio'].iloc[start_p] >= mu_stat:
                start_p -= 1
            end_p = p
            while end_p < len(data_df) - 1 and data_df['Slip_Risk_Ratio'].iloc[end_p] >= mu_stat:
                end_p += 1
                
            duration_sec = (end_p - start_p) * dt
            segment_acc = data_df['Acc_Net_m_s2'].iloc[start_p:end_p+1]
            avg_acc_net = segment_acc.mean() if len(segment_acc) > 0 else 0.0
            disp_mm = 0.5 * avg_acc_net * (duration_sec ** 2) * 1000
            
            z_val = data_df['Normal_Force_g'].iloc[p]
            cause = "⚠️ Load Loss (Low Z / Bounce)" if z_val < 0.85 else "💥 Horizontal Impact (XY / Braking)"
            
            if disp_mm < 0.5:
                efect = "🟢 Micro-vibration"
            elif disp_mm < 5.0:
                efect = "🟡 Minor Displacement (< 5 mm)"
            else:
                efect = "🔴 CRITICAL DISPLACEMENT (> 5 mm)"

            events.append({
                'Index': p,
                'Timestamp (ISO)': data_df['time'].iloc[p],
                'Elapsed Time (s)': round(data_df['elapsed_sec'].iloc[p], 3),
                'Axis / Criterion': '🚨 Slip Risk R(t)',
                'Event Type': f'R >= {mu_stat}',
                'Measured Value': round(data_df['Slip_Risk_Ratio'].iloc[p], 4),
                'Set Limit': mu_stat,
                'Jerk (g/s)': round(data_df['Jerk_XY'].iloc[p], 2),
                'Duration (ms)': round(duration_sec * 1000, 1),
                'Est. Displacement (mm)': round(disp_mm, 2),
                'Root Cause Diagnosis': cause,
                'Estimated Effect': efect
            })
            
    res_df = pd.DataFrame(events)
    if not res_df.empty:
        res_df = res_df.sort_values(by=['Elapsed Time (s)', 'Axis / Criterion']).reset_index(drop=True)
    return res_df

events_df = detect_comprehensive_events(
    plot_df, 
    channels=visible_axes, 
    upper=upper_thresh, 
    lower=lower_thresh, 
    min_dist_s=min_distance_sec,
    check_slip=(enable_physics_model and has_3d_acc),
    mu_stat=mu_s,
    jerk_lim=jerk_threshold
)

# ---------------------------------------------------------
# INTERFAZ PRINCIPAL
# ---------------------------------------------------------
st.title("📈 Measurement Processor")
st.caption("Friction evaluation, Jerk impact analysis, displacement estimation, and root-cause diagnostics.")

tab_plot, tab_events = st.tabs([
    "📊 Interactive Chart", 
    "🚨 Event & Displacement Log"
])

# Obtener evento seleccionado desde la tabla
selected_event = None
if 'selected_event_idx' in st.session_state and not events_df.empty:
    idx = st.session_state['selected_event_idx']
    if idx < len(events_df):
        selected_event = events_df.iloc[idx]

with tab_plot:
    st.subheader(f"Signal Visualization: {target_group}")
    
    # Controles adicionales del gráfico
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        auto_zoom = st.checkbox("🔍 Auto-zoom on selected event (±2s window)", value=True)
    with col_c2:
        show_jerk_trace = st.checkbox("Show Jerk Signal (Jerk_XY) on Plot", value=False)

    if selected_event is not None:
        st.info(
            f"📍 **Selected Event:** **{selected_event['Axis / Criterion']}** at t = **{selected_event['Elapsed Time (s)']} s** | "
            f"Value: **{selected_event['Measured Value']}** | Jerk: **{selected_event['Jerk (g/s)']} g/s** | "
            f"Diagnosis: **{selected_event['Root Cause Diagnosis']}**"
        )
    
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, axis_col in enumerate(visible_axes):
        opacity = 1.0
        line_width = 1.5
        
        if selected_event is not None:
            if axis_col == selected_event['Axis / Criterion']:
                line_width = 2.8
            else:
                opacity = 0.35

        fig.add_trace(go.Scatter(
            x=plot_df['elapsed_sec'],
            y=plot_df[axis_col],
            mode='lines',
            name=axis_col,
            line=dict(width=line_width, color=colors[idx % len(colors)]),
            opacity=opacity,
            hovertext=plot_df['time'],
            hovertemplate='<b>Axis:</b> ' + axis_col + '<br><b>Time:</b> %{x:.2f} s<br><b>Value:</b> %{y:.4f}<extra></extra>'
        ))

    # Trazado opcional de Jerk en la gráfica
    if show_jerk_trace and 'Jerk_XY' in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df['elapsed_sec'],
            y=plot_df['Jerk_XY'],
            mode='lines',
            name='Jerk XY (g/s)',
            line=dict(width=1.5, color='purple', dash='dot'),
            hovertemplate='<b>Jerk:</b> %{y:.2f} g/s<extra></extra>'
        ))

    if enable_physics_model and has_3d_acc and target_group == "Acceleration (g)":
        show_vector_xy = st.checkbox("Show Resultant Horizontal Acceleration (Acc_Horiz_XY)", value=True)
        show_allowed_limit = st.checkbox("Show Dynamic Friction Limit (Acc_Max_Allowed)", value=True)
        
        if show_vector_xy:
            fig.add_trace(go.Scatter(
                x=plot_df['elapsed_sec'],
                y=plot_df['Acc_Horiz_XY'],
                mode='lines',
                name='Resultant Vector XY (g)',
                line=dict(width=2, color='black', dash='solid'),
                hovertemplate='<b>Vector XY:</b> %{y:.4f} g<extra></extra>'
            ))
            
        if show_allowed_limit:
            fig.add_trace(go.Scatter(
                x=plot_df['elapsed_sec'],
                y=plot_df['Acc_Max_Allowed'],
                mode='lines',
                name='Dynamic Friction Limit (µs * N)',
                line=dict(width=1.5, color='red', dash='dash'),
                hovertemplate='<b>Friction Limit:</b> %{y:.4f} g<extra></extra>'
            ))

    if upper_thresh is not None:
        fig.add_hline(y=upper_thresh, line_dash="dash", line_color="crimson", annotation_text=f"+Threshold ({upper_thresh})")
    if lower_thresh is not None:
        fig.add_hline(y=lower_thresh, line_dash="dash", line_color="royalblue", annotation_text=f"-Threshold ({lower_thresh})")

    # Marcar todos los eventos detectados
    if not events_df.empty:
        fig.add_trace(go.Scatter(
            x=events_df['Elapsed Time (s)'],
            y=events_df['Measured Value'],
            mode='markers',
            name='Detected Events',
            marker=dict(symbol='x', size=8, color='red', line=dict(width=1.5)),
            hovertext=events_df['Root Cause Diagnosis'],
            hovertemplate='<b>DETECTED EVENT</b><br><b>Criterion:</b> %{hovertext}<br><b>Time:</b> %{x:.2f} s<br><b>Value:</b> %{y:.4f}<extra></extra>'
        ))

    # Resaltado y Auto-Zoom del evento seleccionado
    if selected_event is not None:
        selected_time = selected_event['Elapsed Time (s)']
        selected_val = selected_event['Measured Value'] if isinstance(selected_event['Measured Value'], (int, float)) else 0.0

        fig.add_vline(x=selected_time, line_width=2.5, line_dash="dash", line_color="gold")
        fig.add_trace(go.Scatter(
            x=[selected_time],
            y=[selected_val],
            mode='markers',
            name='Selected Event',
            marker=dict(symbol='star', size=16, color='yellow', line=dict(width=2, color='black')),
            hoverinfo='skip'
        ))

        # Enfoque automático del rango X al evento seleccionado (±2 segundos)
        if auto_zoom:
            fig.update_xaxes(range=[max(0, selected_time - 2.0), selected_time + 2.0])

    fig.update_layout(
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title="Amplitude",
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_events:
    st.subheader("🚨 Event Diagnostics, Jerk & Real Displacement")
    
    if events_df.empty:
        st.info("No events violating thresholds, Jerk limits, or friction conditions were detected.")
    else:
        st.write(f"Recorded **{len(events_df)}** events evaluated by the physical model:")
        st.caption("👈 **Selecciona una fila de la tabla** para enfocar y hacer Zoom automático sobre ese evento en la gráfica.")
        
        event_selection = st.dataframe(
            events_df[['Elapsed Time (s)', 'Axis / Criterion', 'Measured Value', 'Jerk (g/s)', 'Duration (ms)', 'Est. Displacement (mm)', 'Root Cause Diagnosis', 'Estimated Effect']],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="threshold_table"
        )

        selected_rows = event_selection.get("selection", {}).get("rows", [])
        if selected_rows:
            st.session_state['selected_event_idx'] = selected_rows[0]

        csv_events = events_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Event Diagnostics (CSV)",
            data=csv_events,
            file_name="displacement_and_slip_events.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# MÉTRICAS Y RESUMEN AL FINAL
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Trace Status & Dynamics Overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("⏱️ Samples / Frequency", f"{len(df):,} pts", f"{fs:.1f} Hz")
c2.metric("⌛ Total Duration", f"{df['elapsed_sec'].iloc[-1]/60:.1f} min")

if enable_physics_model and has_3d_acc:
    r_min = plot_df['Slip_Risk_Ratio'].min()
    r_max = plot_df['Slip_Risk_Ratio'].max()
    max_jerk = plot_df['Jerk_XY'].abs().max()
    
    c3.metric("📐 Risk Range (R)", f"{r_min:.2f} to {r_max:.2f}")
    c4.metric("💥 Peak Risk (R_max)", f"{r_max:.3f}", f"{'SLIP' if r_max >= mu_s else 'OK'}")
    c5.metric("🔨 Maximum Jerk", f"{max_jerk:.1f} g/s", f"{len(events_df)} Events", delta_color="inverse")
else:
    c3.metric("🎯 Thresholds Exceeded", f"{len(events_df)} events")
    max_v = plot_df[visible_axes].max().max() if visible_axes else 0.0
    min_v = plot_df[visible_axes].min().min() if visible_axes else 0.0
    c4.metric("🚀 Peak Maximum", f"{max_v:.3f}")
    c5.metric("📉 Minimum", f"{min_v:.3f}")
