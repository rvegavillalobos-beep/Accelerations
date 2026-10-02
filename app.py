import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import io

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLES
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

# Brief descriptive title
st.title("📈 Measurement Processor")
st.caption("Friction evaluation, Jerk impact analysis, displacement estimation, and root-cause diagnostics.")

# ---------------------------------------------------------
# DATA LOADING AND PREPROCESSING
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

    # Time processing
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], errors='coerce')
        df['elapsed_sec'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()
    else:
        df['elapsed_sec'] = np.arange(len(df)) * 0.05
        df['time'] = df['elapsed_sec'].astype(str) + " s"

    # Accelerometer magnitude mapping
    acc_cols = [c for c in df.columns if 'Acc' in c and '(' in c]
    if len(acc_cols) >= 3:
        df['Acc_Mag'] = np.sqrt(df[acc_cols[0]]**2 + df[acc_cols[1]]**2 + df[acc_cols[2]]**2)
    
    # Gyroscope magnitude mapping
    gyro_cols = [c for c in df.columns if 'As' in c and '(' in c]
    if len(gyro_cols) >= 3:
        df['Gyro_Mag'] = np.sqrt(df[gyro_cols[0]]**2 + df[gyro_cols[1]]**2 + df[gyro_cols[2]]**2)

    return df

# ---------------------------------------------------------
# SIDEBAR (CONTROLS & CONFIGURATION)
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

# Variable group selection
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
visible_axes = st.sidebar.multiselect(
    "Visible axes on plot:",
    options=selected_axes,
    default=selected_axes
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Signal Filtering")
apply_smoothing = st.sidebar.checkbox("Apply Moving Average (Smoothing)", value=False)
window_size = st.sidebar.slider("Smoothing Window (samples):", 3, 51, 9, step=2) if apply_smoothing else 1

# Gravity removal enabled by default
remove_gravity = st.sidebar.checkbox("Remove Gravity from Z (Dynamic Acceleration)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🧲 Friction & Slip Parameters")
enable_physics_model = st.sidebar.checkbox("Enable Dynamic Friction Model", value=True)
mu_s = st.sidebar.number_input("Static Friction Coefficient (µs):", value=0.28, step=0.01)
safety_factor = st.sidebar.slider("Safety Factor (%):", 50, 100, 80) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Conventional Thresholds")
enable_upper = st.sidebar.checkbox("Enable Upper Threshold", value=True)
upper_thresh = st.sidebar.number_input("Upper Threshold (|g|):", value=0.16, step=0.01) if enable_upper else None

enable_lower = st.sidebar.checkbox("Enable Lower Threshold", value=False)
lower_thresh = st.sidebar.number_input("Lower Threshold (|g|):", value=-0.16, step=0.01) if enable_lower else None

min_distance_sec = st.sidebar.slider("Minimum Event Separation (s):", 0.1, 10.0, 1.0, 0.1)

# ---------------------------------------------------------
# SIGNAL PROCESSING & PHYSICAL CALCULATIONS
# ---------------------------------------------------------
plot_df = df.copy()

if apply_smoothing:
    for col in selected_axes:
        plot_df[col] = plot_df[col].rolling(window=window_size, center=True).mean().bfill().ffill()

# Intelligent identification of triaxial X, Y, Z columns
acc_x_col = next((c for c in plot_df.columns if 'acc' in c.lower() and 'x' in c.lower()), None)
acc_y_col = next((c for c in plot_df.columns if 'acc' in c.lower() and 'y' in c.lower()), None)
acc_z_col = next((c for c in plot_df.columns if 'acc' in c.lower() and 'z' in c.lower()), None)
has_3d_acc = all([acc_x_col, acc_y_col, acc_z_col])

if remove_gravity and acc_z_col:
    plot_df[acc_z_col] = plot_df[acc_z_col] - plot_df[acc_z_col].mean()

# Sample time step
dt_sample = plot_df['elapsed_sec'].diff().median()
if pd.isna(dt_sample) or dt_sample <= 0:
    dt_sample = 0.05
fs = 1.0 / dt_sample if dt_sample > 0 else 0.0

# PHYSICAL MODEL: JERK, FRICTION, AND REAL DISPLACEMENT
if enable_physics_model and has_3d_acc:
    # 1. Dynamic Normal Force considering gravity on Z
    acc_z_vals = plot_df[acc_z_col].values
    mean_z = np.mean(acc_z_vals)
    normal_g = acc_z_vals if mean_z > 0.5 else (1.0 + acc_z_vals)
    normal_g = np.maximum(0.01, normal_g)
    plot_df['Normal_Force_g'] = normal_g

    # 2. Horizontal XY Acceleration and Risk Ratio R(t)
    plot_df['Acc_Horiz_XY'] = np.sqrt(plot_df[acc_x_col]**2 + plot_df[acc_y_col]**2)
    plot_df['Slip_Risk_Ratio'] = plot_df['Acc_Horiz_XY'] / plot_df['Normal_Force_g']
    plot_df['Acc_Max_Allowed'] = mu_s * plot_df['Normal_Force_g']

    # 3. Jerk calculation (Rate of acceleration change in g/s)
    plot_df['Jerk_XY'] = plot_df['Acc_Horiz_XY'].diff().fillna(0) / dt_sample

    # 4. Net Slip Acceleration (Using Dynamic Coefficient µk ≈ 0.9 * µs)
    mu_k = mu_s * 0.9
    acc_net_g = np.maximum(0.0, plot_df['Acc_Horiz_XY'] - (mu_k * plot_df['Normal_Force_g']))
    plot_df['Acc_Net_m_s2'] = acc_net_g * 9.81

# ---------------------------------------------------------
# DETECTION ALGORITHM & DISPLACEMENT DIAGNOSTICS
# ---------------------------------------------------------
def detect_comprehensive_events(data_df, channels, upper=None, lower=None, min_dist_s=1.0, check_slip=False, mu_stat=0.28):
    dt = data_df['elapsed_sec'].diff().median() or 0.05
    dist_samples = int(max(1, min_dist_s / dt))
    events = []
    
    # A) Simple threshold checks per axis
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
                    'Root Cause Diagnosis': "Specific Threshold Exceeded",
                    'Estimated Effect': "Axis tolerance threshold breach"
                })

    # B) Physical Slip Evaluation
    if check_slip and 'Slip_Risk_Ratio' in data_df.columns:
        slip_peaks, _ = find_peaks(data_df['Slip_Risk_Ratio'].values, height=mu_stat, distance=dist_samples)
        
        for p in slip_peaks:
            # Event duration over threshold
            start_p = p
            while start_p > 0 and data_df['Slip_Risk_Ratio'].iloc[start_p] >= mu_stat:
                start_p -= 1
                
            end_p = p
            while end_p < len(data_df) - 1 and data_df['Slip_Risk_Ratio'].iloc[end_p] >= mu_stat:
                end_p += 1
                
            duration_sec = (end_p - start_p) * dt
            
            # Net acceleration integration during event
            segment_acc = data_df['Acc_Net_m_s2'].iloc[start_p:end_p+1]
            avg_acc_net = segment_acc.mean() if len(segment_acc) > 0 else 0.0
            
            # Approximate displacement in mm (d = 0.5 * a * t^2)
            disp_mm = 0.5 * avg_acc_net * (duration_sec ** 2) * 1000
            
            # Root cause diagnosis (evaluating Z force)
            z_val = data_df['Normal_Force_g'].iloc[p]
            cause = "⚠️ Load Loss (Low Z / Bounce)" if z_val < 0.85 else "💥 Horizontal Impact (XY / Braking)"
            
            # Effect classification
            if disp_mm < 0.5:
                efect = "🟢 Micro-vibration (No real displacement)"
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
    mu_stat=mu_s
)

# ---------------------------------------------------------
# METRIC PANELS & KPIS
# ---------------------------------------------------------
st.markdown("### 📊 Trace Status, Friction & Dynamics Overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("⏱️ Samples / Frequency", f"{len(df):,} pts", f"{fs:.1f} Hz")
c2.metric("⌛ Total Duration", f"{df['elapsed_sec'].iloc[-1]/60:.1f} min")

if enable_physics_model and has_3d_acc:
    r_min = plot_df['Slip_Risk_Ratio'].min()
    r_max = plot_df['Slip_Risk_Ratio'].max()
    slips_count = (plot_df['Slip_Risk_Ratio'] >= mu_s).sum()
    max_jerk = plot_df['Jerk_XY'].abs().max()
    
    c3.metric("📐 Risk Range (R)", f"{r_min:.2f} to {r_max:.2f}")
    c4.metric("💥 Peak Risk (R_max)", f"{r_max:.3f}", f"{'SLIP' if r_max >= mu_s else 'OK'}")
    c5.metric("🔨 Maximum Jerk", f"{max_jerk:.1f} g/s", f"{len(events_df)} Events", delta_color="inverse")

    # Zone distribution
    safe_pct = (plot_df['Slip_Risk_Ratio'] < (mu_s * safety_factor)).mean() * 100
    warn_pct = ((plot_df['Slip_Risk_Ratio'] >= (mu_s * safety_factor)) & (plot_df['Slip_Risk_Ratio'] < mu_s)).mean() * 100
    crit_pct = (plot_df['Slip_Risk_Ratio'] >= mu_s).mean() * 100

    st.markdown("**Trace Time Distribution by Risk Range:**")
    col_s, col_w, col_c = st.columns(3)
    col_s.caption(f"🟢 **Safe (R < {mu_s * safety_factor:.2f}):** {safe_pct:.1f}% of time")
    col_w.caption(f"🟡 **Warning ({mu_s * safety_factor:.2f} ≤ R < {mu_s:.2f}):** {warn_pct:.1f}% of time")
    col_c.caption(f"🔴 **Slip (R ≥ {mu_s:.2f}):** {crit_pct:.1f}% of time")
else:
    c3.metric("🎯 Thresholds Exceeded", f"{len(events_df)} events")
    max_v = plot_df[visible_axes].max().max() if visible_axes else 0.0
    min_v = plot_df[visible_axes].min().min() if visible_axes else 0.0
    c4.metric("🚀 Peak Maximum", f"{max_v:.3f}")
    c5.metric("📉 Minimum", f"{min_v:.3f}")

st.markdown("---")

# ---------------------------------------------------------
# TABS STRUCTURE (FFT AND STATISTICS REMOVED)
# ---------------------------------------------------------
tab_plot, tab_events = st.tabs([
    "📊 Interactive Chart", 
    "🚨 Event & Displacement Log"
])

selected_event = None
if 'selected_event_idx' in st.session_state and not events_df.empty:
    idx = st.session_state['selected_event_idx']
    if idx < len(events_df):
        selected_event = events_df.iloc[idx]

# 1. INTERACTIVE PLOT TAB
with tab_plot:
    st.subheader(f"Signal Visualization: {target_group}")
    
    if selected_event is not None:
        st.info(
            f"📍 **Selected Event:** **{selected_event['Axis / Criterion']}** at t = **{selected_event['Elapsed Time (s)']} s** | "
            f"Diagnosis: **{selected_event['Root Cause Diagnosis']}** | "
            f"Est. Displacement: **{selected_event['Est. Displacement (mm)']} mm**"
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

    # Additional dynamic friction curves
    if enable_physics_model and has_3d_acc and target_group == "Acceleration (g)":
        # Vector Magnitude disabled by default
        show_vector_xy = st.checkbox("Show Resultant Horizontal Acceleration (Acc_Horiz_XY)", value=False)
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

    # Fixed threshold lines
    if upper_thresh is not None:
        fig.add_hline(y=upper_thresh, line_dash="dash", line_color="crimson", annotation_text=f"+Threshold ({upper_thresh})")
    if lower_thresh is not None:
        fig.add_hline(y=lower_thresh, line_dash="dash", line_color="royalblue", annotation_text=f"-Threshold ({lower_thresh})")

    # Mark events
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

    # Selected event highlight marker
    if selected_event is not None:
        selected_time = selected_event['Elapsed Time (s)']
        selected_val = selected_event['Measured Value'] if isinstance(selected_event['Measured Value'], (int, float)) else 0.0

        fig.add_vline(x=selected_time, line_width=2, line_dash="dot", line_color="gold")
        fig.add_trace(go.Scatter(
            x=[selected_time],
            y=[selected_val],
            mode='markers',
            name='Selected',
            marker=dict(symbol='cross', size=14, color='yellow', line=dict(width=2, color='black')),
            hoverinfo='skip'
        ))

    fig.update_layout(
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title="Amplitude",
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# 2. EVENT DIAGNOSTICS & DISPLACEMENT TAB
with tab_events:
    st.subheader("🚨 Event Diagnostics, Jerk & Real Displacement")
    
    if events_df.empty:
        st.info("No events violating thresholds or friction conditions were detected.")
    else:
        st.write(f"Recorded **{len(events_df)}** events evaluated by the physical model:")
        st.caption("👈 **Click on any row in the table** to locate and highlight the point on the interactive chart.")
        
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
