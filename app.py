import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import io

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Procesador de Acelerómetro & Modelo de Deslizamiento",
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

st.title("📈 Procesador Analítico & Modelo de Deslizamiento en Conveyor")
st.caption("Análisis dinámico de vibración, acoplamiento vectorial 3D y evaluación de fricción en tiempo real.")

# ---------------------------------------------------------
# CARGA Y PROCESAMIENTO DE DATOS
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

    # Procesamiento de tiempo
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], errors='coerce')
        df['elapsed_sec'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()
    else:
        df['elapsed_sec'] = np.arange(len(df)) * 0.05
        df['time'] = df['elapsed_sec'].astype(str) + " s"

    # Mapeo de columnas de aceleración
    acc_cols = [c for c in df.columns if 'Acc' in c and '(' in c]
    if len(acc_cols) >= 3:
        df['Acc_Mag'] = np.sqrt(df[acc_cols[0]]**2 + df[acc_cols[1]]**2 + df[acc_cols[2]]**2)
    
    # Mapeo de columnas de giroscopio
    gyro_cols = [c for c in df.columns if 'As' in c and '(' in c]
    if len(gyro_cols) >= 3:
        df['Gyro_Mag'] = np.sqrt(df[gyro_cols[0]]**2 + df[gyro_cols[1]]**2 + df[gyro_cols[2]]**2)

    return df

# ---------------------------------------------------------
# BARRA LATERAL (CONTROLES Y CONFIGURACIÓN)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuración y Filtros")

uploaded_file = st.sidebar.file_uploader(
    "Cargar archivo de datos (.txt, .csv, .tsv)", 
    type=["txt", "csv", "tsv"]
)

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)
else:
    try:
        df = load_and_preprocess_data("20260910103112.txt")
        st.sidebar.info("📂 Usando archivo de traza por defecto (`20260910103112.txt`).")
    except Exception:
        st.warning("Por favor, sube un archivo de traza para comenzar.")
        st.stop()

# Selección de Grupo de Variables
target_group = st.sidebar.radio(
    "Grupo de Variables a Analizar:",
    ["Aceleración (g)", "Velocidad Angular (°/s)", "Ángulos de Inclinación (°)", "Magnetómetro (uT)"]
)

if target_group == "Aceleración (g)":
    selected_axes = [c for c in df.columns if 'Acc' in c]
elif target_group == "Velocidad Angular (°/s)":
    selected_axes = [c for c in df.columns if 'As' in c or 'Gyro' in c]
elif target_group == "Ángulos de Inclinación (°)":
    selected_axes = [c for c in df.columns if 'Angle' in c]
else:
    selected_axes = [c for c in df.columns if 'H' in c and 'uT' in c]

# Ocultar / Mostrar Ejes Dinámicamente
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Ocultar / Mostrar Ejes")
visible_axes = st.sidebar.multiselect(
    "Ejes visibles en la gráfica:",
    options=selected_axes,
    default=selected_axes
)

# Filtro de Suavizado y Gravedad
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Filtrado de Señal")
apply_smoothing = st.sidebar.checkbox("Aplicar Media Móvil (Suavizado)", value=False)
window_size = st.sidebar.slider("Ventana de suavizado (muestras):", 3, 51, 9, step=2) if apply_smoothing else 1

remove_gravity = st.sidebar.checkbox("Remover Gravedad en Z (Aceleración Dinámica)", value=False)

# Configuración de Física y Fricción
st.sidebar.markdown("---")
st.sidebar.subheader("🧲 Parámetros de Fricción y Deslizamiento")
enable_physics_model = st.sidebar.checkbox("Activar Modelo de Fricción Dinámica", value=True)
mu_s = st.sidebar.number_input("Coeficiente de Fricción Estático (µs):", value=0.28, step=0.01)
safety_factor = st.sidebar.slider("Factor de Seguridad (%):", 50, 100, 80) / 100.0

# Configuración de Umbrales Convencionales
st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Configuración de Umbrales Simples")
enable_upper = st.sidebar.checkbox("Activar Umbral Superior", value=True)
upper_thresh = st.sidebar.number_input("Umbral Superior (|g|):", value=0.16, step=0.01) if enable_upper else None

enable_lower = st.sidebar.checkbox("Activar Umbral Inferior", value=False)
lower_thresh = st.sidebar.number_input("Umbral Inferior (|g|):", value=-0.16, step=0.01) if enable_lower else None

min_distance_sec = st.sidebar.slider("Separación Mínima entre Eventos (s):", 0.1, 10.0, 1.0, 0.1)

# ---------------------------------------------------------
# PROCESAMIENTO DE SEÑALES Y CÁLCULOS FÍSICOS
# ---------------------------------------------------------
plot_df = df.copy()

if apply_smoothing:
    for col in selected_axes:
        plot_df[col] = plot_df[col].rolling(window=window_size, center=True).mean().bfill().ffill()

# Detección inteligente de columnas triaxiales X, Y, Z
acc_x_col = next((c for c in plot_df.columns if 'acc' in c.lower() and 'x' in c.lower()), None)
acc_y_col = next((c for c in plot_df.columns if 'acc' in c.lower() and 'y' in c.lower()), None)
acc_z_col = next((c for c in plot_df.columns if 'acc' in c.lower() and 'z' in c.lower()), None)
has_3d_acc = all([acc_x_col, acc_y_col, acc_z_col])

if remove_gravity and acc_z_col:
    plot_df[acc_z_col] = plot_df[acc_z_col] - plot_df[acc_z_col].mean()

# MODELO DE FÍSICA DE DESLIZAMIENTO
if enable_physics_model and has_3d_acc:
    # 1. Aceleración Horizontal Resultante en el Plano XY
    plot_df['Acc_Horiz_XY'] = np.sqrt(plot_df[acc_x_col]**2 + plot_df[acc_y_col]**2)
    
    # 2. Fuerza Normal Dinámica considerando el Eje Z
    acc_z_vals = plot_df[acc_z_col].values
    mean_z = np.mean(acc_z_vals)
    normal_g = acc_z_vals if mean_z > 0.5 else (1.0 + acc_z_vals)
    normal_g = np.maximum(0.01, normal_g) # Prevenir división por cero
    
    plot_df['Normal_Force_g'] = normal_g
    
    # 3. Ratio de Riesgo de Deslizamiento R(t)
    plot_df['Slip_Risk_Ratio'] = plot_df['Acc_Horiz_XY'] / plot_df['Normal_Force_g']
    
    # 4. Aceleración Horizontal Máxima Permitida instante a instante
    plot_df['Acc_Max_Allowed'] = mu_s * plot_df['Normal_Force_g']

# ---------------------------------------------------------
# ALGORITMO DE DETECCIÓN MULTIEJE Y DE DESLIZAMIENTO
# ---------------------------------------------------------
def detect_all_events(data_df, channels, upper=None, lower=None, min_dist_s=1.0, check_slip=False, mu_stat=0.28):
    dt = data_df['elapsed_sec'].diff().median()
    if pd.isna(dt) or dt <= 0:
        dt = 0.05
    dist_samples = int(max(1, min_dist_s / dt))
    
    events = []
    
    # A) Detección de sobrepaso directo por eje
    for col in channels:
        if col not in data_df.columns:
            continue
            
        if upper is not None:
            peaks, _ = find_peaks(data_df[col].values, height=upper, distance=dist_samples)
            for p in peaks:
                events.append({
                    'Índice': p,
                    'Timestamp (ISO)': data_df['time'].iloc[p],
                    'Tiempo Transcurrido (s)': round(data_df['elapsed_sec'].iloc[p], 3),
                    'Eje / Criterio': col,
                    'Tipo de Evento': 'Exceso Superior ↑',
                    'Valor Medido': round(data_df[col].iloc[p], 4),
                    'Límite Establ.': upper
                })
                
        if lower is not None:
            peaks_low, _ = find_peaks(-data_df[col].values, height=-lower, distance=dist_samples)
            for p in peaks_low:
                events.append({
                    'Índice': p,
                    'Timestamp (ISO)': data_df['time'].iloc[p],
                    'Tiempo Transcurrido (s)': round(data_df['elapsed_sec'].iloc[p], 3),
                    'Eje / Criterio': col,
                    'Tipo de Evento': 'Exceso Inferior ↓',
                    'Valor Medido': round(data_df[col].iloc[p], 4),
                    'Límite Establ.': lower
                })

    # B) Detección de Deslizamiento Físico (R >= mu_s)
    if check_slip and 'Slip_Risk_Ratio' in data_df.columns:
        slip_peaks, _ = find_peaks(data_df['Slip_Risk_Ratio'].values, height=mu_stat, distance=dist_samples)
        for p in slip_peaks:
            events.append({
                'Índice': p,
                'Timestamp (ISO)': data_df['time'].iloc[p],
                'Tiempo Transcurrido (s)': round(data_df['elapsed_sec'].iloc[p], 3),
                'Eje / Criterio': '🚨 Deslizamiento Físico (XY / Normal Z)',
                'Tipo de Evento': 'Riesgo R >= µs',
                'Valor Medido': round(data_df['Slip_Risk_Ratio'].iloc[p], 4),
                'Límite Establ.': mu_stat
            })
            
    res_df = pd.DataFrame(events)
    if not res_df.empty:
        res_df = res_df.sort_values(by=['Tiempo Transcurrido (s)', 'Eje / Criterio']).reset_index(drop=True)
    return res_df

events_df = detect_all_events(
    plot_df, 
    channels=visible_axes, 
    upper=upper_thresh, 
    lower=lower_thresh, 
    min_dist_s=min_distance_sec,
    check_slip=(enable_physics_model and has_3d_acc),
    mu_stat=mu_s
)

# ---------------------------------------------------------
# PANELES DE MÉTRICAS Y KPIS CON RANGOS
# ---------------------------------------------------------
# Cálculo preventivo de frecuencia de muestreo (fs) para evitar NameError
dt_sample = df['elapsed_sec'].diff().median()
fs = 1.0 / dt_sample if (pd.notna(dt_sample) and dt_sample > 0) else 0.0

st.markdown("### 📊 Estado General de Traza y Fricción")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("⏱️ Muestras / Frecuencia", f"{len(df):,} pts", f"{fs:.1f} Hz")
c2.metric("⌛ Duración Total", f"{df['elapsed_sec'].iloc[-1]/60:.1f} min")

if enable_physics_model and has_3d_acc:
    r_min = plot_df['Slip_Risk_Ratio'].min()
    r_max = plot_df['Slip_Risk_Ratio'].max()
    slips_count = (plot_df['Slip_Risk_Ratio'] >= mu_s).sum()
    
    c3.metric("📐 Rango de Riesgo (R)", f"{r_min:.2f} a {r_max:.2f}")
    c4.metric("💥 Riesgo Pico (R_max)", f"{r_max:.3f}", f"{'DESLIZAMIENTO' if r_max >= mu_s else 'OK'}")
    c5.metric("🚨 Eventos Registrados", f"{len(events_df)}", f"{slips_count} por Fricción", delta_color="inverse")

    # Distribución porcentual por zonas
    safe_pct = (plot_df['Slip_Risk_Ratio'] < (mu_s * safety_factor)).mean() * 100
    warn_pct = ((plot_df['Slip_Risk_Ratio'] >= (mu_s * safety_factor)) & (plot_df['Slip_Risk_Ratio'] < mu_s)).mean() * 100
    crit_pct = (plot_df['Slip_Risk_Ratio'] >= mu_s).mean() * 100

    st.markdown("**Distribución del Tiempo de Traza por Rangos de Riesgo:**")
    col_s, col_w, col_c = st.columns(3)
    col_s.caption(f"🟢 **Seguro (R < {mu_s * safety_factor:.2f}):** {safe_pct:.1f}% del tiempo")
    col_w.caption(f"🟡 **Advertencia ({mu_s * safety_factor:.2f} ≤ R < {mu_s:.2f}):** {warn_pct:.1f}% del tiempo")
    col_c.caption(f"🔴 **Deslizamiento (R ≥ {mu_s:.2f}):** {crit_pct:.1f}% del tiempo")
else:
    c3.metric("🎯 Thresholds Superados", f"{len(events_df)} eventos")
    max_v = plot_df[visible_axes].max().max() if visible_axes else 0.0
    min_v = plot_df[visible_axes].min().min() if visible_axes else 0.0
    c4.metric("🚀 Pico Máximo", f"{max_v:.3f}")
    c5.metric("📉 Mínimo", f"{min_v:.3f}")

st.markdown("---")

# ---------------------------------------------------------
# PESTAÑAS DE VISUALIZACIÓN Y ANÁLISIS
# ---------------------------------------------------------
tab_plot, tab_events, tab_fft, tab_stats = st.tabs([
    "📊 Gráfica Interactiva", 
    "🚨 Registros de Thresholds y Deslizamiento", 
    "⚡ Análisis de Frecuencia (FFT)", 
    "📋 Estadísticas Descriptivas"
])

selected_event = None
if 'selected_event_idx' in st.session_state and not events_df.empty:
    idx = st.session_state['selected_event_idx']
    if idx < len(events_df):
        selected_event = events_df.iloc[idx]

# 1. PESTAÑA DE PLOTEO INTERACTIVO
with tab_plot:
    st.subheader(f"Visualización de Señales: {target_group}")
    
    if selected_event is not None:
        st.info(
            f"📍 **Evento Seleccionado:** **{selected_event['Eje / Criterio']}** en t = **{selected_event['Tiempo Transcurrido (s)']} s** "
            f"(Valor Medido: **{selected_event['Valor Medido']}** | Tipo: **{selected_event['Tipo de Evento']}**)"
        )
    
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, axis_col in enumerate(visible_axes):
        opacity = 1.0
        line_width = 1.5
        
        if selected_event is not None:
            if axis_col == selected_event['Eje / Criterio']:
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
            hovertemplate='<b>Eje:</b> ' + axis_col + '<br><b>Tiempo:</b> %{x:.2f} s<br><b>Valor:</b> %{y:.4f}<extra></extra>'
        ))

    # Curvas Adicionales de Fricción Dinámica
    if enable_physics_model and has_3d_acc and target_group == "Aceleración (g)":
        show_vector_xy = st.checkbox("Mostrar Aceleración Horizontal Resultante (Acc_Horiz_XY)", value=True)
        show_allowed_limit = st.checkbox("Mostrar Límite Dinámico Permitido por Z (Acc_Max_Allowed)", value=True)
        
        if show_vector_xy:
            fig.add_trace(go.Scatter(
                x=plot_df['elapsed_sec'],
                y=plot_df['Acc_Horiz_XY'],
                mode='lines',
                name='Aceleración Vectorial XY (g)',
                line=dict(width=2, color='black', dash='solid'),
                hovertemplate='<b>Vector XY:</b> %{y:.4f} g<extra></extra>'
            ))
            
        if show_allowed_limit:
            fig.add_trace(go.Scatter(
                x=plot_df['elapsed_sec'],
                y=plot_df['Acc_Max_Allowed'],
                mode='lines',
                name='Límite de Fricción Dinámico (µs * N)',
                line=dict(width=1.5, color='red', dash='dash'),
                hovertemplate='<b>Límite Fricción:</b> %{y:.4f} g<extra></extra>'
            ))

    # Líneas Fijas de Umbral
    if upper_thresh is not None:
        fig.add_hline(y=upper_thresh, line_dash="dash", line_color="crimson", annotation_text=f"+Umbral ({upper_thresh})")
    if lower_thresh is not None:
        fig.add_hline(y=lower_thresh, line_dash="dash", line_color="royalblue", annotation_text=f"-Umbral ({lower_thresh})")

    # Marcar Eventos
    if not events_df.empty:
        fig.add_trace(go.Scatter(
            x=events_df['Tiempo Transcurrido (s)'],
            y=events_df['Valor Medido'],
            mode='markers',
            name='Eventos Detectados',
            marker=dict(symbol='x', size=8, color='red', line=dict(width=1.5)),
            hovertext=events_df['Eje / Criterio'],
            hovertemplate='<b>EVENTO DETECTADO</b><br><b>Criterio:</b> %{hovertext}<br><b>Tiempo:</b> %{x:.2f} s<br><b>Valor:</b> %{y:.4f}<extra></extra>'
        ))

    # Marcador del Evento Seleccionado en la Tabla
    if selected_event is not None:
        selected_time = selected_event['Tiempo Transcurrido (s)']
        selected_val = selected_event['Valor Medido']

        fig.add_vline(x=selected_time, line_width=2, line_dash="dot", line_color="gold")
        fig.add_trace(go.Scatter(
            x=[selected_time],
            y=[selected_val],
            mode='markers',
            name='Seleccionado',
            marker=dict(symbol='cross', size=14, color='yellow', line=dict(width=2, color='black')),
            hoverinfo='skip'
        ))

    fig.update_layout(
        xaxis_title="Tiempo Transcurrido (segundos)",
        yaxis_title="Amplitud",
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# 2. PESTAÑA DE REGISTROS DE THRESHOLDS Y DESLIZAMIENTO
with tab_events:
    st.subheader("🚨 Marcas de Tiempo de Umbrales y Deslizamiento")
    
    if events_df.empty:
        st.info("No se registraron eventos que violen los umbrales ni las condiciones de fricción.")
    else:
        st.write(f"Se contabilizan **{len(events_df)}** eventos en la traza actual:")
        st.caption("👈 **Haz clic en cualquier fila de la tabla** para ubicar y resaltar el instante en la gráfica interactiva.")
        
        event_selection = st.dataframe(
            events_df[['Índice', 'Timestamp (ISO)', 'Tiempo Transcurrido (s)', 'Eje / Criterio', 'Tipo de Evento', 'Valor Medido', 'Límite Establ.']],
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
            label="📥 Descargar Registro de Eventos (CSV)",
            data=csv_events,
            file_name="threshold_and_slip_events.csv",
            mime="text/csv"
        )

# 3. PESTAÑA DE ANÁLISIS ESPECTRAL (FFT)
with tab_fft:
    st.subheader("⚡ Espectro de Frecuencia (Transformada Rápida de Fourier)")
    
    fft_channel = st.selectbox("Seleccionar canal para FFT:", options=visible_axes if visible_axes else selected_axes)
    
    signal_data = plot_df[fft_channel].values - np.mean(plot_df[fft_channel].values)
    n = len(signal_data)
    fft_vals = np.abs(np.fft.rfft(signal_data)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, d=dt_sample)
    
    valid_idx = freqs > 0.1
    freqs_valid = freqs[valid_idx]
    fft_valid = fft_vals[valid_idx]
    
    dom_idx = np.argmax(fft_valid)
    dom_freq = freqs_valid[dom_idx]
    dom_amp = fft_valid[dom_idx]
    
    st.success(f"📌 **Frecuencia Dominante de Vibración:** **{dom_freq:.3f} Hz** (Amplitud: **{dom_amp:.5f}**) ")
    
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(
        x=freqs_valid,
        y=fft_valid,
        mode='lines',
        name='Amplitud FFT',
        line=dict(color='#8884d8', width=1.5)
    ))
    fig_fft.update_layout(
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="Amplitud Espectral",
        height=450,
        template="plotly_white"
    )
    st.plotly_chart(fig_fft, use_container_width=True)

# 4. PESTAÑA DE ESTADÍSTICAS
with tab_stats:
    st.subheader("📋 Resumen Estadístico Completo")
    st.dataframe(df[visible_axes].describe().T if visible_axes else df[selected_axes].describe().T, use_container_width=True)
    
    csv_clean = plot_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Conjunto de Datos Procesado (CSV)",
        data=csv_clean,
        file_name="processed_accelerometer_data.csv",
        mime="text/csv"
    )
