import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import io

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Procesador de Datos de Acelerómetro & IMU",
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

st.title("📈 Procesador Analítico de Acelerómetro & IMU")
st.caption("Herramienta interactiva para análisis de vibración, detección de umbrales y marcas de tiempo.")

# ---------------------------------------------------------
# CARGA Y PROCESAMIENTO DE DATOS
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data(file_bytes_or_path):
    if isinstance(file_bytes_or_path, str):
        df = pd.read_csv(file_bytes_or_path, sep="\t")
    else:
        # Detectar separador dinámicamente
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

# Selección de Señal a Analizar
available_signals = [c for c in df.columns if any(p in c for p in ['Acc', 'As', 'Angle', 'H', 'Temperature'])]
if 'Acc_Mag' in df.columns and 'Acc_Mag' not in available_signals:
    available_signals.append('Acc_Mag')

target_group = st.sidebar.radio(
    "Grupo de Variables:",
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

remove_gravity = st.sidebar.checkbox("Remover Gravedad (Aceleración Dinámica)", value=False)

# Configuración de Thresholds
st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Configuración de Umbrales (Thresholds)")

primary_channel = st.sidebar.selectbox("Canal para Detección de Umbral:", options=selected_axes)

signal_min = float(df[primary_channel].min())
signal_max = float(df[primary_channel].max())

enable_upper = st.sidebar.checkbox("Activar Umbral Superior", value=True)
upper_thresh = st.sidebar.number_input(
    "Umbral Superior:", 
    value=round(signal_max * 0.85, 3), 
    step=0.01
) if enable_upper else None

enable_lower = st.sidebar.checkbox("Activar Umbral Inferior", value=False)
lower_thresh = st.sidebar.number_input(
    "Umbral Inferior:", 
    value=round(signal_min * 1.15, 3), 
    step=0.01
) if enable_lower else None

min_distance_sec = st.sidebar.slider(
    "Separación Mínima entre Eventos (s):", 
    0.1, 10.0, 1.0, 0.1,
    help="Evita contar múltiples veces un mismo pico dentro de una misma fluctuación."
)

# ---------------------------------------------------------
# PROCESAMIENTO DE SEÑALES SELECCIONADAS
# ---------------------------------------------------------
plot_df = df.copy()

if apply_smoothing:
    for col in selected_axes:
        plot_df[col] = plot_df[col].rolling(window=window_size, center=True).mean().bfill().ffill()

if remove_gravity and 'AccZ(g)' in plot_df.columns:
    plot_df['AccZ(g)'] = plot_df['AccZ(g)'] - plot_df['AccZ(g)'].mean()
    if 'Acc_Mag' in plot_df.columns:
        plot_df['Acc_Mag'] = np.sqrt(plot_df['AccX(g)']**2 + plot_df['AccY(g)']**2 + plot_df['AccZ(g)']**2)

# ---------------------------------------------------------
# ALGORITMO DE DETECCIÓN DE THRESHOLDS
# ---------------------------------------------------------
def detect_threshold_events(data_df, signal_col, upper=None, lower=None, min_dist_s=1.0):
    dt = data_df['elapsed_sec'].diff().median()
    if pd.isna(dt) or dt <= 0:
        dt = 0.05
    dist_samples = int(max(1, min_dist_s / dt))
    
    events = []
    
    if upper is not None:
        peaks, _ = find_peaks(data_df[signal_col].values, height=upper, distance=dist_samples)
        for p in peaks:
            events.append({
                'Índice': p,
                'Timestamp (ISO)': data_df['time'].iloc[p],
                'Tiempo Transcurrido (s)': round(data_df['elapsed_sec'].iloc[p], 3),
                'Tipo de Evento': 'Exceso Superior ↑',
                'Valor Medido': round(data_df[signal_col].iloc[p], 4),
                'Umbral Establ.': upper
            })
            
    if lower is not None:
        peaks_low, _ = find_peaks(-data_df[signal_col].values, height=-lower, distance=dist_samples)
        for p in peaks_low:
            events.append({
                'Índice': p,
                'Timestamp (ISO)': data_df['time'].iloc[p],
                'Tiempo Transcurrido (s)': round(data_df['elapsed_sec'].iloc[p], 3),
                'Tipo de Evento': 'Exceso Inferior ↓',
                'Valor Medido': round(data_df[signal_col].iloc[p], 4),
                'Umbral Establ.': lower
            })
            
    res_df = pd.DataFrame(events)
    if not res_df.empty:
        res_df = res_df.sort_values(by='Tiempo Transcurrido (s)').reset_index(drop=True)
    return res_df

events_df = detect_threshold_events(
    plot_df, 
    primary_channel, 
    upper=upper_thresh, 
    lower=lower_thresh, 
    min_dist_s=min_distance_sec
)

# ---------------------------------------------------------
# PANELES DE MÉTRICAS / KPIS
# ---------------------------------------------------------
dt_sample = df['elapsed_sec'].diff().median()
fs = 1.0 / dt_sample if dt_sample > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("⏱️ Muestras / Frecuencia", f"{len(df):,} pts", f"{fs:.1f} Hz")
col2.metric("⌛ Duración Total", f"{df['elapsed_sec'].iloc[-1]/60:.1f} min")
col3.metric("🎯 Thresholds Superados", f"{len(events_df)} eventos")
col4.metric(f"🚀 Pico Máx ({primary_channel})", f"{plot_df[primary_channel].max():.3f}")
col5.metric(f"📉 Mínimo ({primary_channel})", f"{plot_df[primary_channel].min():.3f}")

st.markdown("---")

# ---------------------------------------------------------
# PESTAÑAS DE VISUALIZACIÓN Y ANÁLISIS
# ---------------------------------------------------------
tab_plot, tab_events, tab_fft, tab_stats = st.tabs([
    "📊 Gráfica Interactiva", 
    "🚨 Registros de Thresholds", 
    "⚡ Análisis de Frecuencia (FFT)", 
    "📋 Estadísticas Descriptivas"
])

# 1. PESTAÑA DE PLOTEO INTERACTIVO
with tab_plot:
    st.subheader(f"Visualización de Señales: {target_group}")
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, axis_col in enumerate(visible_axes):
        fig.add_trace(go.Scatter(
            x=plot_df['elapsed_sec'],
            y=plot_df[axis_col],
            mode='lines',
            name=axis_col,
            line=dict(width=1.5, color=colors[idx % len(colors)]),
            hovertext=plot_df['time'],
            hovertemplate='<b>Tiempo:</b> %{x:.2f} s<br><b>Fecha:</b> %{hovertext}<br><b>Valor:</b> %{y:.4f}<extra></extra>'
        ))

    # Líneas y zonas de Umbral
    if upper_thresh is not None:
        fig.add_hline(
            y=upper_thresh, 
            line_dash="dash", 
            line_color="crimson", 
            annotation_text=f"Umbral Sup: {upper_thresh}", 
            annotation_position="top right"
        )
    if lower_thresh is not None:
        fig.add_hline(
            y=lower_thresh, 
            line_dash="dash", 
            line_color="royalblue", 
            annotation_text=f"Umbral Inf: {lower_thresh}", 
            annotation_position="bottom right"
        )

    # Marcar Eventos Detectados
    if not events_df.empty:
        fig.add_trace(go.Scatter(
            x=events_df['Tiempo Transcurrido (s)'],
            y=events_df['Valor Medido'],
            mode='markers',
            name='Picos / Threshold Breached',
            marker=dict(symbol='x', size=10, color='red', line=dict(width=2)),
            hovertext=events_df['Timestamp (ISO)'],
            hovertemplate='<b>EVENTO DETECTADO</b><br><b>Tiempo:</b> %{x:.2f} s<br><b>Fecha:</b> %{hovertext}<br><b>Pico:</b> %{y:.4f}<extra></extra>'
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

# 2. PESTAÑA DE REGISTRO DE THRESHOLDS
with tab_events:
    st.subheader("🚨 Marcas de Tiempo de Umbrales Excedidos")
    
    if events_df.empty:
        st.info("No se registraron eventos que superen los umbrales configurados.")
    else:
        st.write(f"Se han contabilizado **{len(events_df)}** eventos que violan los umbrales estipulados:")
        
        st.dataframe(events_df, use_container_width=True)
        
        # Botón para descargar reporte de eventos
        csv_events = events_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Registro de Thresholds (CSV)",
            data=csv_events,
            file_name="threshold_events_log.csv",
            mime="text/csv"
        )

# 3. PESTAÑA DE ANÁLISIS ESPECTRAL (FFT)
with tab_fft:
    st.subheader("⚡ Espectro de Frecuencia (Transformada Rápida de Fourier)")
    
    fft_channel = st.selectbox("Seleccionar canal para FFT:", options=selected_axes)
    
    signal_data = plot_df[fft_channel].values - np.mean(plot_df[fft_channel].values)
    n = len(signal_data)
    fft_vals = np.abs(np.fft.rfft(signal_data)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, d=dt_sample)
    
    # Filtrar componente DC (0 Hz)
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
    st.dataframe(df[selected_axes].describe().T, use_container_width=True)
    
    # Descarga de datos limpios
    csv_clean = plot_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Conjunto de Datos Procesado (CSV)",
        data=csv_clean,
        file_name="processed_accelerometer_data.csv",
        mime="text/csv"
    )
