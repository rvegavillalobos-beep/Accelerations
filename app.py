import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Análisis de Vibraciones y Umbrales", layout="wide")

# --- GENERACIÓN DE DATOS SINTÉTICOS DE PRUEBA ---
@st.cache_data
def generate_sample_data():
    fs = 1000  # Frecuencia de muestreo (1000 Hz)
    duration = 2.0  # 2 segundos
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Señal base (10 Hz + 50 Hz + ruido)
    acc_x = 0.5 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, len(t))
    acc_y = 0.8 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.1, len(t))
    acc_z = 0.3 * np.cos(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, len(t))
    
    # Inyectar picos (sobrepasos de umbral)
    acc_x[350] = 3.2   # Impacto en X a t ≈ 0.35s
    acc_y[800] = -2.8  # Impacto en Y a t ≈ 0.80s
    acc_z[1200] = 2.5  # Impacto en Z a t ≈ 1.20s
    acc_x[1600] = 4.1  # Impacto severo en X a t ≈ 1.60s

    # Magnitud Vectorial Total (acc_mag)
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    df = pd.DataFrame({
        'tiempo_s': np.round(t, 3),
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'acc_mag': acc_mag
    })
    return df, fs

df, fs = generate_sample_data()

# --- TÍTULO PRINCIPAL ---
st.title("📈 Dashboard de Análisis de Vibraciones")

# --- DEFINICIÓN DE PESTAÑAS ---
tab1, tab2 = st.tabs(["⚡ Umbrales e Interactividad", "🌊 Espectro de Frecuencia (FFT)"])

# ==========================================
# TAB 1: UMBRALES E INTERACTIVIDAD
# ==========================================
with tab1:
    st.header("Análisis e Historial de Umbrales (Thresholds)")
    
    col_thresh, col_info = st.columns([1, 2])
    
    with col_thresh:
        threshold = st.number_input(
            "Definir Umbral Límite (|g|):", 
            value=2.0, 
            step=0.2,
            help="Cualquier lectura (|valor|) que supere este límite será registrada."
        )

    # Identificación de eventos por eje
    events = []
    axes = ['acc_x', 'acc_y', 'acc_z', 'acc_mag']
    
    for axis in axes:
        exceeded = df[df[axis].abs() > threshold]
        for idx, row in exceeded.iterrows():
            events.append({
                'Índice/Tiempo (s)': row['tiempo_s'],
                'Eje Afectado': axis.replace('acc_', '').upper(),
                'Valor Medido': round(row[axis], 3),
                'Umbral Configurado': threshold,
                'Columna_Raw': axis
            })
            
    events_df = pd.DataFrame(events)

    if events_df.empty:
        st.success(f"✅ No se detectaron lecturas que superen el umbral de {threshold}.")
    else:
        # Ordenar eventos por tiempo
        events_df = events_df.sort_values(by='Índice/Tiempo (s)').reset_index(drop=True)
        st.warning(f"⚠️ Se detectaron **{len(events_df)}** eventos que superaron el umbral de {threshold}.")

        st.markdown("**Instrucciones:** Haz clic sobre una fila de la tabla para resaltar el instante y eje exacto en la gráfica inferior.")

        # TABLA INTERACTIVA (Streamlit >= 1.35.0)
        event_selection = st.dataframe(
            events_df[['Índice/Tiempo (s)', 'Eje Afectado', 'Valor Medido', 'Umbral Configurado']],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single_row",
            key="threshold_table"
        )

        # Extraer selección
        selected_time = None
        selected_axis = None
        
        selected_rows = event_selection.get("selection", {}).get("rows", [])
        if selected_rows:
            row_idx = selected_rows[0]
            selected_event = events_df.iloc[row_idx]
            selected_time = selected_event['Índice/Tiempo (s)']
            selected_axis = selected_event['Columna_Raw']
            st.info(f"📍 **Evento Seleccionado:** Eje **{selected_event['Eje Afectado']}** en t = **{selected_time} s** (Valor: **{selected_event['Valor Medido']}**)")

        # --- GRAFICAR CON PLOTLY ---
        fig = go.Figure()

        # Dibujar líneas de cada eje
        colors = {'acc_x': '#EF553B', 'acc_y': '#00CC96', 'acc_z': '#636EFA', 'acc_mag': '#AB63FA'}
        
        for axis in axes:
            opacity = 1.0
            line_width = 1.5
            # Opacar los demás ejes si hay uno seleccionado
            if selected_axis and axis != selected_axis:
                opacity = 0.25
            elif selected_axis and axis == selected_axis:
                line_width = 2.5

            fig.add_trace(go.Scatter(
                x=df['tiempo_s'], 
                y=df[axis], 
                mode='lines', 
                name=axis.replace('acc_', '').upper(),
                line=dict(color=colors[axis], width=line_width),
                opacity=opacity
            ))

        # Líneas horizontales de umbral
        fig.add_hline(y=threshold, line_dash="dash", line_color="orange", annotation_text=f"+Umbral ({threshold})")
        fig.add_hline(y=-threshold, line_dash="dash", line_color="orange", annotation_text=f"-Umbral (-{threshold})")

        # Marcar la posición seleccionada
        if selected_time is not None:
            fig.add_vline(
                x=selected_time, 
                line_width=2, 
                line_dash="dot", 
                line_color="yellow",
                annotation_text=f" Evento {selected_axis.replace('acc_', '').upper()}"
            )
            
            # Obtener el valor Y exacto del punto seleccionado
            y_val = df.loc[df['tiempo_s'] == selected_time, selected_axis].values[0]
            
            fig.add_trace(go.Scatter(
                x=[selected_time],
                y=[y_val],
                mode='markers',
                marker=dict(color='yellow', size=14, symbol='x', line=dict(width=2, color='black')),
                name='Punto Seleccionado'
            ))

        fig.update_layout(
            title="Señal Temporal de Aceleración con Marcador de Evento",
            xaxis_title="Tiempo (segundos)",
            yaxis_title="Aceleración (g)",
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: ESPECTRO DE FRECUENCIA (FFT)
# ==========================================
with tab2:
    st.header("Análisis en el Dominio de la Frecuencia (FFT)")
    st.markdown("""
    La **Transformada Rápida de Fourier (FFT)** descompone la señal en sus componentes de frecuencia.
    Permite identificar a qué ritmo/velocidad está vibrando el equipo (desbalance, desalineación, fallas de baleros).
    """)

    axis_to_fft = st.selectbox("Selecciona el eje para calcular la FFT:", ['acc_x', 'acc_y', 'acc_z', 'acc_mag'])
    
    # Cálculo de la FFT usando SciPy
    N = len(df)
    T = 1.0 / fs
    yf = fft(df[axis_to_fft].values)
    xf = fftfreq(N, T)[:N//2]
    amplitude = 2.0/N * np.abs(yf[0:N//2])

    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(
        x=xf, 
        y=amplitude, 
        mode='lines', 
        name=f"FFT {axis_to_fft.upper()}",
        line=dict(color='#00CC96')
    ))

    fig_fft.update_layout(
        title=f"Espectro de Frecuencia para {axis_to_fft.upper()}",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="Amplitud / Magnitud",
        height=450
    )

    st.plotly_chart(fig_fft, use_container_width=True)

    # Guía rápida de interpretación
    with st.expander("📖 ¿Cómo interpretar este gráfico de FFT?"):
        st.markdown("""
        * **Pico Dominante en Baja Frecuencia (1X RPM):** Indica típicamente **Desbalance Mecánico** (e.g. un ventilador o polea sucia/descentrada).
        * **Pico en el Doble de Frecuencia (2X RPM):** Indica **Desalineación** entre el motor y la carga acoplada.
        * **Múltiples Armónicos (3X, 4X, 5X...):** Indica **Holgura o Juego Mecánico** (pernos flojos, desgaste de bujes).
        * **Ruido/Picos en Altas Frecuencias (>100 Hz):** Indica **Fricción metal-metal**, daño en pistas o baleros de rodamientos.
        """)
