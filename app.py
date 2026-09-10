import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CÁLCULO DE MAGNITUD Y DETECCIÓN DE UMBRALES POR EJE ---
def detect_threshold_events(df, threshold_value):
    """
    Evalúa cada eje y la magnitud total para identificar en qué EJE se superó el umbral.
    """
    # Cálculo de acc_mag si no existe
    if 'acc_mag' not in df.columns and all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

    events = []
    axes = [c for c in ['acc_x', 'acc_y', 'acc_z', 'acc_mag'] if c in df.columns]

    for axis in axes:
        # Se evalúa el valor absoluto para detectar sobrepasos tanto positivos como negativos
        exceeded = df[df[axis].abs() > threshold_value]
        for idx, row in exceeded.iterrows():
            events.append({
                'Índice/Timestamp': row.get('timestamp', idx),
                'Eje Afectado': axis.replace('acc_', '').upper(), # Muestra X, Y, Z o MAG
                'Valor Medido': round(row[axis], 3),
                'Umbral Configurado': threshold_value,
                'Columna_Raw': axis
            })
            
    return pd.DataFrame(events)

# --- PESTAÑA DE THRESHOLDS ---
def render_thresholds_tab(df, default_threshold=2.0):
    st.header("⚡ Análisis e Historial de Umbrales (Thresholds)")

    col1, col2 = st.columns([1, 3])

    with col1:
        threshold = st.number_input(
            "Definir Umbral Límite", 
            value=float(default_threshold), 
            step=0.5,
            help="Cualquier lectura (|valor|) que supere este límite será registrada."
        )

    # Detectar eventos
    events_df = detect_threshold_events(df, threshold)

    if events_df.empty:
        st.success(f"✅ No se detectaron lecturas que superen el umbral de {threshold}.")
        return

    st.warning(f"⚠️ Se detectaron **{len(events_df)}** eventos que superaron el umbral.")

    # Renderizar la tabla de eventos con modo de selección
    st.subheader("Selecciona un evento de la tabla para ubicarlo en la gráfica:")
    
    event_selection = st.dataframe(
        events_df[['Índice/Timestamp', 'Eje Afectado', 'Valor Medido', 'Umbral Configurado']],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single_row",
        key="threshold_table"
    )

    # Determinar si el usuario hizo clic en una fila
    selected_time = None
    selected_axis = None
    
    selected_rows = event_selection.get("selection", {}).get("rows", [])
    if selected_rows:
        row_idx = selected_rows[0]
        selected_event = events_df.iloc[row_idx]
        selected_time = selected_event['Índice/Timestamp']
        selected_axis = selected_event['Columna_Raw']
        st.info(f"📌 Evento seleccionado: **Eje {selected_event['Eje Afectado']}** en el tiempo/índice **{selected_time}** (Valor: {selected_event['Valor Medido']})")

    # --- GRAFICAR LA SEÑAL CON LA UBICACIÓN DEL EVENTO ---
    fig = go.Figure()

    # Trazar ejes
    time_col = 'timestamp' if 'timestamp' in df.columns else df.index
    for axis, color in [('acc_x', 'red'), ('acc_y', 'green'), ('acc_z', 'blue'), ('acc_mag', 'purple')]:
        if axis in df.columns:
            fig.add_trace(go.Scatter(
                x=time_col, 
                y=df[axis], 
                mode='lines', 
                name=axis.upper(),
                opacity=0.4 if selected_axis and axis != selected_axis else 1.0 # Resalta el eje seleccionado
            ))

    # Agregar líneas límites de umbral
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="+Umbral")
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red", annotation_text="-Umbral")

    # Si hay una fila seleccionada, colocar marca puntual y vline
    if selected_time is not None:
        fig.add_vline(
            x=selected_time, 
            line_width=2, 
            line_dash="dot", 
            line_color="yellow",
            annotation_text=f" Evento {selected_axis.upper()}"
        )
        
        # Obtener el punto exacto para resaltar con un marcador
        y_val = df.loc[df['timestamp'] == selected_time, selected_axis].values[0] if 'timestamp' in df.columns else df.loc[selected_time, selected_axis]
        
        fig.add_trace(go.Scatter(
            x=[selected_time],
            y=[y_val],
            mode='markers',
            marker=dict(color='yellow', size=14, symbol='x'),
            name='Punto Seleccionado'
        ))

    fig.update_layout(
        title="Ubicación del Evento en la Señal Temporal",
        xaxis_title="Tiempo / Timestamp",
        yaxis_title="Aceleración",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
