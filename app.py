# ---------------------------------------------------------
# MÓDULO FÍSICO: ANÁLISIS DE DESLIZAMIENTO (µs = 0.28)
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧲 Parámetros de Fricción")
mu_s = st.sidebar.number_input("Coeficiente de Fricción Estático (µs):", value=0.28, step=0.01)
safety_factor = st.sidebar.slider("Factor de Seguridad (%):", 50, 100, 80) / 100.0

# Asumiendo columnas estándar de Aceleración en 'g'
if all(col in plot_df.columns for col in ['AccX(g)', 'AccY(g)', 'AccZ(g)']):
    # 1. Aceleración Horizontal Resultante (Plano X-Y)
    plot_df['Acc_Horiz'] = np.sqrt(plot_df['AccX(g)']**2 + plot_df['AccY(g)']**2)
    
    # 2. Aceleración Normal Efectiva (Gravedad + Componente Dinámica Z)
    # Si AccZ incluye la gravedad (~1.0g), usamos el valor directo; si no, sumamos 1.0
    acc_z_raw = plot_df['AccZ(g)'].values
    z_has_gravity = np.mean(acc_z_raw) > 0.5
    normal_g = acc_z_raw if z_has_gravity else (1.0 + acc_z_raw)
    normal_g = np.maximum(0.01, normal_g) # Evitar división por cero si despega
    
    # 3. Índice de Riesgo de Deslizamiento R(t)
    plot_df['Slip_Risk_Ratio'] = plot_df['Acc_Horiz'] / normal_g
    
    # 4. Umbral Dinámico Permitido instante a instante
    plot_df['Acc_Max_Allowed'] = mu_s * normal_g
    
    # Detección de Eventos de Deslizamiento Físico
    slip_events_mask = plot_df['Slip_Risk_Ratio'] >= mu_s
    warning_events_mask = (plot_df['Slip_Risk_Ratio'] >= (mu_s * safety_factor)) & (~slip_events_mask)
    
    total_slips = slip_events_mask.sum()
    peak_risk = plot_df['Slip_Risk_Ratio'].max()
    margin_left = ((mu_s - peak_risk) / mu_s) * 100
    
    # Mostrar Métricas en el Panel
    st.markdown("### 🧲 Estado Crítico de Fricción y Deslizamiento")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coeficiente µs", f"{mu_s:.2f}")
    c2.metric("Riesgo Pico (R_max)", f"{peak_risk:.3f}", f"{'CRÍTICO' if peak_risk >= mu_s else 'OK'}")
    c3.metric("Margen de Seguridad", f"{margin_left:.1f}%")
    c4.metric("Deslizamientos Detectados", f"{total_slips} picos", delta_color="inverse")
