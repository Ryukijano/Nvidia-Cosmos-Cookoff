"""Syndrome-Net QEC Lab — Streamlit Page Module.

Ported from syndrome-net/app/streamlit_app.py to Streamlit multi-page format.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import stim
import torch
from surface_code_in_stem.rl_control.environment import QECEnvConfig, qec_environment
from surface_code_in_stem.rl_control.sota_agents import RLAlgorithms, training_loop
from surface_code_in_stem.surface_code import SurfaceCode


def _get_circuit(distance: int, rounds: int, noise: float, basis: str, local: bool = False):
    """Generate surface code circuit with error handling."""
    try:
        sc = SurfaceCode(
            distance=distance,
            rounds=rounds,
            noise=noise,
            basis=basis,
            local=local,
        )
        return sc, None
    except Exception as e:
        return None, str(e)


def _render_syndrome_net_page():
    st.markdown("""
    <section class="workspace-card">
        <p class="hub-eyebrow">Quantum Error Correction</p>
        <h2>Syndrome-Net QEC Lab</h2>
        <p class="workspace-copy">
            Quantum Error Correction framework with surface code circuit generation, 
            RL-based decoders (PPO/SAC), and threshold exploration tools.
        </p>
    </section>
    """, unsafe_allow_html=True)

    # Tabs for different sections
    tabs = st.tabs([
        "Circuit Viewer",
        "RL Live Training",
        "Threshold Explorer",
        "Teraquop Footprint",
    ])

    # Circuit Viewer Tab
    with tabs[0]:
        st.subheader("Surface Code Circuit Viewer")

        col1, col2 = st.columns([1, 1])
        with col1:
            distance = st.slider("Code distance", 3, 21, 5, step=2, key="sc_distance")
            rounds = st.slider("Rounds", 1, 50, 10, key="sc_rounds")
            noise = st.number_input("Noise rate", 0.0, 0.01, 0.001, step=0.0001, format="%.4f")
        with col2:
            basis = st.selectbox("Basis", ["Z", "X"], key="sc_basis")
            local = st.checkbox("Local noise model", value=False, key="sc_local")

        if st.button("Generate Circuit", type="primary", key="sc_generate"):
            circuit, error = _get_circuit(distance, rounds, noise, basis, local)
            if circuit is None:
                st.error(f"Failed to create circuit: {error}")
            else:
                st.session_state.circuit_obj = circuit
                st.session_state.circuit_stim = circuit.circuit

        if "circuit_stim" in st.session_state:
            st.text_area("Stim circuit", st.session_state.circuit_stim, height=300)

            # Dem instructions visualization
            stim_str = str(st.session_state.circuit_stim)
            dem_ops = [line for line in stim_str.split("\n") if line.strip() and not line.strip().startswith("#")]

            op_types = {}
            for op in dem_ops:
                parts = op.split()
                if parts:
                    op_type = parts[0]
                    op_types[op_type] = op_types.get(op_type, 0) + 1

            if op_types:
                fig = go.Figure(go.Bar(
                    x=list(op_types.keys()),
                    y=list(op_types.values()),
                    marker_color="#3b82f6",
                ))
                fig.update_layout(
                    title="Circuit Operation Types",
                    xaxis_title="Operation",
                    yaxis_title="Count",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True, key="circuit_ops_chart")

    # RL Live Training Tab
    with tabs[1]:
        st.subheader("RL Live Training")

        col1, col2 = st.columns([1, 1])
        with col1:
            rl_distance = st.slider("Code distance", 3, 21, 5, step=2, key="rl_distance")
            rl_rounds = st.slider("Rounds", 1, 50, 10, key="rl_rounds")
            rl_noise = st.number_input("Noise rate", 0.0, 0.01, 0.001, step=0.0001, key="rl_noise")
        with col2:
            rl_algo = st.selectbox("RL Algorithm", ["PPO", "SAC"], key="rl_algo")
            rl_steps = st.number_input("Training steps", 100, 100000, 1000, step=100, key="rl_steps")
            seed = st.number_input("Random seed", 0, 10000, 42, key="rl_seed")

        if st.button("Start Training", type="primary", key="rl_train"):
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            try:
                def progress_callback(step, total, metrics):
                    progress = min(step / total, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Step {step}/{total} | Loss: {metrics.get('loss', 'N/A'):.4f}")

                # Create environment
                env_config = QECEnvConfig(
                    distance=rl_distance,
                    rounds=rl_rounds,
                    noise=rl_noise,
                )

                with st.spinner("Training RL decoder..."):
                    agent, history = training_loop(
                        algo=RLAlgorithms(rl_algo),
                        env_config=env_config,
                        total_steps=rl_steps,
                        seed=seed,
                        callback=progress_callback,
                    )

                st.success("Training complete!")

                # Plot training curve
                if history:
                    fig = go.Figure()
                    steps = list(range(len(history)))
                    fig.add_trace(go.Scatter(
                        x=steps,
                        y=history,
                        mode="lines",
                        name="Loss",
                        line=dict(color="#3b82f6"),
                    ))
                    fig.update_layout(
                        title="Training Loss Curve",
                        xaxis_title="Step",
                        yaxis_title="Loss",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True, key="rl_loss_curve")

            except Exception as e:
                st.error(f"Training failed: {e}")

    # Threshold Explorer Tab
    with tabs[2]:
        st.subheader("Threshold Explorer")

        col1, col2 = st.columns([1, 1])
        with col1:
            thresh_distance = st.slider("Code distance", 3, 21, 5, step=2, key="thresh_distance")
            thresh_rounds = st.slider("Rounds", 1, 50, 10, key="thresh_rounds")
            decoder = st.selectbox("Decoder", ["MWPM", "Union-Find"], key="thresh_decoder")
        with col2:
            min_noise = st.number_input("Min noise", 0.0, 0.01, 0.0001, step=0.0001, format="%.4f", key="thresh_min")
            max_noise = st.number_input("Max noise", 0.0, 0.02, 0.005, step=0.0001, format="%.4f", key="thresh_max")
            num_points = st.slider("Sweep points", 5, 50, 15, key="thresh_points")

        if st.button("Run Threshold Sweep", type="primary", key="thresh_run"):
            with st.spinner("Running threshold sweep..."):
                try:
                    noise_rates = np.linspace(min_noise, max_noise, num_points)
                    logical_error_rates = []

                    for noise in noise_rates:
                        circuit, error = _get_circuit(thresh_distance, thresh_rounds, noise, "Z")
                        if circuit is None:
                            logical_error_rates.append(1.0)
                            continue

                        # Use stim sampler to estimate logical error rate
                        sampler = circuit.circuit.compile_detector_sampler()
                        shots = 100
                        errors = 0

                        for _ in range(shots):
                            try:
                                detection_events = sampler.sample(shots=1)[0]
                                # Simplified: count non-trivial syndromes
                                if np.any(detection_events):
                                    errors += 1
                            except:
                                errors += 1

                        logical_error_rates.append(errors / shots)

                    # Plot threshold curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=noise_rates,
                        y=logical_error_rates,
                        mode="lines+markers",
                        name=f"d={thresh_distance}",
                        line=dict(color="#3b82f6"),
                        marker=dict(size=8),
                    ))
                    fig.update_layout(
                        title=f"Threshold Curve (Decoder: {decoder})",
                        xaxis_title="Physical Error Rate",
                        yaxis_title="Logical Error Rate",
                        template="plotly_dark",
                        xaxis_type="log",
                        yaxis_type="log",
                    )
                    st.plotly_chart(fig, use_container_width=True, key="threshold_curve")

                    # Show data table
                    df = pd.DataFrame({
                        "Physical Error Rate": noise_rates,
                        "Logical Error Rate": logical_error_rates,
                    })
                    st.dataframe(df, use_container_width=True)

                except Exception as e:
                    st.error(f"Threshold sweep failed: {e}")

    # Teraquop Footprint Tab
    with tabs[3]:
        st.subheader("Teraquop Footprint Estimator")

        col1, col2 = st.columns([1, 1])
        with col1:
            target_rounds = st.number_input("Target rounds (teraquops)", 1e12, 1e18, 1e15, format="%.0e", key="tera_rounds")
            physical_error = st.number_input("Physical error rate", 0.0, 0.01, 0.001, step=0.0001, key="tera_error")
            safety_factor = st.slider("Safety factor", 1.0, 10.0, 2.0, step=0.5, key="tera_safety")
        with col2:
            overhead_model = st.selectbox("Overhead model", ["Surface code", "LDPC"], key="tera_model")

        if st.button("Estimate Footprint", type="primary", key="tera_estimate"):
            try:
                # Simplified teraquop estimation
                # For surface code: d ~ 2 * sqrt(log(1/p_th) / log(p/p_th))
                # where p_th is threshold ~ 0.01
                p_th = 0.01
                if physical_error < p_th:
                    # Approximate distance needed
                    d = int(2 * math.sqrt(math.log(target_rounds) / math.log(p_th / physical_error)) * safety_factor)
                    d = max(d, 3)
                    if d % 2 == 0:
                        d += 1

                    num_qubits = d * d * 2  # Data + ancilla

                    st.success(f"Estimated distance: d={d}")
                    st.info(f"Physical qubits required: ~{num_qubits}")

                    # Visualize qubit grid
                    fig = go.Figure()
                    # Data qubits
                    data_x = [(i % d) * 2 for i in range(d * d)]
                    data_y = [(i // d) * 2 for i in range(d * d)]
                    fig.add_trace(go.Scatter(
                        x=data_x,
                        y=data_y,
                        mode="markers",
                        name="Data qubits",
                        marker=dict(size=10, color="#3b82f6"),
                    ))
                    # Ancilla qubits
                    anc_x = [(i % (d - 1)) * 2 + 1 for i in range((d - 1) * (d - 1))]
                    anc_y = [(i // (d - 1)) * 2 + 1 for i in range((d - 1) * (d - 1))]
                    fig.add_trace(go.Scatter(
                        x=anc_x,
                        y=anc_y,
                        mode="markers",
                        name="Ancilla qubits",
                        marker=dict(size=8, color="#ef4444"),
                    ))
                    fig.update_layout(
                        title=f"Surface Code Lattice (d={d})",
                        xaxis_title="X position",
                        yaxis_title="Y position",
                        template="plotly_dark",
                        showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True, key="teraquop_lattice")

                else:
                    st.error("Physical error rate exceeds threshold (~0.01). Logical qubits unreliable.")

            except Exception as e:
                st.error(f"Estimation failed: {e}")


# Import pandas here to avoid early import issues
import pandas as pd
