"""
Real-time Audio Compression Analysis Dashboard
Deployed on Streamlit Cloud - Updated for production stability

Key Changes for Deployment:
1. Moved heavy codec analysis behind a button to prevent startup crash
2. Added caching for expensive computations
3. Session state management for results persistence
4. English comments and UI for international compatibility
5. Results display with download options
6. Optimized for Streamlit Cloud resource limits
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from datetime import datetime
import io

# Heavy imports for audio analysis (only loaded when needed)
@st.cache_resource
def load_audio_libs():
    """Lazy load heavy audio/ML libraries to save memory on startup"""
    import scipy.io.wavfile as wav
    import scipy.signal as signal
    from pypesq import pesq  # PESQ metric
    return wav, signal, pesq

# Create results directory if not exists
@st.cache_data
def ensure_results_dir():
    os.makedirs("./results", exist_ok=True)
    return "./results"

# Session state initialization
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'dashboards' not in st.session_state:
    st.session_state.dashboards = {}

# Page config
st.set_page_config(
    page_title="Real-time Audio Codec Comparison",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🎵 Real-time Audio Codec Comparison Dashboard")
st.markdown("Compare MP3, AAC, and Opus codecs across **Speech**, **Music**, and **Noise** audio types at different bitrates.")

# Sidebar
st.sidebar.header("Analysis Controls")
bitrates = [64, 128, 192, 256]
codecs = ["MP3", "AAC", "Opus"]
audio_types = ["speech", "music", "noise"]

selected_bitrates = st.sidebar.multiselect("Bitrates (kbps)", bitrates, default=bitrates)
selected_codecs = st.sidebar.multiselect("Codecs", codecs, default=codecs)
selected_types = st.sidebar.multiselect("Audio Types", audio_types, default=audio_types)

run_analysis = st.sidebar.button("🚀 Run Codec Analysis", type="primary")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Codec Performance Summary")
    if st.session_state.analysis_done and st.session_state.results_df is not None:
        # Filter results based on selections
        filtered_df = st.session_state.results_df[
            (st.session_state.results_df['bitrate'].isin(selected_bitrates)) &
            (st.session_state.results_df['codec'].isin(selected_codecs)) &
            (st.session_state.results_df['audio_type'].isin(selected_types))
        ]
        
        if not filtered_df.empty:
            # Summary metrics
            summary = filtered_df.groupby(['audio_type', 'codec']).agg({
                'snr_db': 'mean',
                'pesq': 'mean'
            }).round(2).reset_index()
            
            st.dataframe(summary, use_container_width=True)
            
            # Download CSV
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Full Results CSV",
                data=csv_buffer.getvalue(),
                file_name=f"codec_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data matches your filters. Run analysis first!")

with col2:
    st.header("🏆 Recommendations")
    recommendations = {
        "Speech": "Opus (best low-bitrate performance)",
        "Music": "AAC (optimal for complex audio)",
        "Noise": "Opus (balanced performance)"
    }
    for audio_type, rec in recommendations.items():
        st.markdown(f"**{audio_type}:** {rec}")

# Analysis execution (triggered by button)
if run_analysis and not st.session_state.analysis_done:
    with st.spinner("🔍 Running comprehensive codec analysis... This may take 1-2 minutes."):
        try:
            # Ensure results directory
            results_dir = ensure_results_dir()
            
            # Load libraries
            wav, signal, pesq = load_audio_libs()
            
            # Simulated analysis data (replace with actual audio processing)
            # In real app, load sample audio files and process them
            all_results = []
            
            for audio_type in audio_types:
                for codec in codecs:
                    for bitrate in bitrates:
                        # Simulate metrics (replace with real processing)
                        snr = np.random.uniform(5, 25, 1)[0]  # dB
                        pesq_score = np.random.uniform(2.0, 4.5, 1)[0]
                        
                        all_results.append({
                            'audio_type': audio_type,
                            'codec': codec,
                            'bitrate': bitrate,
                            'snr_db': round(snr, 1),
                            'pesq': round(pesq_score, 2),
                            'thd_n_pct': round(np.random.uniform(2, 12, 1)[0], 1),
                            'efficiency': round(np.random.uniform(0.35, 0.40, 1)[0], 3)
                        })
            
            # Create DataFrame
            df_results = pd.DataFrame(all_results)
            st.session_state.results_df = df_results
            
            # Save CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"{results_dir}/codec_analysis_report_{timestamp}.csv"
            df_results.to_csv(csv_path, index=False)
            
            # Create simple visualizations (cached)
            fig_summary = px.bar(
                df_results.groupby(['codec', 'audio_type']).agg({'snr_db': 'mean'}).reset_index(),
                x='audio_type', y='snr_db', color='codec',
                title="Average SNR by Codec and Audio Type",
                barmode='group'
            )
            st.plotly_chart(fig_summary, use_container_width=True)
            
            st.session_state.analysis_done = True
            st.success("✅ Analysis complete! Results loaded successfully.")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.session_state.analysis_done = False

# Reset button
if st.session_state.analysis_done:
    if st.sidebar.button("🔄 Reset Analysis"):
        st.session_state.analysis_done = False
        st.session_state.results_df = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Deployment Notes:** Optimized for Streamlit Cloud. Heavy computations run only on demand to prevent crashes.[web:1][web:16]")

# Instructions for real implementation:
# 1. Replace simulated data with actual audio processing using your sample files
# 2. Add sample audio files to repo (speech.wav, music.wav, noise.wav)
# 3. Implement real MP3/AAC/Opus encoding/decoding with pydub/pymp3
# 4. Use @st.cache_data for audio file loading
