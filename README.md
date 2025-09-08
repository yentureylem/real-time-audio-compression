# Real-time Audio Compression Analysis

A comprehensive tool for analyzing audio codec performance with realistic metrics and interactive visualizations.

## Features

- **Multi-codec Support**: MP3, AAC, and Opus codec analysis
- **Realistic Metrics**: SNR, PESQ, and THD+N calculations based on actual codec performance
- **Multiple Content Types**: Speech, music, and noise audio analysis
- **Interactive Dashboards**: Plotly-based visualizations
- **Comprehensive Reports**: CSV exports and detailed summaries
- **Real-time Processing**: Optimized for performance analysis

## Metrics Analyzed

- **SNR (Signal-to-Noise Ratio)**: Audio quality measurement in dB
- **PESQ (Perceptual Evaluation of Speech Quality)**: Perceptual quality score (1-5)
- **THD+N (Total Harmonic Distortion + Noise)**: Distortion percentage
- **File Size**: Compression efficiency
- **Processing Time**: Encode/decode performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-compression-analysis.git
cd audio-compression-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python app.py
```

## Dependencies

- numpy
- matplotlib
- plotly
- pandas
- scipy (optional, for advanced signal processing)

## Usage

### Basic Usage

```python
from app import AudioCompressionAnalyzer

# Initialize analyzer
analyzer = AudioCompressionAnalyzer()

# Run complete benchmark
results = analyzer.run_benchmark()

# Generate reports
analyzer.generate_summary_report(results)
```

### Custom Analysis

```python
# Analyze specific audio type
analyzer = AudioCompressionAnalyzer()
speech_audio = analyzer.generate_test_audio('speech')
compressed = analyzer.simulate_compression(speech_audio, 'Opus', 128)

# Calculate metrics
snr = analyzer.metrics.calculate_snr_realistic(speech_audio, compressed, 'Opus', 128)
print(f"SNR: {snr:.1f} dB")
```

## Results

The tool generates:

1. **Interactive HTML Dashboards** (`./results/codec_analysis_[type].html`)
2. **CSV Reports** (`./results/codec_analysis_report_[timestamp].csv`)
3. **Console Output** with real-time metrics

### Sample Results

| Audio Type | Codec | Avg SNR (dB) | Avg PESQ | Avg THD+N (%) |
|------------|-------|--------------|----------|---------------|
| Speech     | MP3   | 18.2         | 3.25     | 4.2           |
| Speech     | AAC   | 20.1         | 3.48     | 3.1           |
| Speech     | Opus  | 21.8         | 3.65     | 2.8           |
| Music      | MP3   | 16.9         | 3.15     | 4.8           |
| Music      | AAC   | 19.3         | 3.52     | 3.4           |
| Music      | Opus  | 20.2         | 3.41     | 3.2           |

## Codec Recommendations

- **Speech**: Opus (best low-bitrate performance)
- **Music**: AAC (optimal for complex audio)
- **Noise**: MP3 (most predictable performance)
- **Low bitrate**: Opus
- **High quality**: AAC
- **Compatibility**: MP3

## Project Structure

```
audio-compression-analysis/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── results/             # Generated outputs
│   ├── *.html          # Interactive dashboards
│   └── *.csv           # Data reports
└── examples/           # Usage examples
    └── basic_usage.py
```

## Technical Details

### Audio Generation
The tool generates synthetic test signals for different content types:
- **Speech**: Combined sinusoids with envelope decay
- **Music**: Multi-harmonic complex tones
- **Noise**: Gaussian white noise

### Compression Simulation
Realistic compression artifacts are modeled based on:
- Codec-specific quality profiles
- Bitrate-dependent noise injection
- Content-adaptive processing

### Quality Metrics
- **SNR**: Calculated with codec-specific adjustments
- **PESQ**: Modeled based on perceptual correlation
- **THD+N**: Content and bitrate dependent modeling

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- Thanks to the audio compression community
- Inspired by real-world codec performance studies
- Built with modern Python data science stack

⭐ If you find this project useful, please give it a star!
