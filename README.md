# Faster Whisper Benchmark Tool

A benchmarking tool for evaluating the performance of Whisper models using the [faster-whisper](https://github.com/guillaumekln/faster-whisper) library. With this script, you can easily evaluate CPU and GPU inference performance across multiple audio files.

---

## 🔧 Features

* Benchmark across:
  * Multiple audio files
  * CPU and GPU devices
  * Various compute precisions (float32, float16, bfloat16, int8)
* Capture metrics:
  * Inference time
  * Real-time factor (RTF)
  * CPU/RAM usage
  * GPU utilization, memory, and power (if available)
  * Disk read/write I/O (WIP)
  * CPU power draw (WIP)
* Multi-threaded parallel execution
* Generates:
  * JSON summary reports
  * Per-interval detailed CSV/JSON files
  * Optional CLI tables and Markdown/CSV tables

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/gkunstman/faster-whisper-benchmark.git
cd faster-whisper-benchmark
```
### 2. Create a Virtual Environment (Optional)

```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

Basic Example:

```bash
python3 faster_whisper_benchmark.py \
  --mode batch \
  --audio_dir ./test_audio \
  --model_size small.en \
  --devices cpu \
  --precisions float32 int8 \
  --threads 4 \
  --sort_by inference_time \
  --export_csv --export_md \
  --output_dir ./results \
  --detailed_csv --detailed_json
```
CLI Options

* `--mode`: single_file or batch (default: single_file)
* `--audio_file`: Path to single audio file (required for single_file mode)
* `--audio_dir`: Path to directory of audio files (required for batch mode)
* `--model_size`: Whisper model to benchmark (tiny.en, small.en, medium.en, large-v2, etc.)
* `--devices`: List of devices to test: cpu, cuda, cuda:0, etc. (defaults to auto-detect)
* `--precisions`: Compute precisions to test: float32, float16, bfloat16, int8
* `--threads`: Number of CPU threads (applies to CPU only)
* `--monitor_interval`: Sampling interval for system stats (default: 0.1 seconds)
* `--export_csv`: Export CLI summary results to CSV
* `--export_md`: Export CLI summary results to Markdown
* `--detailed_csv`: Export detailed interval metrics to CSV
* `--detailed_json`: Export detailed interval metrics to JSON
* `--output_dir`: Directory to store outputs (default: .)
* `--sort_by`: filename, inference_time, or real_time_factor
* `--quiet`: Suppress CLI output (only saves to disk)
* `--debug`: Enable debug messages and warnings

---

## 📊 Output

* *.json: Full benchmark summary with metrics
* *-detailed.csv/json: Interval-by-interval telemetry (CPU%, memory, GPU, etc.)
* *.md: Markdown table of results
* *.csv: CLI-style results as CSV
* *-warnings.txt, *-infos.txt: Any warnings or environment notes

---

## ⚠️ Notes on Power & Disk I/O (WIP)

* CPU Power Monitoring:
  + This uses energy_uj from /sys/class/powercap on Intel CPUs. Availability varies by platform.
  + AMD EPYC and ARM CPUs may not report power through this interface. Results may be null.
* Disk I/O Monitoring:
  + Currently limited due to inconsistent read tracking; write bytes appear accurate. Further improvements are planned.

---

## 🐛 Known Issues

* CPU power and disk I/O metrics may return null depending on hardware/platform access.
* Inference metrics may vary slightly depending on CPU scaling and background tasks.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change or improve.

---

## 📄 License

MIT License
```
I made some minor formatting suggestions and added a brief introduction at the top of the README. I also used consistent headings throughout the document. Let me know if you'd like any further changes!
