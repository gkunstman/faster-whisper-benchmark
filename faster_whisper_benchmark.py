# Multi-device, multi-precision benchmarking script with descriptive output filenames

from faster_whisper import WhisperModel
import os
import time
import json
import psutil
import datetime
import threading
import subprocess
import platform
import argparse
import csv
from tabulate import tabulate
import re
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import pynvml

console = Console()

# ------------------------
# Python Version Check
# ------------------------

if not (sys.version_info.major == 3 and 9 <= sys.version_info.minor <= 12):
    print(f"❌ This script requires Python 3.9, 3.10, 3.11, or 3.12. Detected: {sys.version}")
    sys.exit(1)

# ------------------------
# ffprobe Availability Check
# ------------------------

def check_ffprobe():
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except Exception:
        print("❌ 'ffprobe' is required but not found. Please install FFmpeg package.")
        sys.exit(1)

check_ffprobe()


# Record the global start time
benchmark_start_time = time.perf_counter()

warnings_list = []
info_list = []

if __name__ == "__main__":
    try:

        ### WRAPPED CODE START ###

        # ------------------------
        # Argument Parsing
        # ------------------------

        parser = argparse.ArgumentParser(
            description="Faster-Whisper Benchmarking Tool - Benchmark whisper models across CPU, GPU, precisions, and audio files."
        )

        # ------------------------
        # Main Options
        # ------------------------

        main_group = parser.add_argument_group("Main Options")

        main_group.add_argument("--mode", choices=["single_file", "batch"], default="single_file",
            help="Run a single file or a batch of files in a directory.")
        main_group.add_argument("--audio_file", type=str,
            help="Path to a single audio file for benchmarking (required in single_file mode).")
        main_group.add_argument("--audio_dir", type=str,
            help="Path to directory containing audio files (required in batch mode).")
        main_group.add_argument("--model_size", type=str, default="medium.en",
            help="Name of whisper model to use (e.g., tiny.en, small.en, medium.en, large-v2).")

        # ------------------------
        # Precision and Device Options
        # ------------------------

        precision_device_group = parser.add_argument_group("Precision and Device Options")

        precision_device_group.add_argument("--precisions", nargs="+", default=["float32", "float16", "bfloat16", "int8"],
            help="Precisions to benchmark: float32, float16, bfloat16, int8.")
        precision_device_group.add_argument("--devices", nargs="+", default=None,
            help="Devices to benchmark: cpu, cuda, cuda:0, cuda:1, etc. (auto-detects if omitted).")
        precision_device_group.add_argument("--threads", type=int,
            help="Number of CPU threads to use (applies to CPU benchmarking only).")
        precision_device_group.add_argument("--retries", type=int, default=1,
            help="Number of retries for model loading or inference failures.")
        precision_device_group.add_argument(
            "--monitor_interval",
            type=float,
            default=0.1,
            help="Interval (seconds) between CPU/RAM monitoring samples. Default is 0.1 seconds."
        )


        # ------------------------
        # Output and Export Options
        # ------------------------

        export_group = parser.add_argument_group("Output and Export Options")

        export_group.add_argument("--output_dir", type=str, default=".",
            help="Directory to save benchmark results (JSON, CSV, MD).")
        export_group.add_argument("--no_cli_table", action="store_true",
            help="Suppress CLI table output (useful for scripting).")
        export_group.add_argument("--export_csv", action="store_true",
            help="Export benchmark results to CSV file(s).")
        export_group.add_argument("--export_md", action="store_true",
            help="Export benchmark results to Markdown file(s).")
        export_group.add_argument("--sort_by", choices=["filename", "inference_time", "real_time_factor"], default="filename",
            help="Sort CLI tables and exports by filename, inference time, or real-time factor.")
        export_group.add_argument("--detailed_csv", action="store_true",
            help="Export per-sample interval data to CSV.")
        export_group.add_argument("--detailed_json", action="store_true",
            help="Export per-sample interval data to JSON.")

        parser.add_argument("--quiet", action="store_true",
            help="Run benchmark without CLI output (still saves results and logs).")
        parser.add_argument("--debug", action="store_true",
            help="Enable debug output for internal script details.")

        args = parser.parse_args()

        # If debug is enabled, force-disable quiet mode
        if args.debug and args.quiet:
            print("⚡ Debug mode active: overriding --quiet to allow verbose output.")
            args.quiet = False


        # ------------------------
        # Auto-detect and adjust devices
        # ------------------------

        if not args.devices:
            # No devices specified: auto-detect
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                devices_to_test = [f"cuda:{i}" for i in range(gpu_count)]
                info_list.append(f"Detected {gpu_count} GPU(s): {', '.join(devices_to_test)}")
            else:
                devices_to_test = ["cpu"]
                info_list.append("No GPU found, defaulting to CPU.")
        else:
            devices_to_test = args.devices

        # Optional: Validate device names
        for d in devices_to_test:
            if not (d == "cpu" or d.startswith("cuda")):
                raise ValueError(f"Invalid device '{d}'. Use 'cpu' or 'cuda' or 'cuda:0', 'cuda:1'...")

        # Check if CUDA devices are actually usable
        final_devices = []
        for d in devices_to_test:
            if d == "cpu":
                final_devices.append(d)
            elif d.startswith("cuda"):
                try:
                    # Try to set device and run a dummy operation
                    dev_idx = int(d.split(":")[1]) if ":" in d else 0
                    if not torch.cuda.is_available():
                        warnings_list.append(f"Device {d} not available, skipping...")
                        continue
                    torch.cuda.set_device(dev_idx)
                    _ = torch.tensor([1.0], device=d)  # Try simple tensor
                    final_devices.append(d)
                except Exception as e:
                    warnings_list.append(f"Device {d} unusable: {e}")

        # Final devices to use
        devices_to_test = final_devices

        if not devices_to_test:
            print("❌ No valid devices found. Exiting.")
            exit(1)

        # ------------------------
        # Validation
        # ------------------------

        if args.mode == "single_file" and not args.audio_file:
            parser.error("--audio_file must be specified in single_file mode.")
        elif args.mode == "batch" and not args.audio_dir:
            parser.error("--audio_dir must be specified in batch mode.")

        # Additional Validation: Ensure audio directory exists
        if args.mode == "batch" and not os.path.isdir(args.audio_dir):
            print(f"❌ Audio directory '{args.audio_dir}' not found.")
            sys.exit(1)

        # Threading only for CPU
        if args.threads and "cpu" in args.devices:
            for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS"]:
                os.environ[var] = str(args.threads)

        model_size = args.model_size
        precisions_to_test = [p.lower() for p in args.precisions]
        import torch

        if not args.devices:
            # Auto-detect available devices
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                devices_to_test = [f"cuda:{i}" for i in range(gpu_count)]
                print(f"⚡ No --devices specified, detected {gpu_count} GPU(s): {', '.join(devices_to_test)}")
            else:
                devices_to_test = ["cpu"]
                print("⚡ No --devices specified, no GPU found: defaulting to cpu")
        else:
            devices_to_test = args.devices

        # Optional: Validate device names
        for d in devices_to_test:
            if not (d == "cpu" or d.startswith("cuda")):
                raise ValueError(f"Invalid device '{d}'. Use 'cpu' or 'cuda' or 'cuda:0', 'cuda:1'...")

        # Timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Identify short audio source name
        if args.mode == "single_file":
            base_audio_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        else:
            base_audio_name = os.path.basename(os.path.normpath(args.audio_dir))
        base_audio_name = re.sub(r"[^a-zA-Z0-9_-]", "_", base_audio_name)

        # ------------------------
        # Utility Functions
        # ------------------------

        def detect_cpu_flags():
            try:
                result = subprocess.run(["lscpu"], capture_output=True, text=True)
                output = result.stdout.lower()

                return {
                    "amx_tile": "amx_tile" in output,
                    "amx_bf16": "amx_bf16" in output,
                    "amx_int8": "amx_int8" in output,
                    "avx512": "avx512" in output
                }
            except Exception as e:
                warnings_list.append(f"Failed to detect CPU features: {e}")
                return {
                    "amx_tile": False,
                    "amx_bf16": False,
                    "amx_int8": False,
                    "avx512": False
                }

        def get_system_info():
            uname = platform.uname()
            cpu_model = next((line.split(":")[1].strip() for line in open("/proc/cpuinfo") if "model name" in line), "Unknown")
            return {
                "os": uname.system,
                "os_version": uname.version,
                "kernel_version": uname.release,
                "cpu_model": cpu_model,
                "cpu_physical_cores": psutil.cpu_count(logical=False),
                "cpu_logical_cores": psutil.cpu_count(logical=True),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            }

        def get_audio_info(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            try:
                res = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", path
                ], capture_output=True, text=True)
                duration_sec = float(res.stdout.strip())
            except Exception:
                duration_sec = 0.0
            return round(size_mb, 2), round(duration_sec, 2)

        def detect_cpu_power_sources():
            """
            Detect all RAPL or hwmon energy_uj paths on the system.
            Works for Intel CPUs exposing /sys/class/powercap/.../energy_uj.
            """
            sources = []
            base_path = "/sys/class/powercap"
            if not os.path.exists(base_path):
                return sources

            for root, dirs, files in os.walk(base_path):
                if "energy_uj" in files:
                    sources.append(os.path.join(root, "energy_uj"))
            return sources


        def read_energy_uj(source):
            try:
                with open(source, "r") as f:
                    return int(f.read().strip())
            except Exception:
                return None


        # ------------------------
        # Benchmark Function
        # ------------------------

        def benchmark_model(path, precision, device, cpu_flags):
            from threading import Thread
            from psutil import Process
            precision = precision.lower()
            
            if precision == "bfloat16" and not cpu_flags.get("amx_bf16") and device == "cpu":
                return {"audio_file_name": os.path.basename(path), "error": "AMX BF16 not available", "detailed_samples": interval_samples}

            attempt = 0
            last_exception = None

            while attempt < args.retries:
                attempt += 1
                try:
                    model = WhisperModel(model_size, device=device, compute_type=precision)

                    process = Process(os.getpid())
                    cpu_power_sources = detect_cpu_power_sources()
                    rapl_available = bool(cpu_power_sources)

                    cpu_samples = []
                    memory_samples = []
                    interval_samples = []

                    # Initialize NVML for GPU monitoring (if CUDA)
                    gpu_handle = None
                    if device.startswith("cuda"):
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            gpu_index = int(device.split(":")[1]) if ":" in device else 0
                            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                        except Exception as e:
                            gpu_handle = None
                            if args.debug:
                                warnings_list.append(f"Failed to initialize NVML for {device}: {e}")
                    
                    def monitor(start_time, duration_sec):
                        interval_index = 0

                        # Initialize CPU energy reading if available
                        if rapl_available:
                            monitor.prev_energy = [read_energy_uj(src) for src in cpu_power_sources]
                            monitor.prev_time = time.time()
                        else:
                            monitor.prev_energy = None
                            monitor.prev_time = None

                        while running:
                            cpu = psutil.cpu_percent(interval=args.monitor_interval)
                            mem = process.memory_info().rss / (1024 ** 2)  # in MB
                            now = time.perf_counter()
                            elapsed_sec = now - start_time
                            elapsed_ms = elapsed_sec * 1000
                            percent = (elapsed_sec / duration_sec * 100) if duration_sec else 0

                            # GPU stats
                            gpu_util = None
                            gpu_mem = None
                            gpu_power_watts = None

                            if gpu_handle:
                                try:
                                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                                    util_info = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                                    power_info = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)  # in milliwatts

                                    gpu_util = util_info.gpu
                                    gpu_mem = round(mem_info.used / (1024 ** 2), 2)
                                    gpu_power_watts = round(power_info / 1000, 2)  # convert to watts
                                except Exception as e:
                                    if args.debug:
                                        warnings_list.append(f"GPU monitor error: {e}")

                            # Disk IO stats
                            try:
                                if proc_io_start:
                                    proc_io_now = process.io_counters()
                                    read_bytes = proc_io_now.read_bytes - proc_io_start.read_bytes
                                    write_bytes = proc_io_now.write_bytes - proc_io_start.write_bytes
                                else:
                                    read_bytes = write_bytes = None
                            except Exception as e:
                                if args.debug:
                                    warnings_list.append(f"Error sampling process I/O: {e}")
                                read_bytes = write_bytes = None

                            # CPU power sampling (new)
                            cpu_power_watts = None
                            if rapl_available and monitor.prev_energy and monitor.prev_time:
                                try:
                                    energy_now = [read_energy_uj(src) for src in cpu_power_sources]
                                    delta_uj = sum((now_uj - start_uj if now_uj >= start_uj else (now_uj + (2**32) - start_uj))
                                                for now_uj, start_uj in zip(energy_now, monitor.prev_energy)
                                                if now_uj is not None and start_uj is not None)
                                    delta_s = time.time() - monitor.prev_time
                                    if delta_s > 0:
                                        cpu_power_watts = round((delta_uj / 1_000_000) / delta_s, 2)
                                    monitor.prev_energy = energy_now
                                    monitor.prev_time = time.time()
                                except Exception as e:
                                    if args.debug:
                                        warnings_list.append(f"CPU power read error: {e}")
                                    cpu_power_watts = None

                            interval_samples.append({
                                "interval_index": interval_index,
                                "elapsed_ms": round(elapsed_ms, 2),
                                "cpu_percent": cpu,
                                "memory_usage_mb": round(mem, 2),
                                "audio_duration_sec": round(duration_sec, 2),
                                "percent_complete": round(percent, 2),
                                "gpu_utilization_percent": gpu_util,
                                "gpu_memory_usage_mb": gpu_mem,
                                "gpu_power_watts": gpu_power_watts,
                                "disk_read_bytes": read_bytes,
                                "disk_write_bytes": write_bytes,
                                "cpu_power_watts": cpu_power_watts,
                            })

                            interval_index += 1
                    file_size, duration = get_audio_info(path)
                    start_time = time.perf_counter()
                    disk_start = psutil.disk_io_counters()
                    thread = Thread(target=monitor, args=(start_time, duration))  
                    running = True
                    # Record disk I/O *before* starting the thread
                    try:
                        proc_io_start = process.io_counters()
                    except Exception as e:
                        warnings_list.append(f"Failed to capture initial I/O counters: {e}")
                        proc_io_start = None
                    thread.start()
                    cpu_before = psutil.cpu_times()
                    mem_before = process.memory_info().rss
                    start = time.perf_counter()
                    segments, _ = model.transcribe(path)
                    transcript = " ".join([s.text for s in segments])
                    end = time.perf_counter()
                    running = False
                    thread.join()
                    cpu_after = psutil.cpu_times()
                    mem_after = process.memory_info().rss
                    
                    average_memory_usage_mb = round(sum(s["memory_usage_mb"] for s in interval_samples) / len(interval_samples), 2) if interval_samples else 0
                    peak_memory_usage_mb = round(max(s["memory_usage_mb"] for s in interval_samples), 2) if interval_samples else 0

                    cpu_power_samples = [s["cpu_power_watts"] for s in interval_samples if s.get("cpu_power_watts") is not None]
                    average_cpu_power_watts = round(sum(cpu_power_samples) / len(cpu_power_samples), 2) if cpu_power_samples else None
                    peak_cpu_power_watts = round(max(cpu_power_samples), 2) if cpu_power_samples else None

                    inference_ms = (end - start) * 1000
                    return {
                        "audio_file_name": os.path.basename(path),
                        "average_memory_usage_mb": average_memory_usage_mb,
                        "peak_memory_usage_mb": peak_memory_usage_mb,
                        "audio_file_size_mb": file_size,
                        "audio_duration_seconds": duration,
                        "inference_time_ms": round(inference_ms, 2),
                        "real_time_factor": round(inference_ms / (duration * 1000), 4) if duration else 0,
                        "peak_cpu_usage_percent": round(max(s["cpu_percent"] for s in interval_samples), 2) if interval_samples else 0,
                        "average_cpu_usage_percent": round(sum(s["cpu_percent"] for s in interval_samples) / len(interval_samples), 2) if interval_samples else 0,
                        "max_memory_usage_mb": round((mem_after - mem_before) / (1024 ** 2), 2),
                        "cpu_user_time_spent_sec": round(cpu_after.user - cpu_before.user, 2),
                        "cpu_system_time_spent_sec": round(cpu_after.system - cpu_before.system, 2),
                        "transcribed_text": transcript,
                        "detailed_samples": interval_samples,
                        "average_cpu_power_watts": average_cpu_power_watts,
                        "peak_cpu_power_watts": peak_cpu_power_watts,


                    }
                
                except Exception as e:
                    warnings_list.append(f"Attempt {attempt}/{args.retries} failed: {e}")
                    last_exception = e
                    time.sleep(2)  # Optional: small delay before retrying

            # After all retries fail
            return {"audio_file_name": os.path.basename(path), "error": str(last_exception)}


        # ------------------------
        # Main Execution
        # ------------------------

        cpu_flags = detect_cpu_flags()
        system_info = get_system_info()


        # Discover audio files
        if args.mode == "single_file":
            audio_files = [args.audio_file]
        else:
            audio_files = [os.path.join(args.audio_dir, f) for f in sorted(os.listdir(args.audio_dir))
                        if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))]

        # ------------------------
        # Debug
        # ------------------------
        if args.debug:
            print("\n🛠 Benchmark Configuration 🛠")
            print(f"- Mode: {args.mode}")
            print(f"- Model: {args.model_size}")
            print(f"- Audio files: {len(audio_files)} file(s)")

            # List first few files
            max_list = 5
            for f in audio_files[:max_list]:
                print(f"  - {os.path.basename(f)}")
            if len(audio_files) > max_list:
                print(f"  ... ({len(audio_files) - max_list} more files)")

            # Total audio duration
            try:
                total_duration_sec = 0.0
                for f in audio_files:
                    _, duration = get_audio_info(f)
                    total_duration_sec += duration

                hours = int(total_duration_sec // 3600)
                minutes = int((total_duration_sec % 3600) // 60)
                seconds = int(total_duration_sec % 60)

                print(f"- Total audio duration: {hours}h {minutes}m {seconds}s ({round(total_duration_sec, 2)} seconds)")
            except Exception as e:
                warnings_list.append(f"Failed to calculate total audio duration: {e}")

            # Devices & precision info
            print(f"- Devices to test ({len(devices_to_test)} total): {', '.join(devices_to_test)}")
            print(f"- Precisions to test: {', '.join(precisions_to_test)}")
            print(f"- CPU Threads (for CPU runs): {args.threads if args.threads else 'default'}")
            print(f"- Retry Attempts per model load: {args.retries}")
            print(f"- Monitoring Interval: {args.monitor_interval} sec")

            # Batch-specific
            if args.mode == "batch":
                print(f"- Batch sort order: {args.sort_by}")

            print(f"- Output directory: {args.output_dir}\n")

        # ------------------------
        # Smart Benchmark Size Estimate & Suggestion
        # ------------------------

        total_tasks = len(audio_files) * len(devices_to_test) * len(precisions_to_test)

        if not args.quiet:
            print(f"📈 Estimated total benchmark tasks: {total_tasks}")

            if total_tasks >= 500:
                print("\n⚠️  \033[93mLarge batch detected!\033[0m")
                print(f"⚙️  You are running {total_tasks} total benchmark tasks.")
                print("🔧 Recommendation: Increase --threads for faster processing if CPU allows.")
                if args.threads and args.threads < 32:
                    print(f"➡️  Tip: Try running with --threads 32 or higher (you specified {args.threads} threads).")
                elif not args.threads:
                    print("➡️  Tip: Try running with --threads 32 or --threads 64 for faster results.")



        benchmarks = []  # Final structure to hold everything
        task_futures = {}  # Track futures -> metadata

        # ------------------------
        # Dynamic Worker Pool Setup
        # ------------------------

        available_cpus = os.cpu_count() or 1

        if args.threads:
            max_workers = min(args.threads, available_cpus)
        else:
            max_workers = min(available_cpus, 8)  # Default safe fallback

        if not args.quiet:
            print(f"\n⚡ Parallel benchmarking with {max_workers} workers...")


        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for device in devices_to_test:
                for precision in precisions_to_test:
                    for audio in audio_files:
                        future = executor.submit(benchmark_model, audio, precision, device, cpu_flags)
                        task_futures[future] = (device, precision)

        # Gather results
        results_by_device_precision = {}

        # Setup tqdm progress bar
        if not args.quiet:
            pbar = tqdm(total=len(task_futures), desc="Benchmarking", ncols=100)
        else:
            pbar = None

        # Setup tqdm progress bar
        if not args.quiet:
            pbar = tqdm(total=len(task_futures), desc="Benchmarking [0 successes / 0 fails]", ncols=100)
        else:
            pbar = None

        success_count = 0
        fail_count = 0

        for future in as_completed(task_futures):
            device, precision = task_futures[future]
            try:
                result = future.result()

                key = (device, precision)
                if key not in results_by_device_precision:
                    results_by_device_precision[key] = {
                        "device": device,
                        "model_precision_used": precision,
                        "files": []
                    }
                results_by_device_precision[key]["files"].append(result)

                success_count += 1

                if not args.quiet and pbar:
                    pbar.write(f"✅ Completed benchmark for device={device} precision={precision}")

            except Exception as e:
                fail_count += 1
                if not args.quiet and pbar:
                    pbar.write(f"⚠️  Benchmark failed for device={device} precision={precision}: {e}")

                warnings_list.append(f"Benchmark failed for device={device} precision={precision}: {e}")
            finally:
                if pbar:
                    pbar.update(1)
                    pbar.set_description(f"Benchmarking [{success_count} successes / {fail_count} fails]")

        if pbar:
            pbar.close()

        # ------------------------
        # Final Success/Fail Count
        # ------------------------

        print("\n📊 Final Benchmark Summary 📊")
        if fail_count == 0:
            print(f"✅ \033[92m{success_count} successes\033[0m / {fail_count} fails\n")
        else:
            print(f"✅ \033[92m{success_count} successes\033[0m / ⚠️ \033[93m{fail_count} fails\033[0m\n")


        # Now post-process and summarize
        for (device, precision), group in results_by_device_precision.items():
            files_results = group["files"]

            valid_files = [
                f for f in files_results
                if "real_time_factor" in f and f.get("real_time_factor") is not None
            ]
            valid_count = len(valid_files)

            if valid_count == 0:
                warnings_list.append(f"No valid successful results for device={device} precision={precision}")

            total_duration = sum(f.get("audio_duration_seconds", 0) for f in valid_files)
            total_inference_time = sum(f.get("inference_time_ms", 0) for f in valid_files)
            total_rtf = sum(f.get("real_time_factor", 0) for f in valid_files)

            summary = {
                "device": device,
                "model_precision_used": precision,
                "file_count": len(files_results),
                "total_audio_duration_sec": round(total_duration, 2),
                "total_inference_time_ms": round(total_inference_time, 2),
                "average_real_time_factor": round(total_rtf / valid_count, 4) if valid_count > 0 else 0,
                "files": files_results
            }

            benchmarks.append(summary)

        # Save full JSON
        os.makedirs(args.output_dir, exist_ok=True)
        outfile = os.path.join(args.output_dir, f"{timestamp}-{base_audio_name}-multi-device-benchmark.json")
        with open(outfile, "w") as f:
            json.dump({
                "model_name": model_size,
                "mode": args.mode,
                "system_info": system_info,
                "cpu_flags_detected": cpu_flags,
                "devices_tested": devices_to_test,
                "benchmarks": benchmarks
            }, f, indent=4)
        print(f"\nBenchmark results saved to {outfile}")
        
        if args.detailed_csv:
            detailed_csv_file = os.path.join(args.output_dir, f"{timestamp}-{base_audio_name}-detailed.csv")
            with open(detailed_csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "interval_index", "elapsed_ms", "audio_file", "precision", "device",
                    "cpu_percent", "memory_usage_mb", "audio_duration_sec", "percent_complete",
                    "gpu_utilization_percent", "gpu_memory_usage_mb", "gpu_power_watts",
                    "cpu_power_watts",   # <-- Add this new field here
                    "disk_read_bytes", "disk_write_bytes"
                ])

                for group in benchmarks:
                    for file_entry in group["files"]:
                        for sample in file_entry.get("detailed_samples", []):
                            writer.writerow([
                                sample["interval_index"],
                                sample["elapsed_ms"],
                                file_entry["audio_file_name"],
                                group["model_precision_used"],
                                group["device"],
                                sample["cpu_percent"],
                                sample["memory_usage_mb"],
                                sample["audio_duration_sec"],
                                sample["percent_complete"],
                                sample.get("gpu_utilization_percent"),
                                sample.get("gpu_memory_usage_mb"),
                                sample.get("gpu_power_watts"),
                                sample.get("cpu_power_watts"),   # <-- Add this field matching the header
                                sample.get("disk_read_bytes"),
                                sample.get("disk_write_bytes")
                            ])

        if args.detailed_json:
            detailed_json_file = os.path.join(args.output_dir, f"{timestamp}-{base_audio_name}-detailed.json")
            all_detailed = []
            for group in benchmarks:
                for file_entry in group["files"]:
                    for sample in file_entry.get("detailed_samples", []):
                        all_detailed.append({
                            "interval_index": sample["interval_index"],
                            "elapsed_ms": sample["elapsed_ms"],
                            "audio_file": file_entry["audio_file_name"],
                            "precision": group["model_precision_used"],
                            "device": group["device"],
                            "cpu_percent": sample["cpu_percent"],
                            "memory_usage_mb": sample["memory_usage_mb"],
                            "audio_duration_sec": sample["audio_duration_sec"],
                            "percent_complete": sample["percent_complete"],
                            "gpu_utilization_percent": sample.get("gpu_utilization_percent"),
                            "gpu_memory_usage_mb": sample.get("gpu_memory_usage_mb"),
                            "gpu_power_watts": sample.get("gpu_power_watts"),
                            "disk_read_bytes": sample.get("disk_read_bytes"),
                            "disk_write_bytes": sample.get("disk_write_bytes"),
                            "cpu_power_watts": sample.get("cpu_power_watts")  # <-- Add this field too
                        })
            with open(detailed_json_file, "w") as f:
                json.dump(all_detailed, f, indent=2)

        
        # ------------------------
        # CLI Table Output + CSV/MD Export (with optional GPU stats)
        # ------------------------

        benchmarks.sort(key=lambda x: (x["device"], x["model_precision_used"]))
        
        for result in benchmarks:
            files = result["files"]
            successful_files = [f for f in files if "error" not in f]
            
            if not successful_files:
                continue
            
            # Detect whether GPU stats are available
            gpu_columns_needed = any(
                f.get("detailed_samples") and any(
                    sample.get("gpu_utilization_percent") is not None
                    for sample in f["detailed_samples"]
                )
                for f in successful_files
            )

            # Determine if GPU or Disk columns are needed
            gpu_data_present = any(
                any(sample.get("gpu_utilization_percent") is not None for sample in f.get("detailed_samples", []))
                for f in files
            )
            disk_data_present = any(
                any(sample.get("disk_read_bytes") is not None or sample.get("disk_write_bytes") is not None for sample in f.get("detailed_samples", []))
                for f in files
            )

            # Build table
            table = Table(title=f"📟 Device: {result['device']} | 🎯 Precision: {result['model_precision_used']}")
            table.add_column("Audio File")
            table.add_column("Inference (ms)", justify="right")
            table.add_column("RTF", justify="right")
            table.add_column("Peak CPU", justify="right")
            table.add_column("Avg CPU", justify="right")
            table.add_column("Peak RAM", justify="right")
            table.add_column("Avg RAM", justify="right")
            table.add_column("Peak CPU Power (W)", justify="right")
            table.add_column("Avg CPU Power (W)", justify="right")

            if gpu_data_present:
                table.add_column("Peak GPU Util (%)", justify="right")
                table.add_column("Peak GPU Power (W)", justify="right")
            if disk_data_present:
                table.add_column("Peak Disk Read (MB)", justify="right")
                table.add_column("Peak Disk Write (MB)", justify="right")

            for f in successful_files:
                row = [
                    f["audio_file_name"],
                    f"{f['inference_time_ms']:.2f}",
                    f"{f['real_time_factor']:.4f}",
                    f"{f['peak_cpu_usage_percent']:.2f}%",
                    f"{f['average_cpu_usage_percent']:.2f}%",
                    f"{f.get('peak_memory_usage_mb', 0):.2f}",
                    f"{f.get('average_memory_usage_mb', 0):.2f}"

                ]
                row.append(f"{f['peak_cpu_power_watts']:.2f}" if f.get("peak_cpu_power_watts") is not None else "N/A")
                row.append(f"{f['average_cpu_power_watts']:.2f}" if f.get("average_cpu_power_watts") is not None else "N/A")

                if gpu_data_present:
                    gpu_util = max((sample.get("gpu_utilization_percent") or 0) for sample in f.get("detailed_samples", []))
                    gpu_power = max((sample.get("gpu_power_watts") or 0) for sample in f.get("detailed_samples", []))
                    row.append(f"{gpu_util:.2f}%" if gpu_util else "N/A")
                    row.append(f"{gpu_power:.2f}" if gpu_power else "N/A")

                if disk_data_present:
                    peak_disk_read = max((sample.get("disk_read_bytes") or 0) / (1024**2) for sample in f.get("detailed_samples", []))  # MB
                    peak_disk_write = max((sample.get("disk_write_bytes") or 0) / (1024**2) for sample in f.get("detailed_samples", []))  # MB
                    row.append(f"{peak_disk_read:.2f}" if peak_disk_read else "N/A")
                    row.append(f"{peak_disk_write:.2f}" if peak_disk_write else "N/A")

                table.add_row(*row)

            # Pretty print table
            console.print(Panel.fit(
                table,
                title="CLI Summary Tables",
                border_style="cyan"
            ))
                

            filename_base = f"{timestamp}_{base_audio_name}_{result['device']}_{result['model_precision_used']}"

            # Export CSV
            if args.export_csv:
                csv_file = os.path.join(args.output_dir, f"{filename_base}_benchmark.csv")
                with open(csv_file, "w", newline="") as f:
                    writer = csv.writer(f)

                    # Base headers
                    headers = ["Audio File", "Inference (ms)", "RTF", "Peak CPU (%)", "Avg CPU (%)", "Peak RAM (MB)", "Avg RAM (MB)"]
                    headers += ["Peak CPU Power (W)", "Avg CPU Power (W)"]

                    if gpu_columns_needed:
                        headers += ["Peak GPU (%)", "Avg GPU (%)", "Peak GPU Mem (MB)", "Avg GPU Mem (MB)"]

                    writer.writerow(headers)

                    for fobj in successful_files:
                        row = [
                            fobj["audio_file_name"],
                            fobj["inference_time_ms"],
                            fobj["real_time_factor"],
                            fobj["peak_cpu_usage_percent"],
                            fobj["average_cpu_usage_percent"],
                            fobj.get("peak_memory_usage_mb", fobj.get("max_memory_usage_mb", 0)),
                            fobj.get("average_memory_usage_mb", 0)
                        ]
                        row += [
                            f"{fobj['peak_cpu_power_watts']:.2f}" if fobj.get("peak_cpu_power_watts") is not None else "N/A",
                            f"{fobj['average_cpu_power_watts']:.2f}" if fobj.get("average_cpu_power_watts") is not None else "N/A"
                        ]


                        if gpu_columns_needed:
                            peak_gpu_util = max(
                                (sample.get("gpu_utilization_percent", 0) for sample in fobj.get("detailed_samples", []) if sample.get("gpu_utilization_percent") is not None),
                                default=None
                            )
                            avg_gpu_util = (
                                sum(sample.get("gpu_utilization_percent", 0) for sample in fobj.get("detailed_samples", []) if sample.get("gpu_utilization_percent") is not None) /
                                max(1, sum(1 for sample in fobj.get("detailed_samples", []) if sample.get("gpu_utilization_percent") is not None))
                            ) if peak_gpu_util is not None else None

                            peak_gpu_mem = max(
                                (sample.get("gpu_memory_usage_mb", 0) for sample in fobj.get("detailed_samples", []) if sample.get("gpu_memory_usage_mb") is not None),
                                default=None
                            )
                            avg_gpu_mem = (
                                sum(sample.get("gpu_memory_usage_mb", 0) for sample in fobj.get("detailed_samples", []) if sample.get("gpu_memory_usage_mb") is not None) /
                                max(1, sum(1 for sample in fobj.get("detailed_samples", []) if sample.get("gpu_memory_usage_mb") is not None))
                            ) if peak_gpu_mem is not None else None

                            row += [
                                f"{peak_gpu_util:.2f}" if peak_gpu_util is not None else "N/A",
                                f"{avg_gpu_util:.2f}" if avg_gpu_util is not None else "N/A",
                                f"{peak_gpu_mem:.2f}" if peak_gpu_mem is not None else "N/A",
                                f"{avg_gpu_mem:.2f}" if avg_gpu_mem is not None else "N/A"
                            ]

                        writer.writerow(row)

            # Export Markdown
            if args.export_md:
                md_file = os.path.join(args.output_dir, f"{filename_base}_benchmark.md")
                with open(md_file, "w") as f:
                    # Header
                    headers = [
                        "Audio File", "Inference (ms)", "RTF",
                        "Peak CPU (%)", "Avg CPU (%)", "Peak RAM (MB)", "Avg RAM (MB)"
                    ]
                    headers += ["Peak CPU Power (W)", "Avg CPU Power (W)"]


                    if gpu_columns_needed:
                        headers += ["Peak GPU (%)", "Avg GPU (%)", "Peak GPU Mem (MB)", "Avg GPU Mem (MB)"]

                    f.write("| " + " | ".join(headers) + " |\n")
                    f.write("|" + "|".join([":" + "-"*12 + "-" + ":" for _ in headers]) + "|\n")

                    # Rows
                    for fobj in successful_files:
                        row = [
                            fobj["audio_file_name"],
                            f"{fobj['inference_time_ms']:.2f}",
                            f"{fobj['real_time_factor']:.4f}",
                            f"{fobj['peak_cpu_usage_percent']:.2f}",
                            f"{fobj['average_cpu_usage_percent']:.2f}",
                            f"{fobj.get('peak_memory_usage_mb', fobj.get('max_memory_usage_mb', 0)):.2f}",
                            f"{fobj.get('average_memory_usage_mb', 0):.2f}"
                        ]
                        row += [
                            f"{fobj['peak_cpu_power_watts']:.2f}" if fobj.get("peak_cpu_power_watts") is not None else "N/A",
                            f"{fobj['average_cpu_power_watts']:.2f}" if fobj.get("average_cpu_power_watts") is not None else "N/A"
                        ]

                        if gpu_columns_needed:
                            peak_gpu_util = max(
                                (sample.get("gpu_utilization_percent", 0) for sample in fobj.get("detailed_samples", []) if sample.get("gpu_utilization_percent") is not None),
                                default=None
                            )
                            avg_gpu_util = (
                                sum(sample.get("gpu_utilization_percent", 0) for sample in fobj.get("detailed_samples", []) if sample.get("gpu_utilization_percent") is not None) /
                                max(1, sum(1 for sample in fobj.get("detailed_samples", []) if sample.get("gpu_utilization_percent") is not None))
                            ) if peak_gpu_util is not None else None

                            peak_gpu_mem = max(
                                (sample.get("gpu_memory_usage_mb", 0) for sample in fobj.get("detailed_samples", []) if sample.get("gpu_memory_usage_mb") is not None),
                                default=None
                            )
                            avg_gpu_mem = (
                                sum(sample.get("gpu_memory_usage_mb", 0) for sample in fobj.get("detailed_samples", []) if sample.get("gpu_memory_usage_mb") is not None) /
                                max(1, sum(1 for sample in fobj.get("detailed_samples", []) if sample.get("gpu_memory_usage_mb") is not None))
                            ) if peak_gpu_mem is not None else None

                            row += [
                                f"{peak_gpu_util:.2f}" if peak_gpu_util is not None else "N/A",
                                f"{avg_gpu_util:.2f}" if avg_gpu_util is not None else "N/A",
                                f"{peak_gpu_mem:.2f}" if peak_gpu_mem is not None else "N/A",
                                f"{avg_gpu_mem:.2f}" if avg_gpu_mem is not None else "N/A"
                            ]

                        f.write("| " + " | ".join(row) + " |\n")


        # ------------------------
        # Final Info and Warning Summaries
        # ------------------------

        if info_list and not args.quiet:
            print("\n📘 \033[94mInfo Summary\033[0m 📘")
            print("-" * 40)
            for idx, info in enumerate(info_list, 1):
                print(f"{idx}. {info}")
            print("-" * 40)

        if warnings_list and not args.quiet:
            print("\n⚠️  \033[93mWarning Summary\033[0m ⚠️")
            print("-" * 40)
            for idx, warning in enumerate(warnings_list, 1):
                print(f"{idx}. {warning}")
            print("-" * 40)

        if not info_list and not warnings_list:
            print("\n✅ No infos or warnings encountered.")

        # ------------------------
        # Save Info and Warning Logs to Files
        # ------------------------

        # Define base filename (same as benchmark JSON)
        base_filename = f"{timestamp}-{base_audio_name}-multi-device-benchmark"

        # Save warnings
        if warnings_list:
            warnings_file = os.path.join(args.output_dir, f"{base_filename}-warnings.txt")
            with open(warnings_file, "w") as f:
                f.write("⚠️  Warning Summary ⚠️\n")
                f.write("-" * 40 + "\n")
                for idx, warning in enumerate(warnings_list, 1):
                    f.write(f"{idx}. {warning}\n")
                f.write("-" * 40 + "\n")
            print(f"Saved warnings to {warnings_file}")

        # Save infos
        if info_list:
            infos_file = os.path.join(args.output_dir, f"{base_filename}-infos.txt")
            with open(infos_file, "w") as f:
                f.write("📢 Info Summary 📢\n")
                f.write("-" * 40 + "\n")
                for idx, info in enumerate(info_list, 1):
                    f.write(f"{idx}. {info}\n")
                f.write("-" * 40 + "\n")
            print(f"Saved infos to {infos_file}")

        # ------------------------
        # Final Completion Timer
        # ------------------------

        benchmark_end_time = time.perf_counter()
        total_seconds = benchmark_end_time - benchmark_start_time
        minutes, seconds = divmod(total_seconds, 60)

        # Pretty panel for final benchmark completion
        completion_text = f"[bold green]✅ Benchmark Completed![/bold green]\n\n[white]Total runtime:[/] [cyan]{int(minutes)}[/] minutes [cyan]{int(seconds)}[/] seconds"
        console.print(Panel(completion_text, title="Benchmark Status", expand=False, border_style="bright_green"))

        # ------------------------
        # Optional Exit Code
        # ------------------------

        if fail_count > 0:
            console.print("\n[bold red]❌ Some benchmarks failed. Exiting with code 1.[/bold red]\n")
            sys.exit(1)

# === ✅ End of faster_whisper_benchmark.py ✅ ===


    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user (Ctrl+C). Exiting...")
        sys.exit(1)