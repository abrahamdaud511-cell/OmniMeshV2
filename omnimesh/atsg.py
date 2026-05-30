import threading
import time
from collections import deque
from typing import Dict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed. Install with: pip install psutil")

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("WARNING: GPUtil not installed. Install with: pip install gputil")

from ..config import ModelConfig


class PIDController:
    """PID Controller untuk ATSG"""
    def __init__(self, Kp=0.5, Ki=0.1, Kd=0.05, setpoint=0.8):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = 0

    def update(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


class AdaptiveTrainingStabilityGovernor:
    """
    Memonitor resource dan menyesuaikan parameter training secara real-time.
    Memantau: GPU utilization, GPU memory, CPU usage, temperature.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.history = deque(maxlen=100)
        self.pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.05, setpoint=0.8)
        self.throttle_level = 1.0
        self.monitor_thread = None
        self.stop_flag = False
        self.current_batch_size = config.batch_size
        self.original_batch_size = config.batch_size
        self.current_topk = config.top_k_experts
        self.original_topk = config.top_k_experts
        self._lock = threading.Lock()

    def start(self):
        """Start monitoring in background thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_flag = False
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("✅ ATSG monitoring started")

    def stop(self):
        self.stop_flag = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    def _monitor_loop(self):
        while not self.stop_flag:
            try:
                cpu_util = psutil.cpu_percent(interval=1) if HAS_PSUTIL else 50
                mem_util = psutil.virtual_memory().percent if HAS_PSUTIL else 50

                gpu_util = 0
                gpu_mem = 0
                gpu_temp = 60

                if HAS_GPUTIL:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_util = gpu.load * 100
                            gpu_mem = gpu.memoryUtil
                            gpu_temp = gpu.temperature if hasattr(gpu, 'temperature') else 60
                    except Exception:
                        pass

                target_util = 80.0
                error = (target_util - (gpu_util + gpu_mem*100)) / 100
                adjustment = self.pid.update(error)

                with self._lock:
                    if gpu_mem > self.config.gpu_mem_threshold or gpu_temp > self.config.temp_threshold:
                        self._reduce_batch_size()
                    elif adjustment < -0.2 and cpu_util > self.config.cpu_threshold:
                        self._throttle_data_loading(0.8)
                    elif adjustment > 0.2 and cpu_util < 50:
                        self._restore_optimal()

                    if gpu_util > 95 and gpu_temp > 75:
                        self.current_topk = max(2, self.current_topk - 1)
                    elif gpu_util < 60 and self.current_topk < self.original_topk:
                        self.current_topk = min(self.original_topk, self.current_topk + 1)

                self._log_status(cpu_util, mem_util, gpu_util, gpu_mem, gpu_temp)
                time.sleep(5)
            except Exception as e:
                print(f"ATSG monitor error: {e}")
                time.sleep(5)

    def _reduce_batch_size(self):
        new_bs = max(self.current_batch_size // 2, 1)
        if new_bs != self.current_batch_size:
            self.current_batch_size = new_bs
            print(f"🔥 ATSG: Batch size reduced to {self.current_batch_size}")
            return True
        return False

    def _throttle_data_loading(self, factor):
        self.throttle_level = factor
        print(f"⏸️ ATSG: Data loading throttled to {factor*100}%")

    def _restore_optimal(self):
        if self.current_batch_size < self.original_batch_size:
            self.current_batch_size = min(self.current_batch_size * 2, self.original_batch_size)
            print(f"✅ ATSG: Batch size restored to {self.current_batch_size}")

    def _log_status(self, cpu, mem, gpu_util, gpu_mem, temp):
        pass  # Optional logging

    def get_training_params(self) -> Dict:
        with self._lock:
            return {
                'batch_size': self.current_batch_size,
                'topk': self.current_topk,
                'throttle_level': self.throttle_level
            }
