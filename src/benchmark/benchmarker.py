import torch
import numpy as np

from measurement_strategy import (
    MeasurementStrategy, 
    PerfCounterStrategy, 
    CudaEventStrategy, 
    NvtxStrategy
)


class Benchmarker:
    def __init__(
        self,
        model,
        device: str = 'cuda',
        warmup_count: int = 2,
        inference_count: int = 100,
        strategy: MeasurementStrategy = None,
    ):
        self.device = self._get_device(device)
        self.model = model.to(device).eval()
        self.strategy = strategy
        
        self.warmup = warmup_count
        self.runs = inference_count

    def _get_device(self, device: str) -> torch.device:
        """Проверка доступности переданного устройства."""
        if device == 'cpu':
            return torch.device('cpu')

        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                return torch.device('cpu')
            return torch.device(device)

        raise ValueError()

    def set_strategy(self, strategy: MeasurementStrategy):
        """Смена стратегии измерения времени."""
        self.strategy = strategy

    def _warmup(self, input_tensor: torch.Tensor):
        """Прогрев модели перед измерением времени инференса."""
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = self.model(input_tensor)

    def measure_latency(self, input_tensor: torch.Tensor) -> dict:
        """Измерение latency (ms) инференса модели."""
        input_tensor = input_tensor.to(self.device)
        self._warmup(input_tensor)

        latencies = []
        with torch.no_grad():
            for i in range(self.runs):
                self.strategy.start_measurement()
                _ = self.model(input_tensor)
                self.strategy.end_measurement()
                
                latency_ms = self.strategy.get_elapsed_ms()
                latencies.append(latency_ms)
        
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    def measure_throughput(
        self, input_tensor: torch.Tensor, batch_sizes=[1, 2, 4, 8, 16, 32, 64],
	) -> dict:
        """Измерение throughput (img/s)."""     
        results = {}
   
        for batch_size in batch_sizes:
            batch_input = input_tensor.repeat(batch_size, 1, 1, 1).to(self.device)
            self._warmup(batch_input)

            self.strategy.start_measurement()
            for _ in range(self.runs):
                _ = self.model(batch_input)
            self.strategy.end_measurement()
            
            total_time_ms = self.strategy.get_elapsed_ms()

            total_images = batch_size * self.runs
            throughput = total_images / (total_time_ms / 1000)

            time_per_image_ms = total_time_ms / total_images

            results[batch_size] = {
                'throughput_images_per_sec': throughput,
                'time_per_image_ms': time_per_image_ms,
                'total_time_ms': total_time_ms,
                'total_images': total_images
            }
        
        return results
