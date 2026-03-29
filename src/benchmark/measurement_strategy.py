import torch
import time
from abc import ABC, abstractmethod


class MeasurementStrategy(ABC):
    """Интерфейс произвольного способа измерения времени."""
    
    @abstractmethod
    def start_measurement(self):
        """Начало измерения времени."""
    
    @abstractmethod
    def end_measurement(self):
        """Завершение измерения времени."""
    
    @abstractmethod
    def get_elapsed_ms(self) -> float:
        """Получение измеренного времени в миллисекундах."""


class PerfCounterStrategy(MeasurementStrategy):
    """End-to-end измерение времени через time.perf_counter()."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start_measurement(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    
    def end_measurement(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
    
    def get_elapsed_ms(self) -> float:
        if self.start_time is None or self.end_time is None:
            raise RuntimeError()
        return (self.end_time - self.start_time) * 1000


class CudaEventStrategy(MeasurementStrategy):
    """Измерение времени через torch.cuda.Event"""
    
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start_measurement(self):
        self.start_event.record()
    
    def end_measurement(self):
        self.end_event.record()
        self.end_event.synchronize()
    
    def get_elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)


class NvtxStrategy(MeasurementStrategy):
    """Измерение с аннотацией NVTX для профилировщика."""
    
    def __init__(self, mark_name='measurement'):
        if not torch.cuda.is_available():
            raise RuntimeError()
        self.mark_name = mark_name
        self.start_time = None
        self.end_time = None
    
    def start_measurement(self):
        torch.cuda.nvtx.range_push(self.mark_name)
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    
    def end_measurement(self):
        torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        torch.cuda.nvtx.range_pop()
    
    def get_elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
