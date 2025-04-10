import numpy as np
import random


class DelayGenerator:
    def __init__(self, delay_lower, delay_upper, error_lower, error_upper, max_retries=5):
        """
        Args:
            delay_lower/upper: Base delay bounds in seconds.
            error_lower/upper: Packet error probability bounds (0<=lower<upper<1).
            max_retries: Maximum retransmission attempts.
        """
        self.delay_lower = delay_lower
        self.delay_upper = delay_upper
        self.error_lower = error_lower
        self.error_upper = error_upper
        self.max_retries = max_retries

    def generate_delay(self):
        """Generate delay with retransmission simulation."""
        delay = np.random.uniform(self.delay_lower, self.delay_upper)
        error_prob = np.random.uniform(self.error_lower, self.error_upper)
        
        total_delay = delay
        current_retry = 1
        current_delay = delay
        
        while current_retry <= self.max_retries and random.random() < error_prob:
            current_delay *= 1.5  # 退避策略：线性/指数增长
            total_delay += current_delay
            current_retry += 1
            
        return total_delay

class MultiModeDelayGenerator:
    def __init__(self):
        self.modes = {
            "low_latency": {"delay_lower": 1.0, "delay_upper": 2.5, "error_lower": 0.05, "error_upper": 0.2},
            "high_latency": {"delay_lower": 3.0, "delay_upper": 10.0, "error_lower": 0.1, "error_upper": 0.3},
            "unstable": {"delay_lower": 0.5, "delay_upper": 2.0, "error_lower": 0.3, "error_upper": 0.8},
        }
    
    def get_generator(self, mode=None):
        if mode is None:
            mode = random.choice(list(self.modes.keys()))  # 随机选择模式
        params = self.modes[mode]
        return DelayGenerator(**params)