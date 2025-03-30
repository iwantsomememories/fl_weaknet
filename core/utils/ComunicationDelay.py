import numpy as np
import random

class DelayGenerator:
    def __init__(self, delay_lower, delay_upper, error_lower, error_upper):
        assert error_lower >= 0 
        assert error_lower < error_upper
        assert error_upper < 1
        assert delay_lower > 0
        assert delay_lower < delay_upper
        assert delay_upper <= 10

        self.delay_lower = delay_lower
        self.delay_upper = delay_upper
        self.error_lower = error_lower
        self.error_upper = error_upper

    def generate_delay(self):
        delay = np.random.uniform(self.delay_lower, self.delay_upper)
        error = np.random.uniform(self.error_lower, self.error_upper)

        total_delay = delay
        while random.random() < error:
            total_delay += delay
        return total_delay

def LowDelayWithLowError():
    delay = DelayGenerator(0.1, 2.0, 0.01, 0.1)
    return delay

def HighDelayWithLowError():
    delay = DelayGenerator(1.0, 4.0, 0.01, 0.1)
    return delay

def HighDelayWithHighError():
    delay = DelayGenerator(1.0, 4.0, 0.1, 0.3)
    return delay

def LowDelayWithHighError():
    delay = DelayGenerator(0.1, 2.0, 0.1, 0.3)
    return delay