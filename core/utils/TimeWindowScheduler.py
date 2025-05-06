from enum import Enum
from typing import List
import numpy as np
from utils.ModelParameterOperation import compute_scarcity_reward

class BaseScheduler:
    def __init__(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def select_window(self, time_costs):
        """
        选择时间窗口
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class UCBScheduler(BaseScheduler):
    def __init__(self, num_clients: int, alpha: float = 1.0, beta: float = 0.5):
        """
        :param candidate_windows: 候选时间窗口列表（单位：秒）
        """
        self.num_clients = num_clients
        self.alpha = alpha 
        self.beta = beta 
    
    def set_candidate_windows(self, candidate_windows: List[float]):
        self.windows = candidate_windows
        self.counts = np.zeros(len(candidate_windows))  # 每个窗口被选择的次数
        self.rewards = np.zeros(len(candidate_windows)) # 每个窗口的历史平均奖励

        self.effiecny_max = self.num_clients/max(candidate_windows)  # 最大效率
    
    def select_window(self):
        """UCB算法选择最优时间窗口"""
        ucb_values = []
        total_counts = np.sum(self.counts)
        
        for i in range(len(self.windows)):
            if self.counts[i] == 0:
                # 冷启动阶段：优先选择未尝试的窗口
                return self.windows[i]
            # 历史平均奖励 + 探索项
            exploration = np.sqrt(2 * np.log(total_counts) / self.counts[i])
            ucb_values.append(self.rewards[i] + exploration)
        
        return self.windows[np.argmax(ucb_values)]
    
    def update(self, chosen_window, reward):
        """更新窗口的奖励统计"""
        idx = self.windows.index(chosen_window)
        self.counts[idx] += 1
        self.rewards[idx] += (reward - self.rewards[idx]) / self.counts[idx]  # 增量更新平均奖励
    
    def get_reward(self, global_params, client_params_list, time_window):
        if len(client_params_list) == 0:
            return 0
        else:
            scarcity = compute_scarcity_reward(global_params, client_params_list, method="mean")
            effiency_raw = len(client_params_list) / time_window 

            return self.alpha * effiency_raw / self.effiecny_max + self.beta * scarcity

class NaiveScheduler(BaseScheduler):
    def __init__(self, mode):
        if mode not in ["average", "max", "fixed"]:
            raise ValueError("Invalid mode. Choose 'average' or 'max'.")
        self.mode = mode
    
    def select_window(self, time_costs, time_window):
        if len(time_costs) == 0:
            return 2 * time_window
        if self.mode == "average":
            return sum(time_costs)/len(time_costs)
        elif self.mode == "max":
            return max(time_costs)
        elif self.mode == "fixed":
            return time_window

if __name__ == "__main__":
    import random, time

