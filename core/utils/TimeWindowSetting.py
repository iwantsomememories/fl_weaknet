from enum import Enum

def Average(time_costs: list) -> float:
    return sum(time_costs)/len(time_costs)

def Max(time_costs: list) -> float:
    return max(time_costs)

class TimeWindow(Enum):
    average = Average
    max = Max


if __name__ == "__main__":
    import random, time

