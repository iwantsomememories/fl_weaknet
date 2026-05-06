import csv
import random


def generate_complex_weaknet_trace(
    path="trace_complex_weaknet.csv",
    rounds=200,
    clients=20,
    seed=42,
):
    random.seed(seed)

    # 三类客户端：低时延、高时延、不稳定
    client_profiles = {}
    for cid in range(clients):
        if cid < 7:
            client_profiles[cid] = {
                "type": "low_delay",
                "delay_range": (0.5, 1.5),
                "loss_prob": 0.04,
                "retrans_prob": 0.06,
                "spike_prob": 0.03,
            }
        elif cid < 14:
            client_profiles[cid] = {
                "type": "high_delay",
                "delay_range": (1.8, 3.5),
                "loss_prob": 0.06,
                "retrans_prob": 0.08,
                "spike_prob": 0.05,
            }
        else:
            client_profiles[cid] = {
                "type": "unstable",
                "delay_range": (1.0, 2.8),
                "loss_prob": 0.12,
                "retrans_prob": 0.18,
                "spike_prob": 0.10,
            }

    # 网络级突发干扰区间：这些轮次内整体链路质量恶化
    burst_ranges = [(40, 55), (110, 125), (165, 175)]

    # 客户端短时断连区间
    disconnect_events = {
        3: [(30, 36), (122, 128)],
        7: [(75, 86)],
        12: [(45, 51), (150, 156)],
        18: [(100, 112)],
    }

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "client_id",
            "client_type",
            "base_delay",
            "retransmissions",
            "total_delay",
            "lost",
            "disconnected",
            "event_type",
        ])

        for r in range(1, rounds + 1):
            in_burst = any(start <= r <= end for start, end in burst_ranges)

            for cid in range(clients):
                profile = client_profiles[cid]
                low, high = profile["delay_range"]

                base_delay = random.uniform(low, high)
                retransmissions = 0
                lost = 0
                disconnected = 0
                event_type = "normal"

                # 1. 短时断连：优先级最高
                is_disconnected = any(
                    start <= r <= end
                    for start, end in disconnect_events.get(cid, [])
                )

                if is_disconnected:
                    disconnected = 1
                    lost = 1
                    total_delay = 0.0
                    event_type = "disconnection"

                else:
                    # 2. 突发丢包：网络级 burst 期间丢包率显著升高
                    loss_prob = profile["loss_prob"]
                    if in_burst:
                        loss_prob = min(0.45, loss_prob + 0.25)

                    if random.random() < loss_prob:
                        lost = 1
                        total_delay = 0.0
                        event_type = "burst_loss" if in_burst else "loss"

                    else:
                        # 3. 重传：未丢失时可能发生一次或多次重传
                        retrans_prob = profile["retrans_prob"]
                        if in_burst:
                            retrans_prob = min(0.50, retrans_prob + 0.20)

                        if random.random() < retrans_prob:
                            retransmissions = random.choice([1, 1, 2])
                            event_type = "retransmission"

                        total_delay = base_delay * (1 + retransmissions)

                        # 4. 时延尖峰：未丢失、未断连时，偶发大时延
                        spike_prob = profile["spike_prob"]
                        if in_burst:
                            spike_prob = min(0.35, spike_prob + 0.15)

                        if random.random() < spike_prob:
                            total_delay *= random.uniform(3.0, 5.0)
                            event_type = "delay_spike"

                writer.writerow([
                    r,
                    cid,
                    profile["type"],
                    round(base_delay, 4),
                    retransmissions,
                    round(total_delay, 4),
                    lost,
                    disconnected,
                    event_type,
                ])


if __name__ == "__main__":
    generate_complex_weaknet_trace()