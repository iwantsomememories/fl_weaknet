import threading
import queue
import time
from typing import Any, List, Tuple

class AsyncNetworkWrapper:
    def __init__(self, network):
        self.network = network
        self.message_queue = queue.Queue()
        self._stop_event = threading.Event()
        # 启动长驻接收线程
        self.recv_thread = threading.Thread(
            target=self._recv_loop, 
            daemon=True
        )
        self.recv_thread.start()

    def _recv_loop(self):
        """长驻线程的接收循环"""
        while not self._stop_event.is_set():
            try:
                data = self.network.recv()  # 阻塞调用
                self.message_queue.put(data)
            except Exception as e:
                self.message_queue.put(e)

    def recv_with_timeout(self, timeout):
        """主线程调用：带超时的接收"""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None, None  # 超时


    def recv_with_timeout(self, timeout: float) -> Tuple[Any, Any, Any]:
        """主线程调用：带超时的接收"""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None, None  # 超时

    def shutdown(self) -> List[Tuple[Any, Any, Any]]:
        """
        安全关闭，并返回未处理的消息队列
        Returns:
            List[Tuple[sender_rank, message_code, payload]]: 未处理的消息列表
        """
        # 1. 停止接收线程
        self._stop_event.set()
        self.recv_thread.join()

        # 2. 清空队列并返回所有未处理消息
        unprocessed_messages = []
        while not self.message_queue.empty():
            try:
                msg = self.message_queue.get_nowait()
                if isinstance(msg, tuple) and len(msg) == 3:  # 过滤异常
                    unprocessed_messages.append(msg)
            except queue.Empty:
                break
        
        return unprocessed_messages