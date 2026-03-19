import socket
import threading
import queue
import time

class RobotController:
    def __init__(self, host='192.168.0.200', port=2090):
        self.host = host
        self.port = port
        self.client_socket = None
        self.connected = False
        self.response_queue = queue.Queue()
        self._connect()

    def _connect(self):
        
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.connected = True
            print(f"已连接到机器人 {self.host}:{self.port}")
            # 启动接收消息的线程
            threading.Thread(target=self._receive_messages, daemon=True).start()
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            self.connected = False
            return False

    def _receive_messages(self):
        """接收机器人返回的消息"""
        while self.connected:
            try:
                response = self.client_socket.recv(8192)
                if response:
                    response_str = response.decode('utf-8').strip()
                    self.response_queue.put(response_str)
                else:
                    print("机器人已断开连接")
                    break
            except Exception as e:
                print(f"接收消息时出错: {e}")
                break
        self.connected = False
        self.client_socket.close()

    def send_command(self, command, timeout=5):
        """
        发送指令到机器人并等待响应
        
        Args:
            command (str): 要发送的指令
            timeout (float): 等待响应的超时时间（秒）
            
        Returns:
            str: 机器人的响应，如果超时则返回None
        """
        if not self.connected:
            if not self._connect():
                return None

        try:
            # 清空之前的响应队列
            while not self.response_queue.empty():
                self.response_queue.get()

            # 发送指令
            formatted_message = f"[1#{command}]"
            self.client_socket.sendall(formatted_message.encode('utf-8'))
            
            # 等待响应
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get(timeout=0.1)
                    return response
                except queue.Empty:
                    continue
            return None
        except Exception as e:
            print(f"发送指令失败: {e}")
            self.connected = False
            return None

    def close(self):
        """关闭连接"""
        if self.connected:
            self.connected = False
            self.client_socket.close()

# 使用示例
if __name__ == "__main__":
    # 创建控制器实例
    controller = RobotController()
    
    try:
        # 发送指令示例
        response = controller.send_command("System.Login 0")
        print(f"机器人响应: {response}")
        
        # 发送另一个指令
        response = controller.send_command("Robot.PowerEnable 1,1")
        print(f"机器人响应: {response}")
        response = controller.send_command("Move.Axis 3,30")
	#$print(f"机器人响应: {response}")
        
    finally:
        # 确保关闭连接
        controller.close() 
