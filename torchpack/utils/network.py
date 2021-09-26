import socket

__all__ = ['get_free_tcp_port']


def get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
        tcp.bind(('0.0.0.0', 0))
        port = tcp.getsockname()[1]
    return port
