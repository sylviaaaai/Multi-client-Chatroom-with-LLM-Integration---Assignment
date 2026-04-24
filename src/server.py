#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import threading
from typing import Dict, Tuple


BUFFER_SIZE = 4096
ENCODING = "utf-8"


# sock -> username
clients: Dict[socket.socket, str] = {}
clients_lock = threading.Lock()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-client chatroom server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host/IP")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    return parser


def send_line(sock: socket.socket, message: str) -> bool:
    """
    Send one logical message line terminated by '\n'.
    Returns True on success, False on failure.
    """
    try:
        sock.sendall((message + "\n").encode(ENCODING))
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def broadcast(message: str, exclude_sock: socket.socket | None = None) -> None:
    """
    Send a message to all connected clients except exclude_sock.
    Failed sockets will be cleaned up.
    """
    dead_sockets: list[socket.socket] = []

    with clients_lock:
        current_clients = list(clients.keys())

    for client_sock in current_clients:
        if client_sock is exclude_sock:
            continue
        ok = send_line(client_sock, message)
        if not ok:
            dead_sockets.append(client_sock)

    for dead_sock in dead_sockets:
        cleanup_client(dead_sock, broadcast_leave=False)


def cleanup_client(client_sock: socket.socket, broadcast_leave: bool = True) -> None:
    """
    Remove client from registry and optionally broadcast leave message.
    Safe to call multiple times.
    """
    username = None

    with clients_lock:
        if client_sock in clients:
            username = clients.pop(client_sock)

    try:
        client_sock.close()
    except OSError:
        pass

    if username and broadcast_leave:
        leave_msg = f"[SYSTEM] {username} has left the chatroom!"
        print(leave_msg)
        broadcast(leave_msg, exclude_sock=client_sock)


def recv_line(file_obj) -> str | None:
    """
    Read one line from socket.makefile('r').
    Returns stripped line, or None on EOF/error.
    """
    try:
        line = file_obj.readline()
    except OSError:
        return None

    if not line:
        return None
    return line.rstrip("\n")


def handle_client(client_sock: socket.socket, client_addr: Tuple[str, int]) -> None:
    """
    Per-client worker thread.
    Protocol:
    - first line: username
    - subsequent lines: chat messages
    """
    reader = None
    username = None

    try:
        reader = client_sock.makefile("r", encoding=ENCODING, newline="\n")

        # First message must be username
        username = recv_line(reader)
        if username is None:
            return

        username = username.strip()
        if not username:
            send_line(client_sock, "[SYSTEM] Invalid username. Connection closed.")
            return

        with clients_lock:
            # Optional: prevent duplicate usernames
            if username in clients.values():
                send_line(client_sock, "[SYSTEM] Username already in use. Connection closed.")
                return
            clients[client_sock] = username

        welcome_msg = f"[SYSTEM] Welcome {username} to the chatroom!"
        join_msg = f"[SYSTEM] {username} has joined the chatroom!"

        print(f"{client_addr} connected as {username}")
        send_line(client_sock, welcome_msg)
        broadcast(join_msg, exclude_sock=None)
        print(join_msg)

        while True:
            message = recv_line(reader)
            if message is None:
                break

            message = message.strip()
            if not message:
                continue

            formatted = f"{username}: {message}"
            print(formatted)
            broadcast(formatted, exclude_sock=client_sock)

    except (ConnectionResetError, OSError):
        pass
    finally:
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        cleanup_client(client_sock, broadcast_leave=True)
        print(f"{client_addr} disconnected")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_sock.bind((args.host, args.port))
        server_sock.listen()
        print(f"Server listening on {args.host}:{args.port}")

        while True:
            client_sock, client_addr = server_sock.accept()
            thread = threading.Thread(
                target=handle_client,
                args=(client_sock, client_addr),
                daemon=True,
            )
            thread.start()

    except KeyboardInterrupt:
        print("\nServer shutting down...")
    finally:
        with clients_lock:
            current_clients = list(clients.keys())

        for client_sock in current_clients:
            try:
                send_line(client_sock, "[SYSTEM] Server is shutting down.")
                client_sock.close()
            except OSError:
                pass

        server_sock.close()


if __name__ == "__main__":
    main()