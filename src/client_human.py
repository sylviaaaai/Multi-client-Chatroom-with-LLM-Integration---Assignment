#!/usr/bin/env python3
from __future__ import annotations

import argparse
import msvcrt
import socket
import sys
import threading
import time


ENCODING = "utf-8"
PROMPT = ">> "
USERNAME_MAX_LEN = 16

print_lock = threading.Lock()
input_buffer_lock = threading.Lock()
input_buffer: list[str] = []


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human chatroom client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host/IP")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    return parser


def redraw_prompt() -> None:
    with input_buffer_lock:
        current_text = "".join(input_buffer)

    sys.stdout.write(PROMPT + current_text)
    sys.stdout.flush()


def clear_current_input() -> None:
    with input_buffer_lock:
        current_text = "".join(input_buffer)

    visible_text = PROMPT + current_text
    if not visible_text:
        return

    # Use backspace-based clearing instead of carriage-return redraws so IDE
    # consoles are less likely to insert blank lines on each keystroke.
    sys.stdout.write("\b \b" * len(visible_text))
    sys.stdout.flush()


def safe_print(message: str) -> None:
    with print_lock:
        clear_current_input()
        sys.stdout.write(message + "\n")
        redraw_prompt()


def local_echo_sent_message(message: str) -> None:
    with print_lock:
        clear_current_input()
        sys.stdout.write(f"{PROMPT}{message}\n")
        redraw_prompt()


def receive_loop(sock: socket.socket, stop_event: threading.Event) -> None:
    reader = None
    try:
        reader = sock.makefile("r", encoding=ENCODING, newline="\n")
        while not stop_event.is_set():
            line = reader.readline()
            if not line:
                safe_print("[SYSTEM] Disconnected from server.")
                stop_event.set()
                break

            message = line.rstrip("\n")
            safe_print(message)

    except (ConnectionResetError, OSError):
        safe_print("[SYSTEM] Connection lost.")
        stop_event.set()
    finally:
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        try:
            sock.close()
        except OSError:
            pass
        stop_event.set()


def get_username() -> str:
    while True:
        username = input("Please enter your username: ").strip()
        if not username:
            print("Username cannot be empty.")
            continue
        if len(username) > USERNAME_MAX_LEN:
            print(f"Username must be at most {USERNAME_MAX_LEN} characters.")
            continue
        return username


def push_character(char: str) -> None:
    with input_buffer_lock:
        input_buffer.append(char)


def pop_character() -> bool:
    with input_buffer_lock:
        if not input_buffer:
            return False
        input_buffer.pop()
        return True


def consume_input_buffer() -> str:
    with input_buffer_lock:
        message = "".join(input_buffer)
        input_buffer.clear()
        return message


def read_message(stop_event: threading.Event) -> str | None:
    while not stop_event.is_set():
        if not msvcrt.kbhit():
            time.sleep(0.03)
            continue

        key = msvcrt.getwch()

        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue

        if key == "\r":
            with print_lock:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return consume_input_buffer()

        if key == "\x03":
            raise KeyboardInterrupt

        if key == "\b":
            removed = pop_character()
            if removed:
                with print_lock:
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            continue

        if key in ("\n",):
            continue

        push_character(key)
        with print_lock:
            sys.stdout.write(key)
            sys.stdout.flush()

    return None


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    username = get_username()
    stop_event = threading.Event()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((args.host, args.port))
        sock.sendall((username + "\n").encode(ENCODING))

        receiver = threading.Thread(
            target=receive_loop,
            args=(sock, stop_event),
            daemon=True,
        )
        receiver.start()

        while not stop_event.is_set():
            with print_lock:
                redraw_prompt()

            line = read_message(stop_event)
            if line is None:
                break

            if stop_event.is_set():
                break

            message = line.strip()
            if not message:
                with print_lock:
                    redraw_prompt()
                continue

            try:
                local_echo_sent_message(message)
                sock.sendall((message + "\n").encode(ENCODING))
            except (BrokenPipeError, ConnectionResetError, OSError):
                safe_print("[SYSTEM] Unable to send message. Server may be offline.")
                stop_event.set()
                break

    except ConnectionRefusedError:
        print("[SYSTEM] Cannot connect to server. Is the server running?")
    except OSError as exc:
        print(f"[SYSTEM] Socket error: {exc}")
    except KeyboardInterrupt:
        print("\n[SYSTEM] Client exiting...")
    finally:
        stop_event.set()
        try:
            sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
