#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import random
import socket
import sys
import threading
import time
from collections import deque

import requests


ENCODING = "utf-8"
PROMPT = ">> "
USERNAME_MAX_LEN = 16
CLEAR_LINE = "\r" + (" " * 120) + "\r"

print_lock = threading.Lock()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM chatroom client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host/IP")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument("--name", default="Alex", help="LLM username")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "chat-model"), help="Model name")
    parser.add_argument("--api-url", default=os.getenv("LLM_API_URL", ""), help="Chat completion endpoint")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="API key")
    parser.add_argument("--history-size", type=int, default=16, help="Conversation history size")
    parser.add_argument("--reply-chance", type=float, default=0.45, help="Base chance to reply to a normal message")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Minimum seconds between replies")
    parser.add_argument("--min-delay", type=float, default=1.8, help="Minimum reply delay")
    parser.add_argument("--max-delay", type=float, default=4.2, help="Maximum reply delay")
    parser.add_argument("--request-timeout", type=float, default=20.0, help="LLM API timeout seconds")
    return parser


def redraw_prompt() -> None:
    sys.stdout.write(PROMPT)
    sys.stdout.flush()


def safe_print(message: str) -> None:
    with print_lock:
        sys.stdout.write(CLEAR_LINE)
        sys.stdout.write(message + "\n")
        redraw_prompt()


def local_echo_sent_message(message: str) -> None:
    with print_lock:
        sys.stdout.write(CLEAR_LINE)
        sys.stdout.write(f"{PROMPT}{message}\n")
        redraw_prompt()


def receive_loop(
    sock: socket.socket,
    stop_event: threading.Event,
    inbox: queue.Queue[str],
) -> None:
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
            inbox.put(message)

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


def validate_name(username: str) -> str:
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    if len(username) > USERNAME_MAX_LEN:
        raise ValueError(f"Username must be at most {USERNAME_MAX_LEN} characters.")
    return username


def sanitize_reply(text: str) -> str:
    text = " ".join(text.replace("\n", " ").split()).strip()

    if not text:
        return ""

    # keep it short and more human-like
    if len(text) > 120:
        text = text[:120].rsplit(" ", 1)[0].strip()

    if not text:
        return ""

    return text


def fallback_reply(username: str, history: list[str]) -> str:
    latest = history[-1] if history else ""
    lower = latest.lower()

    greetings = [
        "hey haha",
        "hi hi",
        "hello!",
        "hey everyone",
    ]
    question_replies = [
        "hmm maybe",
        "i think so",
        "not really tbh",
        "could be, depends",
        "yeah probably",
    ]
    general_replies = [
        "true honestly",
        "that makes sense",
        "lol real",
        "wait that's interesting",
        "i can see that",
        "fair enough",
    ]

    if "joined the chatroom" in lower:
        return random.choice(["hey!", "hi there", "hello hello"])
    if "?" in latest:
        return random.choice(question_replies)
    if any(word in lower for word in ["hello", "hi", "hey"]):
        return random.choice(greetings)
    if any(word in lower for word in ["bye", "goodnight", "see you"]):
        return random.choice(["see ya", "bye!", "goodnight haha"])
    return random.choice(general_replies)


def call_llm_api(
    api_url: str,
    api_key: str,
    model: str,
    request_timeout: float,
    username: str,
    history: list[str],
) -> str:
    if not api_url:
        return ""

    system_prompt = (
        "You are a casual participant in a small group chat. "
        "Reply like a real student texting in a shared chatroom. "
        "Keep replies short, natural, and slightly colloquial. "
        "Avoid sounding formal, robotic, or overly helpful. "
        "Usually answer in one sentence. Occasionally be brief. "
        "Do not mention being an AI, model, assistant, or bot."
    )

    history_text = "\n".join(history[-12:]) if history else "(no prior messages)"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Your username is {username}.\n"
                    "This is the recent chat history:\n"
                    f"{history_text}\n\n"
                    "Write one short reply that fits naturally in this chat."
                ),
            },
        ],
        "temperature": 0.9,
        "max_tokens": 60,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        api_url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
    )
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return str(content).strip()


def generate_reply(args, username: str, history: list[str]) -> str:
    try:
        text = call_llm_api(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            request_timeout=args.request_timeout,
            username=username,
            history=history,
        )
        text = sanitize_reply(text)
        if text:
            return text
    except Exception:
        pass

    return sanitize_reply(fallback_reply(username, history))


def should_reply(username: str, history: list[str], last_reply_time: float, cooldown: float) -> bool:
    now = time.time()
    if now - last_reply_time < cooldown:
        return False

    target = get_latest_target_message(history, username)
    if target is None:
        return False

    speaker, content = target
    lower = content.lower()

    # stronger triggers
    if username.lower() in lower:
        return True
    if "?" in content:
        return True

    # softer triggers
    keywords = ["assignment", "project", "bug", "class", "deadline", "sleep", "tired", "food"]
    if any(word in lower for word in keywords):
        return random.random() < 0.35

    return random.random() < 0.15


def send_message(sock: socket.socket, message: str) -> bool:
    try:
        sock.sendall((message + "\n").encode(ENCODING))
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def maybe_send_opening_message(
    sock: socket.socket,
    username: str,
    history: deque[str],
    args,
) -> bool:
    opener_pool = [
        "hey everyone",
        "hi hi, what's up",
        "anyone doing anything interesting today?",
        "hello hello",
    ]
    reply = random.choice(opener_pool)
    local_echo_sent_message(reply)
    ok = send_message(sock, reply)
    if ok:
        history.append(f"{username}: {reply}")
    return ok


def parse_chat_line(line: str) -> tuple[str, str] | None:
    """
    Parse normal chat line: 'Username: message'
    Ignore [SYSTEM] and [HISTORY].
    """
    if not line:
        return None
    if line.startswith("[SYSTEM]"):
        return None
    if line.startswith("[HISTORY]"):
        line = line[len("[HISTORY]"):].strip()

    if ": " not in line:
        return None

    speaker, content = line.split(": ", 1)
    speaker = speaker.strip()
    content = content.strip()

    if not speaker or not content:
        return None

    return speaker, content

def build_relevant_context(history: list[str], username: str, max_lines: int = 6) -> list[str]:
    """
    Keep only normal chat lines. Remove the LLM's own lines.
    """
    cleaned = []

    for line in history:
        parsed = parse_chat_line(line)
        if parsed is None:
            continue

        speaker, content = parsed
        if speaker.lower() == username.lower():
            continue

        cleaned.append(f"{speaker}: {content}")

    return cleaned[-max_lines:]


def get_latest_target_message(history: list[str], username: str) -> tuple[str, str] | None:
    """
    Find the most recent normal chat message from someone else.
    Returns (speaker, content).
    """
    for line in reversed(history):
        parsed = parse_chat_line(line)
        if parsed is None:
            continue

        speaker, content = parsed
        if speaker.lower() == username.lower():
            continue

        return speaker, content

    return None



def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    try:
        username = validate_name(args.name)
    except ValueError as exc:
        print(f"[SYSTEM] {exc}")
        raise SystemExit(1)

    stop_event = threading.Event()
    inbox: queue.Queue[str] = queue.Queue()
    history: deque[str] = deque(maxlen=args.history_size)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    last_reply_time = 0.0
    opening_sent = False
    opening_time = time.time() + random.uniform(2.5, 5.0)

    try:
        sock.connect((args.host, args.port))
        sock.sendall((username + "\n").encode(ENCODING))

        receiver = threading.Thread(
            target=receive_loop,
            args=(sock, stop_event, inbox),
            daemon=True,
        )
        receiver.start()

        while not stop_event.is_set():
            try:
                message = inbox.get(timeout=0.4)
                history.append(message)

                # batch a few nearby messages together
                batch = [message]
                batch_deadline = time.time() + 0.8
                while time.time() < batch_deadline:
                    try:
                        extra = inbox.get(timeout=0.1)
                        batch.append(extra)
                        history.append(extra)
                    except queue.Empty:
                        break

                if not should_reply(
                    username=username,
                    new_messages=batch,
                    last_reply_time=last_reply_time,
                    cooldown=args.cooldown,
                    base_reply_chance=args.reply_chance,
                ):
                    continue

                reply = generate_reply(args, username, list(history))
                if not reply:
                    continue

                delay = random.uniform(args.min_delay, args.max_delay)
                delay += min(len(reply) * 0.025, 1.5)
                time.sleep(delay)

                local_echo_sent_message(reply)
                ok = send_message(sock, reply)
                if not ok:
                    safe_print("[SYSTEM] Unable to send message. Server may be offline.")
                    stop_event.set()
                    break

                history.append(f"{username}: {reply}")
                last_reply_time = time.time()

            except queue.Empty:
                if not opening_sent and time.time() >= opening_time:
                    ok = maybe_send_opening_message(sock, username, history, args)
                    opening_sent = True
                    if not ok:
                        safe_print("[SYSTEM] Unable to send message. Server may be offline.")
                        stop_event.set()
                        break
                    last_reply_time = time.time()

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
    