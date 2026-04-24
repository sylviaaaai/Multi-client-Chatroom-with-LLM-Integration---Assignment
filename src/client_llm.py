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

STYLE_PROFILES = {
    "calm": {
        "fillers": ["honestly", "fair", "maybe", "probably"],
        "avoid": ["lol", "ugh", "lmao"],
        "tone_hint": "calm, plain, low-energy, not too slangy",
    },
    "playful": {
        "fillers": ["lol", "haha", "honestly", "wait"],
        "avoid": ["ugh"],
        "tone_hint": "light, playful, casual, a little expressive",
    },
    "dry": {
        "fillers": ["yeah", "true", "fair", "tbh"],
        "avoid": ["lol", "haha", "ugh"],
        "tone_hint": "dry, brief, understated, low-emotion",
    },
    "warm": {
        "fillers": ["hey", "honestly", "aw", "yeah"],
        "avoid": ["ugh"],
        "tone_hint": "friendly, warm, easygoing, not too slangy",
    },
}

print_lock = threading.Lock()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM chatroom client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host/IP")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument("--name", default="Alex", help="LLM username")

    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "chat-model"), help="Model name")
    parser.add_argument("--api-url", default=os.getenv("LLM_API_URL", ""), help="Chat completion endpoint")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="API key")

    parser.add_argument("--history-size", type=int, default=24, help="Conversation history size")
    parser.add_argument("--join-greeting-chance", type=float, default=0.70, help="Chance to greet a newly joined user")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Minimum seconds between replies")
    parser.add_argument("--min-delay", type=float, default=1.6, help="Minimum reply delay")
    parser.add_argument("--max-delay", type=float, default=3.8, help="Maximum reply delay")
    parser.add_argument("--request-timeout", type=float, default=20.0, help="LLM API timeout seconds")
    parser.add_argument("--debug", action="store_true", help="Print debug messages")
    parser.add_argument(
        "--style",
        choices=["auto", "calm", "playful", "dry", "warm"],
        default="auto",
        help="Speaking style for this AI client",
    )
    return parser


def resolve_style(username: str, style_arg: str) -> str:
    if style_arg != "auto":
        return style_arg
    styles = ["calm", "playful", "dry", "warm"]
    return styles[sum(ord(c) for c in username) % len(styles)]


def get_style_profile(username: str, style_arg: str) -> dict:
    style_name = resolve_style(username, style_arg)
    return STYLE_PROFILES[style_name]


def redraw_prompt() -> None:
    sys.stdout.write(PROMPT)
    sys.stdout.flush()


def safe_print(message: str) -> None:
    with print_lock:
        sys.stdout.write(CLEAR_LINE)
        sys.stdout.write(message + "\n")
        redraw_prompt()


def debug_print(enabled: bool, message: str) -> None:
    if enabled:
        safe_print(f"[DEBUG] {message}")


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

    # Trim quotes often produced by APIs/models
    text = text.strip("\"'“”‘’").strip()

    # Keep replies short and chat-like
    if len(text) > 100:
        text = text[:100].rsplit(" ", 1)[0].strip()

    # Remove obvious speaker prefix if model returns one
    if ": " in text:
        prefix, rest = text.split(": ", 1)
        if len(prefix) <= 20:
            text = rest.strip()

    text = soften_repeated_slang(text)
    return text.strip()


def soften_repeated_slang(text: str) -> str:
    replacements = {
        "lol lol": "lol",
        "haha haha": "haha",
        "ugh ugh": "ugh",
        "lol real": "real",
    }
    lowered = text.casefold()
    for bad, good in replacements.items():
        if bad in lowered:
            text = text.replace(bad, good)
            text = text.replace(bad.title(), good)
    return text


def send_message(sock: socket.socket, message: str) -> bool:
    try:
        sock.sendall((message + "\n").encode(ENCODING))
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def parse_chat_line(line: str) -> tuple[str, str] | None:
    """
    Parse a normal chat line of the form 'Username: message'.
    Ignore most system lines. If server later adds [HISTORY], still support it.
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


def parse_join_line(line: str) -> str | None:
    prefix = "[SYSTEM] "
    suffix = " has joined the chatroom!"

    if not line.startswith(prefix) or not line.endswith(suffix):
        return None

    joined_name = line[len(prefix):-len(suffix)].strip()
    return joined_name or None


def build_relevant_context(history: list[str], username: str, max_lines: int = 6) -> list[str]:
    """
    Keep only normal chat lines and remove the LLM's own lines.
    """
    cleaned: list[str] = []

    for line in history:
        parsed = parse_chat_line(line)
        if parsed is None:
            continue

        speaker, content = parsed
        if speaker.casefold() == username.casefold():
            continue

        cleaned.append(f"{speaker}: {content}")

    return cleaned[-max_lines:]


def get_latest_target_message(history: list[str], username: str) -> tuple[str, str] | None:
    """
    Find the latest normal chat message from someone else.
    """
    for line in reversed(history):
        parsed = parse_chat_line(line)
        if parsed is None:
            continue

        speaker, content = parsed
        if speaker.casefold() == username.casefold():
            continue

        return speaker, content

    return None


def get_latest_joined_user(history: list[str], username: str) -> str | None:
    for line in reversed(history):
        joined_name = parse_join_line(line)
        if joined_name is None:
            continue
        if joined_name.casefold() == username.casefold():
            continue
        return joined_name

    return None


def should_reply(username: str, history: list[str], last_reply_time: float, cooldown: float) -> bool:
    now = time.time()
    if now - last_reply_time < cooldown:
        return False

    target = get_latest_target_message(history, username)
    if target is None:
        return False

    _speaker, content = target
    lower = content.casefold()

    # Strong triggers
    if username.casefold() in lower:
        return True
    if "?" in content:
        return True

    # Medium triggers
    keywords = [
        "assignment", "project", "bug", "class", "deadline",
        "sleep", "tired", "food", "lunch", "dinner","weather", "meal"
        "exam", "quiz", "homework", "code", "error",
    ]
    if any(word in lower for word in keywords):
        return random.random() < 0.65

    # Default: still reply sometimes, but not always
    return random.random() < 0.45


def greeting_reply(joined_name: str) -> str:
    return random.choice([
        f"hey {joined_name}",
        f"hi {joined_name}",
        f"welcome {joined_name}",
        f"hey {joined_name}, what's up",
        f"hi hi {joined_name}",
    ])


def fallback_reply(username: str, history: list[str], style_profile: dict) -> str:
    target = get_latest_target_message(history, username)
    if target is None:
        return ""

    _speaker, content = target
    lower = content.casefold()

    if "?" in content:
        pool = [
            "maybe honestly",
            "i think so",
            "not sure yet",
            "probably",
            "could be",
        ]
    elif "tired" in lower or "sleep" in lower:
        pool = [
            "same honestly",
            "yeah i'm tired too",
            "i barely slept either",
            "today feels long",
        ]
    elif "assignment" in lower or "project" in lower or "bug" in lower or "code" in lower:
        pool = [
            "same mine still has issues",
            "yeah i'm still fixing stuff",
            "i'm still working on it",
            "still debugging tbh",
        ]
    elif any(word in lower for word in ["food", "lunch", "dinner"]):
        pool = [
            "now i'm hungry too",
            "that sounds good actually",
            "same i need food",
        ]
    else:
        pool = [
            "true honestly",
            "yeah that's fair",
            "same tbh",
            "that makes sense",
        ]

    filtered = [
        p for p in pool
        if not any(bad in p.casefold() for bad in style_profile["avoid"])
    ]
    return random.choice(filtered or pool)


def extract_text_from_response(data) -> str:
    """
    Support a few common response shapes.
    """
    if isinstance(data, str):
        return data

    if not isinstance(data, dict):
        return ""

    # OpenAI-compatible Chat Completions
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content

                # Some APIs return content as structured list
                if isinstance(content, list):
                    chunks = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                chunks.append(text)
                    if chunks:
                        return " ".join(chunks)

            text = first.get("text")
            if isinstance(text, str):
                return text

    # Other possible shapes
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text

    response_text = data.get("response")
    if isinstance(response_text, str):
        return response_text

    content = data.get("content")
    if isinstance(content, str):
        return content

    return ""


def call_llm_api(
    api_url: str,
    api_key: str,
    model: str,
    request_timeout: float,
    username: str,
    history: list[str],
    style_profile: dict,
) -> str:
    if not api_url:
        return ""

    target = get_latest_target_message(history, username)
    if target is None:
        return ""

    target_speaker, target_content = target
    context_lines = build_relevant_context(history, username, max_lines=6)
    context_text = "\n".join(context_lines) if context_lines else "(no prior context)"

    system_prompt = (
        "You are a casual student in a small group chat. "
        f"Your speaking style is: {style_profile['tone_hint']}. "
        "Reply to the target message naturally and stay on topic. "
        "Keep the reply short, natural, and human-like. "
        "Usually one sentence, under 20 words. "
        "Do not sound formal, robotic, or overly helpful. "
        "Do not overuse repeated catchphrases. "
        "Do not mention being an AI, assistant, or bot."
    )

    user_prompt = (
        f"Your username: {username}\n\n"
        f"Recent chat context:\n{context_text}\n\n"
        f"Target message to reply to:\n{target_speaker}: {target_content}\n\n"
        "Write exactly one short reply to that target message."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
        "max_tokens": 50,
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
    return extract_text_from_response(data).strip()


def generate_reply(args, username: str, history: list[str]) -> str:
    style_profile = get_style_profile(username, args.style)
    try:
        text = call_llm_api(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            request_timeout=args.request_timeout,
            username=username,
            history=history,
            style_profile=style_profile,
        )
        text = sanitize_reply(text)
        if text:
            return text
        safe_print("[DEBUG] LLM returned empty text, using fallback.")
    except Exception as e:
        safe_print(f"[DEBUG] LLM API failed: {e}")

    return sanitize_reply(fallback_reply(username, history, style_profile))


def should_send_spontaneous_message(
    username: str,
    history: list[str],
    now: float,
    last_other_message_time: float,
    last_sent_time: float,
    last_spontaneous_time: float,
    consecutive_spontaneous_count: int,
) -> bool:
    # room must be quiet longer
    if now - last_other_message_time < random.uniform(16.0, 26.0):
        return False

    # do not speak too soon after own previous message
    if now - last_sent_time < 18.0:
        return False

    # do not send spontaneous messages too frequently
    if now - last_spontaneous_time < 35.0:
        return False

    # never do more than one spontaneous turn in a row
    if consecutive_spontaneous_count >= 1:
        return False

    # much lower probability
    return random.random() < 0.05

def generate_spontaneous_message(username: str, history: list[str]) -> str:
    context = build_relevant_context(history, username, max_lines=6)
    joined = " | ".join(context).lower()

    if any(word in joined for word in ["assignment", "project", "bug", "code", "deadline", "homework"]):
        candidates = [
            "are you still working on the assignment",
            "wait are you all basically done already",
            "mine still has a few issues tbh",
            "i feel like i'm still fixing small bugs",
        ]
        return random.choice(candidates)

    if any(word in joined for word in ["food", "lunch", "dinner", "hungry"]):
        candidates = [
            "now i'm kind of hungry too",
            "so what are people eating later",
            "okay now i want food",
        ]
        return random.choice(candidates)

    if any(word in joined for word in ["sleep", "tired", "exhausted"]):
        candidates = [
            "i'm still kind of tired honestly",
            "today feels slow",
            "yeah i might sleep early today",
        ]
        return random.choice(candidates)

    pool = [
        "so what is everyone up to now",
        "what's everyone doing rn",
        "are you all still around",
        "it got kind of quiet",
    ]
    weights = [0.34, 0.28, 0.24, 0.14]
    return random.choices(pool, weights=weights, k=1)[0]



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

    opening_sent = True
    opening_time = float("inf")

    last_other_message_time = time.time()
    last_sent_time = 0.0
    last_spontaneous_time = 0.0
    consecutive_spontaneous_count = 0
    
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

                parsed = parse_chat_line(message)
                if parsed is not None:
                    speaker, _content = parsed
                    if speaker.casefold() != username.casefold():
                        last_other_message_time = time.time()
                        consecutive_spontaneous_count = 0

                batch = [message]

                # Batch a few nearby messages together
                batch_deadline = time.time() + 0.8
                while time.time() < batch_deadline:
                    try:
                        extra = inbox.get(timeout=0.1)
                        history.append(extra)
                        batch.append(extra)
                    except queue.Empty:
                        break

                current_history = list(history)

                joined_name = get_latest_joined_user(batch, username)
                if (
                    joined_name is not None
                    and random.random() < args.join_greeting_chance
                    and time.time() - last_reply_time >= args.cooldown
                ):
                    reply = greeting_reply(joined_name)
                    delay = random.uniform(args.min_delay, args.max_delay)
                    time.sleep(delay)

                    local_echo_sent_message(reply)
                    ok = send_message(sock, reply)
                    if not ok:
                        safe_print("[SYSTEM] Unable to send message. Server may be offline.")
                        stop_event.set()
                        break

                    history.append(f"{username}: {reply}")
                    last_sent_time = time.time()
                    last_reply_time = last_sent_time
                    last_spontaneous_time = last_sent_time
                    consecutive_spontaneous_count = 0
                    continue

                if not should_reply(
                    username=username,
                    history=current_history,
                    last_reply_time=last_reply_time,
                    cooldown=args.cooldown,
                ):
                    continue

                target_before = get_latest_target_message(current_history, username)
                if target_before is None:
                    continue

                debug_print(args.debug, f"Target before reply: {target_before[0]}: {target_before[1]}")

                reply = generate_reply(args, username, current_history)
                if not reply:
                    continue

                delay = random.uniform(args.min_delay, args.max_delay)
                delay += min(len(reply) * 0.025, 1.2)
                time.sleep(delay)

                target_after = get_latest_target_message(list(history), username)
                if target_after != target_before:
                    debug_print(args.debug, "Skipped outdated reply because conversation moved on.")
                    continue

                local_echo_sent_message(reply)
                ok = send_message(sock, reply)
                if not ok:
                    safe_print("[SYSTEM] Unable to send message. Server may be offline.")
                    stop_event.set()
                    break

                history.append(f"{username}: {reply}")
                last_sent_time = time.time()
                last_reply_time = last_sent_time
                consecutive_spontaneous_count = 0

            except queue.Empty:
                now = time.time()

                if should_send_spontaneous_message(
                    username=username,
                    history=list(history),
                    now=now,
                    last_other_message_time=last_other_message_time,
                    last_sent_time=last_sent_time,
                    last_spontaneous_time=last_spontaneous_time,
                    consecutive_spontaneous_count=consecutive_spontaneous_count,
                ):
                    reply = generate_spontaneous_message(username, list(history))
                    reply = sanitize_reply(reply)

                    if reply:
                        delay = random.uniform(1.5, 3.0)
                        time.sleep(delay)

                        local_echo_sent_message(reply)
                        ok = send_message(sock, reply)
                        if not ok:
                            safe_print("[SYSTEM] Unable to send message. Server may be offline.")
                            stop_event.set()
                            break

                        history.append(f"{username}: {reply}")
                        last_sent_time = time.time()
                        last_reply_time = last_sent_time
                        last_spontaneous_time = last_sent_time
                        consecutive_spontaneous_count += 1

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
