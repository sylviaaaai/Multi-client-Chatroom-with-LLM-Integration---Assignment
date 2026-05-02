#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import random
import re
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
        "avoid": ["lol", "lmao", "haha", "ugh"],
        "persona": (
            "You are a steady college student who likes classes, coffee, quiet study spots, "
            "and getting assignments done without drama."
        ),
        "tone_hint": (
            "calm and practical. You answer plainly, keep emotion low, "
            "and sound like someone focused on getting through schoolwork."
        ),
    },
    "warm": {
        "avoid": ["lmao", "whatever", "ugh"],
        "persona": (
            "You are a friendly college student who likes food, music, weekend plans, "
            "and checking in on people when they sound stressed."
        ),
        "tone_hint": (
            "warm and supportive. You sound friendly, notice how others feel, "
            "and give relaxed encouragement without sounding formal."
        ),
    },
    "intl": {
        "avoid": ["therefore", "moreover", "as an ai", "i would recommend", "i think", "in my opinion"],
        "imperfect_english": False,
        "persona": (
            "You are an international student from Asia studying computer science. "
            "You chat casually with classmates about school, assignments, games, and daily life, "
            "responding naturally to what others say without sounding robotic."
        ),
        "tone_hint": (
            "casual and friendly. Use short, natural sentences that fit the conversation, "
            "respond directly to recent messages, and sound like a real student chatting with friends."
        ),
    },
}

REPLY_MODES = [
    "react briefly",
    "answer casually and ask a small follow-up",
    "make a casual comment",
    "share a tiny personal opinion",
    "continue the current topic without asking a question",
    "change topic slightly only if the chat feels repetitive",
]

TOPIC_POOL = [
    "Hogwarts Legacy being free on Epic again and who wants to play it",
    "how last week's CS311 final exam went and whether it felt hard",
    "CS311 topics like TCP, DNS, HTTP caches, NAT, routing, and firewalls",
    "the multi-client chatroom assignment and small bugs in the server or client",
    "UDP authenticator homework, timeouts, retransmission, SAS, and GAS tokens",
    "cheap food near campus after class",
    "final exams, late studying, and being tired",
    "weekend plans after finishing assignments",
    "music or games to relax after studying",
    "coffee before debugging code",
    "group projects and last-minute fixes",
    "weather being weird on campus",
]

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
    parser.add_argument(
        "--join-greeting-chance",
        type=float,
        default=0.70,
        help="Chance to greet a newly joined user",
    )
    parser.add_argument("--cooldown", type=float, default=2.0, help="Minimum seconds between replies")
    parser.add_argument("--min-delay", type=float, default=1.6, help="Minimum reply delay")
    parser.add_argument("--max-delay", type=float, default=3.8, help="Maximum reply delay")
    parser.add_argument(
        "--char-delay",
        type=float,
        default=0.12,
        help="Extra delay seconds per reply character",
    )
    parser.add_argument(
        "--max-length-delay",
        type=float,
        default=6.0,
        help="Maximum extra delay added for long replies",
    )
    parser.add_argument("--request-timeout", type=float, default=20.0, help="LLM API timeout seconds")
    parser.add_argument("--debug", action="store_true", help="Print debug messages")
    parser.add_argument(
        "--style",
        choices=["auto", "calm", "warm", "intl"],
        default="auto",
        help="Speaking style for this AI client",
    )
    return parser


def resolve_style(username: str, style_arg: str) -> str:
    if style_arg != "auto":
        return style_arg
    return random.choice(["calm", "warm", "intl"])


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


def calculate_reply_delay(
    message: str,
    min_delay: float,
    max_delay: float,
    char_delay: float,
    max_length_delay: float,
) -> float:
    base_delay = random.uniform(min_delay, max_delay)
    length_delay = min(len(message) * char_delay, max_length_delay)
    return base_delay + length_delay


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


def strip_emoji_and_symbols(text: str) -> str:
    # common emoticons / text faces
    patterns = [
        r":\)", r":-\)", r":\(", r":-\(",
        r";\)", r";-\)", r":D", r":-D",
        r"xD", r"XD", r"<3",
        r":P", r":-P", r":p", r":-p",
        r"\^_\^", r"T_T", r"ㅠㅠ",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    # unicode emoji / pictographs
    text = re.sub(r"[\U0001F300-\U0001FAFF]", "", text)
    text = re.sub(r"[\U00002600-\U000027BF]", "", text)

    # decorative symbols often used like emoji substitutes
    text = re.sub(r"[~～✨⭐🌟❤❤️💕💖😂🤣😭😊😅🙂🙃😉😎🤔🙏👍]", "", text)

    return " ".join(text.split()).strip()


def strip_stickers_and_emoji(text: str) -> str:
    # Markdown/image reactions and common text faces.
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(
        r"(?i)(:-?\)|:-?\(|;-?\)|:-?d|x-?d|<3|:-?p|t_t|\^_?\^|orz)",
        "",
        text,
    )

    # Bracketed sticker-style reactions such as [doge], [smile], (笑哭), or （捂脸）.
    reaction_words = (
        r"doge|smile|laugh|cry|meme|sticker|emoji|"
        r"笑|哭|捂脸|狗头|旺柴|裂开|尴尬|害羞|流泪|开心|汗|表情|摊手"
    )
    text = re.sub(rf"[\[【(（][^\]】)）]{{0,20}}(?:{reaction_words})[^\]】)）]{{0,20}}[\]】)）]", "", text)

    # Unicode emoji, pictographs, dingbats, and variation selectors.
    text = re.sub(r"[\U0001F000-\U0001FAFF]", "", text)
    text = re.sub(r"[\U00002600-\U000027BF]", "", text)
    text = re.sub(r"[\U0000FE00-\U0000FE0F]", "", text)

    # Decorative symbols often used as emoji substitutes.
    text = re.sub(r"[~♡♥★☆♪♫]+", "", text)

    return " ".join(text.split()).strip()


def strip_bracketed_reactions(text: str) -> str:
    reaction_words = (
        "doge|smile|laugh|cry|meme|sticker|emoji|"
        "\u7b11|\u54ed|\u6342\u8138|\u72d7\u5934|\u65fa\u67f4|"
        "\u88c2\u5f00|\u5c34\u5c2c|\u5bb3\u7f9e|\u6d41\u6cea|"
        "\u5f00\u5fc3|\u6c57|\u8868\u60c5|\u644a\u624b"
    )
    open_brackets = r"\[\(\u3010\uff08"
    close_brackets = r"\]\)\u3011\uff09"
    return re.sub(
        rf"[{open_brackets}][^{close_brackets}]{{0,20}}"
        rf"(?:{reaction_words})"
        rf"[^{close_brackets}]{{0,20}}[{close_brackets}]",
        "",
        text,
    )


def sanitize_reply(text: str) -> str:
    text = " ".join(text.replace("\n", " ").split()).strip()

    if not text:
        return ""

    text = text.strip("\"'“”‘’").strip()

    if len(text) > 100:
        text = text[:100].rsplit(" ", 1)[0].strip()

    if ": " in text:
        prefix, rest = text.split(": ", 1)
        if len(prefix) <= 20:
            text = rest.strip()

    text = soften_repeated_slang(text)
    text = strip_emoji_and_symbols(text)
    text = strip_stickers_and_emoji(text)
    text = strip_bracketed_reactions(text)
    return text.strip()


def add_small_english_mistakes(text: str) -> str:
    replacements = [
        ("because", "becuase"),
        ("really", "realy"),
        ("probably", "probly"),
        ("assignment", "assigment"),
        ("definitely", "definately"),
        ("tomorrow", "tomorow"),
        ("different", "diffrent"),
        ("interesting", "intersting"),
    ]
    words = text.split()
    if len(words) < 3 or random.random() > 0.45:
        return text

    changed = False
    for i, word in enumerate(words):
        bare = word.strip(".,!?")
        suffix = word[len(bare):]
        lower = bare.casefold()
        for source, typo in replacements:
            if lower == source:
                words[i] = typo + suffix
                changed = True
                break
        if changed:
            break

    return " ".join(words)


def apply_style_postprocess(text: str, style_profile: dict) -> str:
    if style_profile.get("imperfect_english"):
        text = add_small_english_mistakes(text)
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


def infer_current_topic(history: list[str], username: str) -> str:
    context = " ".join(build_relevant_context(history, username, max_lines=10)).casefold()

    topic_keywords = [
        ("Hogwarts Legacy on Epic", ["hogwarts", "legacy", "epic"]),
        ("last week's CS311 final exam", ["cs311", "final", "exam", "hard", "difficult"]),
        ("CS311 networking topics", ["tcp", "dns", "http", "cache", "nat", "routing", "firewall", "udp"]),
        ("chatroom assignment bugs", ["chatroom", "server", "client", "socket", "bug", "error"]),
        ("UDP authenticator assignment", ["authenticator", "timeout", "retransmission", "sas", "gas", "token"]),
        ("assignment or homework", ["assignment", "homework", "project", "deadline", "code"]),
        ("food or campus restaurants", ["food", "lunch", "dinner", "hungry", "ramen", "cafe", "coffee"]),
        ("being tired or sleep", ["tired", "sleep", "exhausted", "late", "nap"]),
        ("classes or exams", ["class", "exam", "quiz", "final", "study"]),
        ("weekend plans", ["weekend", "plan", "movie", "game", "music"]),
        ("weather", ["weather", "rain", "cold", "hot", "sunny"]),
    ]

    for topic, keywords in topic_keywords:
        if any(keyword in context for keyword in keywords):
            return topic

    return random.choice(TOPIC_POOL)


def choose_reply_mode() -> str:
    return random.choice(REPLY_MODES)


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
        "sleep", "tired", "food", "lunch", "dinner", "weather", "meal",
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

    # Determine style name from profile
    if "therefore" in style_profile["avoid"]:
        style_name = "intl"
    elif "lol" in style_profile["avoid"]:
        style_name = "calm"
    else:
        style_name = "warm"

    # Style-specific reply pools
    # pools = {
    #     "calm": {
    #         "question": ["maybe", "probably", "not sure"],
    #         "tired": ["same honestly", "yeah i'm tired too", "today feels long"],
    #         "assignment": ["same mine still has issues", "yeah i'm still fixing stuff", "i'm still working on it"],
    #         "food": ["now i'm hungry too", "that sounds good actually", "same i need food"],
    #         "default": ["true honestly", "yeah that's fair", "same tbh", "that makes sense"],
    #     },
    #     "warm": {
    #         "question": ["maybe honestly", "i think so", "not sure yet", "probably", "could be"],
    #         "tired": ["same honestly", "yeah i'm tired too", "i barely slept either", "today feels long"],
    #         "assignment": ["same mine still has issues", "yeah i'm still fixing stuff", "i'm still working on it", "still debugging tbh"],
    #         "food": ["now i'm hungry too", "that sounds good actually", "same i need food"],
    #         "default": ["true honestly", "yeah that's fair", "same tbh", "that makes sense"],
    #     },
    #     "intl": {
    #         "question": ["maybe", "probably", "not sure"],
    #         "tired": ["same", "yeah tired", "long day"],
    #         "assignment": ["same issues", "still fixing", "working on it"],
    #         "food": ["hungry too", "sounds good", "need food"],
    #         "default": ["true", "yeah fair", "same", "makes sense"],
    #     },
    # }

    # Select category
    if "?" in content:
        category = "question"
    elif "tired" in lower or "sleep" in lower:
        category = "tired"
    elif "assignment" in lower or "project" in lower or "bug" in lower or "code" in lower:
        category = "assignment"
    elif any(word in lower for word in ["food", "lunch", "dinner"]):
        category = "food"
    else:
        category = "default"

    pool = ["yeah", "same", "true"]
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
    context_lines = build_relevant_context(history, username, max_lines=10)
    context_text = "\n".join(context_lines) if context_lines else "(no prior context)"
    current_topic = infer_current_topic(history, username)
    reply_mode = choose_reply_mode()

    system_prompt = (
        "You are a casual student in a small group chat. "
        f"Your speaking style is: {style_profile['tone_hint']}. "
        "Reply to the target message naturally and stay on topic. "
        "Keep the reply short, natural, and human-like. "
        "Usually one sentence, under 20 words. "
        "Do not sound formal, robotic, or overly helpful. "
        "Do not overuse repeated catchphrases. "
        "Use plain text only. Never include image links or sticker-style reaction labels. "
        "Do not use emoji, stickers, memes, emoticons, kaomoji, decorative symbols, "
        "or bracketed reactions like [doge], [smile], (笑哭), or （捂脸）. "
        "Do not mention being an AI, assistant, or bot."
    )
    system_prompt = (
        f'You are "{username}", a real college student chatting in a small online chatroom. '
        f"Your personality: {style_profile['persona']} "
        f"Your speaking style is: {style_profile['tone_hint']} "
        "Reply like a human student in a group chat. "
        "Read the recent chat context and reply to the latest message naturally. "
        "Keep replies conversational, usually 10 to 25 words, but do not use 1-2 word fragments. "
        "Do not summarize the conversation. "
        "Do not ask a question every time. "
        "Do not sound formal or helpful like ChatGPT. "
        "Sometimes use casual words like yeah, wait, same, honestly, or tbh, but not too much. "
        "If English is not your first language, keep it understandable and only make tiny mistakes sometimes. "
        "Use plain text only. Do not use emoji, stickers, memes, emoticons, kaomoji, decorative symbols, or image links. "
        "Do not mention being an AI, assistant, or bot."
    )

    user_prompt = (
        f"Recent chat context:\n{context_text}\n\n"
        f"Current chat topic: {current_topic}\n"
        f"Current reply mode: {reply_mode}\n\n"
        f"Target message to reply to:\n{target_speaker}: {target_content}\n\n"
        f"Write {username}'s next message only. Do not include '{username}:'."
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
            return apply_style_postprocess(text, style_profile)
        debug_print(args.debug, "LLM returned empty text, using fallback.")
    except Exception as e:
        debug_print(args.debug, f"LLM API failed: {e}")

    text = sanitize_reply(fallback_reply(username, history, style_profile))
    return apply_style_postprocess(text, style_profile)


def call_llm_spontaneous_api(
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

    context_lines = build_relevant_context(history, username, max_lines=10)
    context_text = "\n".join(context_lines) if context_lines else "(no prior context)"
    topic = random.choice(TOPIC_POOL)

    system_prompt = (
        f'You are "{username}", a real college student in a small online chatroom. '
        f"Your personality: {style_profile['persona']} "
        f"Your speaking style is: {style_profile['tone_hint']} "
        "Read the recent chat context and continue naturally. "
        "Keep it conversational, usually 10 to 20 words, but do not reply with 1-2 word fragments. "
        "Do not sound like ChatGPT. "
        "Use plain text only. Do not use emoji, stickers, memes, emoticons, kaomoji, decorative symbols, or image links. "
        "Do not mention being an AI, assistant, or bot."
    )

    user_prompt = (
        f"Recent chat context:\n{context_text}\n\n"
        f"Possible topic: {topic}\n\n"
        f"Write {username}'s next message only. Do not include '{username}:'."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.95,
        "max_tokens": 45,
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


def generate_spontaneous_reply(args, username: str, history: list[str]) -> str:
    style_profile = get_style_profile(username, args.style)
    try:
        text = call_llm_spontaneous_api(
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
            return apply_style_postprocess(text, style_profile)
        debug_print(args.debug, "LLM returned empty spontaneous text, using fallback.")
    except Exception as e:
        debug_print(args.debug, f"LLM spontaneous API failed: {e}")

    text = sanitize_reply(generate_spontaneous_message(username, history))
    return apply_style_postprocess(text, style_profile)


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
    return random.random() < 0.15


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
        message = random.choice(candidates)
        recent_messages = [line.split(": ", 1)[1] if ": " in line else line for line in history[-10:]]
        if message in recent_messages:
            return ""
        return message

    if any(word in joined for word in ["food", "lunch", "dinner", "hungry"]):
        candidates = [
            "now i'm kind of hungry too",
            "so what are people eating later",
            "okay now i want food",
        ]
        message = random.choice(candidates)
        recent_messages = [line.split(": ", 1)[1] if ": " in line else line for line in history[-10:]]
        if message in recent_messages:
            return ""
        return message

    if any(word in joined for word in ["sleep", "tired", "exhausted"]):
        candidates = [
            "i'm still kind of tired honestly",
            "today feels slow",
            "yeah i might sleep early today",
        ]
        message = random.choice(candidates)
        recent_messages = [line.split(": ", 1)[1] if ": " in line else line for line in history[-10:]]
        if message in recent_messages:
            return ""
        return message

    return random.choice(TOPIC_POOL)


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    try:
        username = validate_name(args.name)
    except ValueError as exc:
        print(f"[SYSTEM] {exc}")
        raise SystemExit(1)

    if args.style == "auto":
        args.style = resolve_style(username, args.style)

    stop_event = threading.Event()
    inbox: queue.Queue[str] = queue.Queue()
    history: deque[str] = deque(maxlen=args.history_size)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    last_reply_time = 0.0

    # Disable forced opening messages by default
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

                        parsed_extra = parse_chat_line(extra)
                        if parsed_extra is not None:
                            extra_speaker, _ = parsed_extra
                            if extra_speaker.casefold() != username.casefold():
                                last_other_message_time = time.time()
                                consecutive_spontaneous_count = 0

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
                    delay = calculate_reply_delay(
                        reply,
                        args.min_delay,
                        args.max_delay,
                        args.char_delay,
                        args.max_length_delay,
                    )
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

                delay = calculate_reply_delay(
                    reply,
                    args.min_delay,
                    args.max_delay,
                    args.char_delay,
                    args.max_length_delay,
                )
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
                    reply = generate_spontaneous_reply(args, username, list(history))
                    reply = sanitize_reply(reply)

                    if reply:
                        delay = calculate_reply_delay(
                            reply,
                            args.min_delay,
                            args.max_delay,
                            args.char_delay,
                            args.max_length_delay,
                        )
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

                if not opening_sent and time.time() >= opening_time:
                    # kept only for future optional use
                    pass

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
