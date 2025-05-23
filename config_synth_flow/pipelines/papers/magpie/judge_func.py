from hanzidentifier import has_chinese, is_traditional


def only_traditional_chinese(dct: dict, messages_col: str = "messages") -> bool:
    for message in dct[messages_col]:
        if not is_traditional(message["content"]):
            return False
    return True


def only_chinese(dct: dict, message_col: str) -> bool:
    for message in dct[message_col]:
        if not has_chinese(message["content"]):
            return False
    return True


def num_chars(
    dct: dict,
    message_col: str = "messages",
    num_user_max_chars: int = 100,
    num_assistant_max_chars: int = float("inf"),
) -> bool:
    for message in dct[message_col]:
        if message["role"] == "user":
            if len(message["content"]) > num_user_max_chars:
                return False
        else:
            if len(message["content"]) > num_assistant_max_chars:
                return False
    return True


def banned_words(dct: dict, message_col: str = "messages", banned_words: list[str] = None) -> bool:
    messages = dct[message_col]
    banned_words = banned_words or [
        "文章",
        "文中",
        "本文",
        "上述",
        "以上提供",
        "根據上面",
        "上文",
        "引用",
        "無法回答",
    ]
    for message in messages:
        if any(word in message["content"] for word in banned_words):
            return False
    return True


def user_aassistant_exclusive(dct: dict, message_col: str = "messages") -> bool:
    messages = dct[message_col]
    for i in range(0, len(messages), 2):
        user, assistant = messages[i], messages[i + 1]
        if user["content"] in assistant["content"] or assistant["content"] in user["content"]:
            return False
    return True
