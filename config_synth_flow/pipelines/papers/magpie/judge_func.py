from hanzidentifier import has_chinese, is_traditional
import json

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


global_banned_words = None
def banned_words(dct: dict, message_col: str = "messages", banned_words: list[str] = None, from_file: str = None) -> bool:
    global global_banned_words
    if global_banned_words is None:
        if from_file:
            print(f"Loading banned words from {from_file}")
            with open(from_file) as f:
                global_banned_words = json.load(f)
        else:
            global_banned_words = []
    
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
        "這項",
        "背景知識",
        "根據資料",
        "背景內容",
        "背景資料",
        "提供的資料",
        "以上內容",
        "根據上列",
    ]
    banned_words = global_banned_words + banned_words
    
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
