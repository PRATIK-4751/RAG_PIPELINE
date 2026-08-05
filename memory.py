class Memory:
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        # keep last max_turns*2 messages (user + assistant)
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]

    def clear(self):
        self.messages = []

    def as_text(self):
        lines = []
        for m in self.messages:
            who = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{who}: {m['content']}")
        return "\n".join(lines)

    def last_user(self):
        for m in reversed(self.messages):
            if m["role"] == "user":
                return m["content"]
        return None
