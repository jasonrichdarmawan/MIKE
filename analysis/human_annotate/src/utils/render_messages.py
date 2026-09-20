def render_messages(messages):
    for message in messages:
        role = message["role"]
        content = message["content"]
        print(f"{role}: {content}\n")