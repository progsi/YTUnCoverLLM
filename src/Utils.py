import os

def read_textfile(path: str):

    with open(path, "r") as f:
        content = f.read()
    return content

def get_key(service: str):

    return read_textfile(os.path.join("keys", f"{service}.txt"))

