import requests
import os

from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint, set_default_backend 
from sglang.srt.utils import load_image
from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server
from sglang.utils import launch_server_cmd

model = "./deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path " + model + " --trust-remote-code --tp 1 --host 0.0.0.0 --port 30000"
)

# 添加重试机制
import time
max_retries = 5
for i in range(max_retries):
    try:
        wait_for_server(f"http://localhost:{port}", timeout=120)
        break
    except Exception:
        if i == max_retries - 1:
            raise
        time.sleep(10)

set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))

@function
def basic_qa(s, question):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=512))

state = basic_qa("List 3 countries and their capitals.")
print_highlight(state["answer"])


@function
def multi_turn_qa(s):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user("Please give me a list of 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=512))
    s += user("Please give me another list of 3 countries and their capitals.")
    s += assistant(gen("second_answer", max_tokens=512))
    return s


state = multi_turn_qa()
print_highlight(state["first_answer"])
print_highlight(state["second_answer"])