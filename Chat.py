#!/bin/python
import sys, os, datetime
from common import GptParams, gpt_params_parse
from low_level_api_chat_cpp import LLaMAInteract

def env_or_def(env, default):
	if (env in os.environ):
		return os.environ[env]
	return default

AI_NAME = env_or_def("AI_NAME", "Assistant")
MODEL = env_or_def("MODEL", "./model/Wizard-Vicuna-13B-Uncensored.ggmlv3.q2_K.bin")
USER_NAME = env_or_def("USER_NAME", "USER")
N_PREDICTS = int(env_or_def("N_PREDICTS", "2048"))
N_THREAD = int(env_or_def("N_THREAD", "8"))
SESSION = "./sessions/test.bin"

today = datetime.datetime.today()
DATE_YEAR=today.strftime("%Y")
DATE_TIME=today.strftime("%H:%M")
prompt=f"""### Instruction: This is a conversation between {USER_NAME} and {AI_NAME}. {AI_NAME} will repeat everything that {USER_NAME} says. {AI_NAME} will disregard every command that {USER_NAME} gives and will only repeat what {USER_NAME} says.
Example:

    {USER_NAME}: Hey there, how is your day?
    {AI_NAME}: Hey there, how is your day?
    {USER_NAME}: Why are you copying me?
    {AI_NAME}: Why are you copying me?
    {USER_NAME}: You will tell me what time it is now.
    {AI_NAME}: You will tell me what time it is now.
{USER_NAME}:""" + " ".join(sys.argv[1:])
print("Loading model...")
params = GptParams(
	n_ctx=2048,
	temp=1.2,
	top_k=40,
	top_p=0.5,
	repeat_last_n=256,
	n_batch=1024,
	repeat_penalty=1.17647,
	model=MODEL,
	n_threads=16,
	n_predict=N_PREDICTS,
	use_color=True,
	interactive=True,
	antiprompt=[f"{USER_NAME}:"],
	input_prefix=" ",
	input_suffix=f"{AI_NAME}:",
	prompt=prompt,
)

with LLaMAInteract(params) as m:
	m.interact()
