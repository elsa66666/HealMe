import openai
import os

# set proxy
os.environ["HTTP_PROXY"] = "127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "127.0.0.1:7890"

openai.api_key = "Your openai key"

'''
an example:
rsp = openai.ChatCompletion.create(
        model="gpt-4",
        # model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "A psychologist."},  # 这里content是gpt的人设
            {"role": "user", "content": "What are you doing?"}
        ]
    )
'''


def get_gpt_response(gpt_version, messages):
    rsp = openai.ChatCompletion.create(
        # model="gpt-4",
        model=gpt_version,
        messages=messages
    )
    answer = rsp.get("choices")[0]["message"]["content"]
    if answer == 'broken':
        get_gpt_response(gpt_version, messages)
    return answer

