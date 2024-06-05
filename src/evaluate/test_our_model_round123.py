from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
import argparse
import os
import re

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def ask_llama(tokenizer, model, query):
    try:
        inputs = tokenizer(query, return_tensors="pt")
        # Generate
        generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=2048)
        answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer
    except Exception as exc:
        print(exc)
        return 'broken'


def cut_pattern(count_index, string, pattern):
    count = 0
    for m in re.finditer(pattern, string):
        count += 1
        if count == count_index:
            return string[m.end():]
    return string


def response_round(p_index, round_index, query, tokenizer, model):
    response_task = ask_llama(tokenizer=tokenizer, model=model, query=query)
    while response_task == "broken":
        response_task = ask_llama(tokenizer=tokenizer, model=model, query=query)
    response_task = cut_pattern(count_index=(round_index + 1) * 2,
                                string=response_task,
                                pattern='INST]')
    print('Psychologist ' + str(p_index) + ': ' + response_task)
    return response_task


def ask_round1_model(p_round1):
    MODEL_PATH = '/data/xiaomengxi/psychology/BELLE-main/train/round1_output/round1_432'
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
    prompt0 = (
            "You are a psychologist helping patients to separate the situation from their thought."
            + "I will give you the patient's thought and you need to generate a reply to guide them to separate "
              "the situation from their thought. Your reply should be short and in one paragraph."
            + "\nThe patient's thought: " + p_round1
            + "\nYour reply:"
    )
    query0 = ("<s>[INST] <<SYS>> \nYou are a psychologist. \n<</SYS>>"
              + prompt0 + '[/INST]')
    response0 = response_round(p_index=0, round_index=0, query=query0,
                               tokenizer=tokenizer, model=model)
    return response0


def ask_round2_model(p_round1, d_round1, p_round2):
    MODEL_PATH = '/data/xiaomengxi/psychology/BELLE-main/train/round1_output/round2_432'
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
    prompt1 = (
            "You are a psychologist helping patients to separate the situation from their thought."
            + "I will give you the patient's thought and you need to generate a reply to guide them to separate "
              "the situation from their thought. Your reply should be short and in one paragraph."
            + "\nThe patient's thought: " + p_round1
            + "\nYour reply:"
    )
    prompt2 = (
            "You are a psychologist helping patients to stop being trapped in negative thoughts, by brainstorming other possibilities under the same situation."
            + "I will give you the patient's thought and you need to generate a reply to guide them to brainstorm. "
              "Your reply should be short and in one paragraph."
            + "\nThe patient's thought: " + p_round2
            + "\nYour reply:"
    )
    query0 = ("<s>[INST] <<SYS>> \nYou are a psychologist. \n<</SYS>>"
              + prompt1 + '[/INST]'
              + d_round1 + '[/INST]'
              + prompt2 + '[/INST]')
    response0 = response_round(p_index=1, round_index=1, query=query0,
                               tokenizer=tokenizer, model=model)
    return response0


def ask_round3_model(p_round1, d_round1, p_round2, d_round2, p_round3):
    MODEL_PATH = '/data/xiaomengxi/psychology/BELLE-main/train/round1_output/round3_480'
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
    prompt1 = (
            "You are a psychologist helping patients to separate the situation from their thought."
            + "I will give you the patient's thought and you need to generate a reply to guide them to separate "
              "the situation from their thought. Your reply should be short and in one paragraph."
            + "\nThe patient's thought: " + p_round1
            + "\nYour reply:"
    )

    prompt2 = (
            "You are a psychologist helping patients to stop being trapped in negative thoughts, by brainstorming other possibilities under the same situation."
            + "I will give you the patient's thought and you need to generate a reply to guide them to brainstorm. "
              "Your reply should be short and in one paragraph."
            + "\nThe patient's thought: " + p_round2
            + "\nYou reply:"
    )

    prompt3 = (
            "You are a psychologist. A patient asks you for help. After brainstorming, the patient has changed his/her thought."
            "Please generate the final reply to the patient with empathy and persuasion. You should first recognize the patient's brainstorming and your reply should be short and in one paragraph."
            + "\nThe patient's original thought: " + p_round1
            + "\n The patient's brainstorming: " + p_round3
            + "\nYour reply:"
    )
    query0 = ("<s>[INST] <<SYS>> \nYou are a psychologist. \n<</SYS>>"
              + prompt1 + '[/INST]'
              + d_round1 + '[/INST]'
              + prompt2 + '[/INST]'
              + d_round2 + '[/INST]'
              + prompt3 + '[/INST]'
              )
    response0 = response_round(p_index=1, round_index=2, query=query0,
                               tokenizer=tokenizer, model=model)
    return response0


def generate_round1_our_model():
    train_df = pd.read_csv("../../data/test/gpt_round_1.csv",
                           encoding='gbk')
    # toggle index
    thinking_pattern_list = train_df['thinking_trap'].values.tolist()
    thought_list = train_df['thought'].values.tolist()
    patient_round1_list = train_df['patient_round1'].values.tolist()

    MODEL_PATH = '/data/xiaomengxi/psychology/BELLE-main/train/round1_output/round2_432'
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')

    for i in range(0, len(patient_round1_list)):
        prompt0 = (
                "You are a psychologist helping patients to separate the situation from their thought."
                + "I will give you the patient's thought and you need to generate a reply to guide them to separate "
                  "the situation from their thought. Your reply should be short and in one paragraph."
                + "\nThe patient's thought: " + patient_round1_list[i]
                + "\nYour reply:"
        )

        prompt0 = (
                "You are a psychologist helping patients to stop being trapped in negative thoughts, by brainstorming other possibilities under the same situation."
                + "I will give you the patient's thought and you need to generate a reply to guide them to brainstorm. "
                  "Your reply should be short and in one paragraph."
                + "\nThe patient's thought: " + patient_round1_list[i]
                + "\nYour reply:"
        )

        query0 = ("<s>[INST] <<SYS>> \nYou are a psychologist. \n<</SYS>>"
                  + prompt0 + '[/INST]')
        response0 = response_round(p_index=i, query=query0,
                                   tokenizer=tokenizer, model=model)

        df1 = pd.DataFrame({'index': [i],
                            'thinking_trap': [thinking_pattern_list[i]],
                            'thought': [thought_list[i]],
                            'patient_round1': [patient_round1_list[i]],
                            'psychologist_round1': [response0]})
        # df1.to_csv('data/gpt4_generation/511_gpt4_generation_851_3000.csv', mode='a', index=False, header=False)
        df1.to_csv('our_model_round_2.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    # generate_round1_our_model()
    # p_round1 = "I've been feeling really overwhelmed lately. My friend Lily, someone very close to me, just passed away in a car accident. It's been extremely difficult for me to cope with this loss. I feel so sad and helpless, like there's nothing I can do to change what happened or how I'm feeling."
    # d_round1 = "I'm truly sorry to hear about your loss. It's completely understandable that you're feeling overwhelmed and sad. Let's try to separate the situation from your thoughts. The situation is that your friend Lily passed away in a car accident. Your thoughts are that there's nothing you can do to change this or how you're feeling. Is that correct?"
    # p_round2 = "Yes, that's correct. I keep thinking that there\'s nothing I can do to change what happened or how I feel about it. It feels like I'm stuck in these feelings of sadness and helplessness. I\'m struggling to find a way to move forward or to deal with these emotions effectively."
    # ask_round1_model(p_round1)
    print("Input your thought: ")
    p_r1 = input()
    d_r1 = ask_round1_model(p_round1=p_r1)
    p_r2 = input()
    d_r2 = ask_round2_model(p_round1=p_r1, d_round1=d_r1, p_round2=p_r2)
    p_r3 = input()
    ask_round3_model(p_round1=p_r1, d_round1=d_r1, p_round2=p_r2,
                     d_round2=d_r2, p_round3=p_r3)
