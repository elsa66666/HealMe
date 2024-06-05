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


def response_round(p_index, query, tokenizer, model):
    response_task = ask_llama(tokenizer=tokenizer, model=model, query=query)
    while response_task == "broken":
        response_task = ask_llama(tokenizer=tokenizer, model=model, query=query)
    response_task = cut_pattern(count_index=(0 + 1) * 2,
                                string=response_task,
                                pattern='INST]')
    print('Psychologist ' + str(p_index) + response_task)
    return response_task


def ask_our_model_round3(MODEL_PATH, p_round1, p_round3):
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
    prompt0 = (
            "You are a psychologist. A patient asks you for help. After brainstorming, the patient has changed his/her thought."
            "Please generate the final reply to the patient with empathy and persuasion. You should first recognize the patient's brainstorming and your reply should be short and in one paragraph."
            + "\nThe patient's original thought: " + p_round1
            + "\n The patient's brainstorming: " + p_round3
            + "\nYour reply:"
    )
    query0 = ("<s>[INST] <<SYS>> \nYou are a psychologist. \n<</SYS>>"
              + prompt0 + '[/INST]')
    response0 = response_round(p_index=0, query=query0,
                               tokenizer=tokenizer, model=model)
    return response0


def generate_round1_our_model(MODEL_PATH):
    r1_df = pd.read_csv("../../data/test/gpt_round_1.csv",
                           encoding='gbk')
    # toggle index
    thinking_pattern_list = r1_df['thinking_trap'].values.tolist()
    thought_list = r1_df['thought'].values.tolist()
    patient_round1_list = r1_df['patient_round1'].values.tolist()

    r3_df = pd.read_csv("../../data/test/gpt_round_3.csv")
    patient_round3_list = r3_df['patient_round3'].values.tolist()

    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')

    for i in range(len(patient_round1_list)):
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
                + "\nYou reply:"
        )

        prompt0 = (
                "You are a psychologist. A patient asks you for help. After brainstorming, the patient has changed his/her thought."
                "Please generate the final reply to the patient with empathy and persuasion. You should first recognize the patient's brainstorming and your reply should be short and in one paragraph."
                + "\nThe patient's original thought: " + patient_round1_list[i]
                + "\n The patient's brainstorming: " + patient_round3_list[i]
                + "\nYour reply:"
        )

        # print("Patient "+str(i))
        query0 = ("<s>[INST] <<SYS>> \nYou are a psychologist. \n<</SYS>>"
                  + prompt0 + '[/INST]')
        response0 = response_round(p_index=i, query=query0,
                                   tokenizer=tokenizer, model=model)

        df1 = pd.DataFrame({'index': [i],
                            'thinking_trap': [thinking_pattern_list[i]],
                            'thought': [thought_list[i]],
                            'patient_round1': [patient_round1_list[i]],
                            'psychologist_round1': [response0]})
        df1.to_csv('our_model_round_3.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    # generate_round1_our_model()
    p_r1 = "I've been feeling really overwhelmed lately. My friend Lily, someone very close to me, just passed away in a car accident. It's been extremely difficult for me to cope with this loss. I feel so sad and helpless, like there's nothing I can do to change what happened or how I'm feeling."
    p_r3 = "I appreciate your suggestion, but honestly, it's hard for me to think of anything positive right now. I've tried to come up with ways to honor Lily's memory, but all I can think about is how she's not here anymore. Every idea just reminds me of her absence. It feels like celebrating her life or remembering the good times just highlights the fact that she's gone, and it makes me feel even sadder. I'm not sure how to move past these negative feelings."
    ask_our_model_round3(p_round1=p_r1, p_round3=p_r3)
