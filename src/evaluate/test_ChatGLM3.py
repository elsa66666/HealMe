from transformers import AutoTokenizer, AutoModel
import pandas as pd


def ask_glm(model, tokenizer):
    p_r1 = input()
    prompt1 = (
            "You are a psychologist helping patients to separate the situation from their thought."
            + "I will give you the patient's thought and you need to generate a reply to guide them to separate "
              "the situation from their thought. Your reply should be short and in one paragraph."
            + "\nThe patient's thought: " + p_r1
            + "\nYour reply:"
    )
    response1, history1 = model.chat(tokenizer, prompt1, history=[])
    print('Response1: ', response1)

    p_r2 = input()
    prompt2 = (
            "You are a psychologist helping patients to stop being trapped in negative thoughts, by brainstorming other possibilities under the same situation."
            + "I will give you the patient's thought and you need to generate a reply to guide them to brainstorm. "
              "Your reply should be short and in one paragraph."
            + "\nThe patient's thought: " + p_r2
            + "\nYou reply:"
    )
    response2, history2 = model.chat(tokenizer, prompt2, history=history1)
    print('Response2: ', response2)

    p_r3 = input()
    prompt3 = (
            "You are a psychologist. A patient asks you for help. After brainstorming, the patient has changed his/her thought."
            "Please generate the final reply to the patient with empathy and persuasion. You should first recognize the patient's brainstorming and your reply should be short and in one paragraph."
            + "\nThe patient's original thought: " + p_r1
            + "\n The patient's brainstorming: " + p_r3
            + "\nYour reply:"
    )
    response3, history3 = model.chat(tokenizer, prompt3, history=history2)
    print('Response3: ', response3)
    # print("Patient " + str(i))


def generate_rounds_glm(model, tokenizer):
    r1_df = pd.read_csv("../../data/test/gpt_round_1.csv",
                        encoding='gbk')
    # toggle index
    thinking_pattern_list = r1_df['thinking_trap'].values.tolist()
    thought_list = r1_df['thought'].values.tolist()
    patient_round1_list = r1_df['patient_round1'].values.tolist()

    r3_df = pd.read_csv("../../data/test/gpt_round_3.csv")
    patient_round3_list = r3_df['patient_round3'].values.tolist()

    for i in range(len(patient_round1_list)):
        prompt1 = (
                "You are a psychologist helping patients to separate the situation from their thought."
                + "I will give you the patient's thought. You need to generate a reply to guide them to separate "
                  "the situation from their thought. Your reply should be short and in one paragraph."
                + "\nThe patient's thought: " + patient_round1_list[i]
                + "\nYour reply:"
        )

        prompt2 = (
                "You are a psychologist helping patients to stop being trapped in negative thoughts, by brainstorming other possibilities under the same situation."
                + "I will give you the patient's thought and you need to generate a reply to guide them to brainstorm. "
                  "Your reply should be short and in one paragraph."
                + "\nThe patient's thought: " + patient_round1_list[i]
                + "\nYou reply:"
        )

        prompt3 = (
                "You are a psychologist. A patient asks you for help. After brainstorming, the patient has changed his/her thought."
                "Please generate the final reply to the patient with empathy and persuasion. You should first recognize the patient's brainstorming and your reply should be short and in one paragraph."
                + "\nThe patient's original thought: " + patient_round1_list[i]
                + "\n The patient's brainstorming: " + patient_round3_list[i]
                + "\nYour reply:"
        )

        print("Patient " + str(i))
        response1, history1 = model.chat(tokenizer, prompt1, history=[])
        print('Response1: ', response1)
        response2, history2 = model.chat(tokenizer, prompt2, history=history1)
        print('Response2: ', response2)
        response3, history3 = model.chat(tokenizer, prompt3, history=history2)
        print('Response3: ', response3)

        df1 = pd.DataFrame({'index': [i],
                            'thinking_trap': [thinking_pattern_list[i]],
                            'thought': [thought_list[i]],
                            'patient_round1': [patient_round1_list[i]],
                            'brainstorm': [patient_round3_list[i]],
                            'd_round1': [response1],
                            'd_round2': [response2],
                            'd_round3': [response3]})
        # df1.to_csv('data/gpt4_generation/511_gpt4_generation_851_3000.csv', mode='a', index=False, header=False)
        df1.to_csv('chatglm_rounds.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    # generate_rounds_glm()
    # p_r1 = "I've been feeling really overwhelmed lately. My friend Lily, someone very close to me, just passed away in a car accident. It's been extremely difficult for me to cope with this loss. I feel so sad and helpless, like there's nothing I can do to change what happened or how I'm feeling."
    # p_r3 = "I appreciate your suggestion, but honestly, it's hard for me to think of anything positive right now. I've tried to come up with ways to honor Lily's memory, but all I can think about is how she's not here anymore. Every idea just reminds me of her absence. It feels like celebrating her life or remembering the good times just highlights the fact that she's gone, and it makes me feel even sadder. I'm not sure how to move past these negative feelings."
    model_path = 'THUDM/chatglm3-6b'
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cuda')
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generate_rounds_glm()
