import argparse
import json
import pandas as pd

'''
orig_data: {"instruction": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？", "input": "", "output": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"}
convert: {
    "id": xxx,
    "conversations":[
        {"from": "human", "value": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？"},
        {"from": "assistant", "value": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"},
    ]
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_data",
        default='../../../../openai-quickstart-python/dataset/5.1.2/train/conversation_0_199.csv'
    )
    parser.add_argument(
        "--write_data",
        default='conversation_0_199_conv.json'
    )
    parser.add_argument(
        "--dataset_name",
        default='512conversation'
    )
    args = parser.parse_args()
    f_write = open(args.write_data, "w")

    train_df = pd.read_csv("../../../../dataset/5.1.2/train/conversation_0_199.csv",
                           encoding='utf-8')

    index_list = train_df["index"].values.tolist()
    thought_list = train_df['thought'].values.tolist()
    thinking_trap_list = train_df['thinking_trap'].values.tolist()
    p_round1_list = train_df['patient_round1'].values.tolist()
    d_round1_list = train_df['doctor_round1'].values.tolist()
    p_round2_list = train_df['patient_round2'].values.tolist()
    d_round2_list = train_df['doctor_round2'].values.tolist()
    p_round3_list = train_df['patient_round3'].values.tolist()
    d_round3_list = train_df['polished_doctor_round3'].values.tolist()

    analyse_thinking_trap_df = pd.read_csv(
        "../../../../dataset/5.1.2/train/analyse_thinking_trap.csv", encoding='utf-8')
    analyse_thinking_trap_list = analyse_thinking_trap_df['analyse_thinking_trap'].values.tolist()

    for i in range(len(index_list)):
        prompt0 = (
                "You are a psychologist practicing to use 'coginitve reframing' strategy to help patients to reframe their negative emotion. "
                + "I will play the role of the patient. You should follow these three steps:\n"
                + "1.At first, the patient will express their negative emotion. And you need to guide them to seperate their situation and thought.\n"
                + "2.Then the patient will reply to you. And you need to guide them to brainstorm other possible posibilities under the same situation.\n"
                + "3.When the patient show you other posibilities, them will not be trapped in their negative thoughts. That is what 'cognitive reframing' means. And you need to combine their reframed thought(if their reframed thought is valuable) with empathy and persuation to give them the final reply.\n"
                + "Reply 'Yes, I understand the task'. And I will start the conversation with you.")
        d_round0 = 'Yes, I understand the task.'
        conversations = [
            {"from": "human", "value": prompt0},  # 讲述task
            {"from": "assistant", "value": d_round0},  # understand the task
            {"from": "human", "value": "The patient's thought:\n" + p_round1_list[i]
                                       + "\nYou need to guide them to separate their situation and "
                                         "thought.\n Your response:"},
            # patient express feeling
            {"from": "assistant", "value": d_round1_list[i]},  # doctor ask separate
            {"from": "human", "value": "The patient's reply:\n" + p_round2_list[i]
                                       + "\nAnd you need to guide them to brainstorm other possible "
                                         "posibilities under the same situation.\n Your response:"},
            # patient separate
            {"from": "assistant", "value": d_round2_list[i]},  # doctor ask brainstorm
            {"from": "human", "value": "The patient's reply:\n" + p_round3_list[
                i] + "\nGenerate the final reply with empathy and persuasion. "
                     "Your reply should be short and in one paragraph. \nYour reply:"},
            {"from": "assistant", "value": d_round3_list[i]}
        ]
        uniq_id = args.dataset_name + "-" + str(index_list[i])
        item = {"id": uniq_id, "conversations": conversations}
        f_write.write(json.dumps(item, ensure_ascii=False) + "\n")

        thinking_trap_analysis = analyse_thinking_trap_list[i].lstrip('"').rstrip('"')
        analysis_json = {'thinking_trap': thinking_trap_list[i], 'analysis': thinking_trap_analysis}
        prompt = ("You are a psychologist analyzing the thinking trap of the patients. "
                  + "I will offer you the patient's thought and a list of thinking traps. "
                  + "\n Please choose the most appropriate one from the list and provide the reason why the patient is stuck in the thinking trap. And the reason should be short and in one paragraph."
                  + "\nThen return the information the the following JSON format: {'thinking_trap': xxx, 'analysis': xxx}"
                  + "Note xxx is placeholder."
                  + "\n The patient's thought: \n"
                  + thought_list[i] + "\n List of thinking traps: \n"
                  + "[Catastrophizing]: by giving greater weight to the worst possible outcome. \n "
                  + "[Discounting the positive]: experiences by insisting that they \"don\'t count\". \n"
                  + "[Overgeneralization]: Overgeneralization is making faulty generalizations from insufficient evidence. \n"
                  + "[Personalization]: assigning a disproportionate amount of personal blame to oneself. \n"
                  + "[All-or-Nothing Thinking]: viewing things as either good or bad and nothing in between. \n"
                  + "[Mental filtering]: occurs when an individual dwells only on the negative details of a situation. \n"
                  + "[Mind reading]: inferring a person's probable (usually negative) thoughts from their behavior. \n"
                  + "[Fortune telling]: is predicting outcomes (usually negative) of events. \n"
                  + "[Should statements]: where a person demands particular behaviors regardless of the realistic circumstances. \n"
                  + "[Labeling and mislabeling]: attributing a person's actions to their character rather than the situation. \n"
                  + "Your return:\n"
                  )
        conversations = [
            {"from": "human", "value": prompt},
            {"from": "assistant", "value": str(analysis_json)}]
        uniq_id = "512thinking_trap_analysis-" + str(index_list[i])
        item = {"id": uniq_id, "conversations": conversations}
        f_write.write(json.dumps(item, ensure_ascii=False) + "\n")

    f_write.close()


if __name__ == "__main__":
    main()
