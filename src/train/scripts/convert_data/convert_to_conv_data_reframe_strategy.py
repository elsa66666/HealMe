import argparse
import json

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
        default='F:\psychology\openai-quickstart-python\dataset\\5.1.2\\[deprecated]finetune\\5_1_2_whole_data.json'
    )
    parser.add_argument(
        "--write_data",
        default='512reframe_strategy_conv.json'
    )
    parser.add_argument(
        "--dataset_name",
        default='512reframe_strategy'
    )
    args = parser.parse_args()
    f_write = open(args.write_data, "w")
    with open(args.orig_data) as f:
        lines = f.readlines()
        num_id = 1
        for line in lines:
            data = json.loads(line)
            human1 = ("You are a psychologist. A patient asks you for help. "
                      + "I will offer you the patient's thought."
                      + "Please reply to the patient with empathy and persuasion. "
                      + "Let's generate the reply step by step. "
                      + "Can you understand the task?")
            patient_thought = data['original_thought']
            reframe_strategy = data['reframe_strategy']
            human3 = ("At first, you need to reframe the patient's thought."
                      + "I will offer you the patient's thought and a list of reframe strategies. "
                      + "\n Please choose the most appropriate strategy and reframe the patient's thought."
                      + "\n The patient's thought: \n"
                      + patient_thought
                      + "\n List of reframe strategies: \n"
                      + "[Impermanence]: means understanding that negative experiences are finite and temporary, and that others have also experienced or even overcome similar forms of adversity.\n"
                      + "[Growth Mindset]: is the belief that one's skills and abilities are not immutable but can instead be changed and improved over time.\n"
                      + "[Optimism]: does not mean to negate or deny the negative aspects of a situation, but instead to shift the emphasis to the more positive aspects of the situation, including expectations for a bright future.\n"
                      + "[Self-Affirmation]: means to assert a more holistic or expansive version of oneself by listing one's values, skills, and positive characteristics.\n"
                      + "[Neutralizing]: involves removing or rewriting negative phrases and terms, so they are more neutral.\n"
                      + "Your choice of the reframe strategy and the reframed thought:")
            reframed_thought = data['reframed_thought']
            reply3 = ("The reframe strategy is: "
                      + reframe_strategy
                      + "\nThe reframed thought is: \n"
                      + reframed_thought)

            human4 = ("Then you can generate your reply to the patient. \n"
                      + "Please incorporate the reframed thought with empathy and persuasion in your reply. \n"
                      + "And as a psychologist you cannot deny the patient's thought. \n"
                      + "Your reply should be short. \n"
                      + "Your reply:")
            response = data['response']
            conversations = [{"from": "human", "value": human1},
                             {"from": "assistant", "value": 'Yes.'},
                             {"from": "human", "value": human3},
                             {"from": "assistant", "value": reply3},
                             {"from": "human", "value": human4},
                             {"from": "assistant", "value": response},
                             ]
            # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
            uniq_id = data['id'] if "id" in data else args.dataset_name + "-" + str(num_id)
            item = {"id": uniq_id, "conversations": conversations}
            f_write.write(json.dumps(item, ensure_ascii=False) + "\n")
            num_id += 1
    f_write.close()


if __name__ == "__main__":
    main()
