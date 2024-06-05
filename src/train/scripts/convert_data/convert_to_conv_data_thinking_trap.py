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
        default='512thinking_trap_conv.json'
    )
    parser.add_argument(
        "--dataset_name",
        default='512thinking_trap'
    )
    args = parser.parse_args()
    f_write = open(args.write_data,"w")
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
            human2 = ("At first, you should figure out the thinking trap of the patient. "
                    + "I will offer you the patient's thought and a list of thinking traps. "
                    + "\n Please choose the most appropriate thinking trap for the patient."
                    + "\n The patient's thought: \n" 
            + patient_thought 
            + "\n List of thinking traps: \n" 
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
            + "Your choice of the thinking trap:")

            thinking_trap = data['thinking_pattern']
            if thinking_trap == 'Jumping to conclusions: mind reading':
                thinking_trap = 'Mind reading'
            if thinking_trap == 'Jumping to conclusions: Fortune-telling':
                thinking_trap = 'Fortune telling'
            if thinking_trap == 'Black-and-white or polarized thinking / All or nothing thinking':
                thinking_trap = 'All-or-Nothing Thinking'

            reframe_strategy = data['reframe_strategy']
            human3 = ("Then you can reframe the patient's thought using "
            + reframe_strategy
            + " strategy, while considering the thinking trap. \n"
            + "Your reframed thought:")

            reframed_thought = data['reframed_thought']

            human4 = ("Then you can generate your reply to the patient. \n"
                + "Please reply to the patient with empathy and persuasion while considering his or her thinking trap. \n"
                + "And as a psychologist you cannot deny the patient's thought. \n"
                + "Your reply should be short. \n"
                + "Your reply:")
            response = data['response']
            conversations = [{"from": "human", "value": human1},
                             {"from": "assistant", "value": 'Yes.'},
                             {"from": "human", "value": human2},
                            {"from": "assistant", "value": thinking_trap},
                          #   {"from": "human", "value": human3},
                          #   {"from": "assistant", "value": reframed_thought},
                             {"from": "human", "value": human4},
                             {"from": "assistant", "value": response},
                             ]
            # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
            uniq_id = data['id'] if "id" in data else args.dataset_name+"-"+str(num_id)
            item = {"id":uniq_id, "conversations": conversations}
            f_write.write(json.dumps(item, ensure_ascii=False)+"\n")
            num_id += 1
    f_write.close()


if __name__ == "__main__":
    main()