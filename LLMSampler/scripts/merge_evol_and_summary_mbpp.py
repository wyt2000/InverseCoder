import jsonlines
import ast
import argparse

mbpp_asserts_path = '/lustre/S/wuyt/dataset/mbpp/mbpp-prompt.jsonl.asserts'

def main(
    sol_path:  str,
    inst_path: str,
    save_path: str
):
    # compose task_id and inst
    with open(sol_path) as f:
        responses = [
            (eval(line)['solution'].strip(), eval(line)['task_id'].strip())
            for line in f.readlines()
        ]
    with open(inst_path) as f:
        instructions = [
            eval(line)['response'].strip()
            for line in f.readlines()
        ]
    dataset = []
    with jsonlines.open(save_path, mode='w') as writer:	
        for inst, resp in zip(instructions, responses):
            code = resp
            line = {
                'task_id' : code[1],
                'instruction' : inst,
            }
            dataset.append(line)

    # compose inst with asserts -> prompt
    mbpp2ass = {}
    with open(mbpp_asserts_path) as f:
        for line in f.readlines():
            line = eval(line)
            mbpp2ass[line['task_id']] = line['prompt']
    mbpp2inst = {}
    for line in dataset:
        task_id = line['task_id']
        mbpp2inst[task_id] = line['instruction'] + '\nYour code should satisfy the following assertion:\n' + mbpp2ass[task_id]

    # save
    with jsonlines.open(save_path, mode='w') as writer:	
        for task, inst in mbpp2inst.items():
            line = {
                'task_id' : task,
                'prompt' : inst 
            }
            writer.write(line)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sol_path', type=str, required=True)
    parser.add_argument('--inst_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    main(args.sol_path, args.inst_path, args.save_path)

