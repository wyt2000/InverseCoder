import random
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

prefixes = [
    'Write',
    'Create',
    'Implement',
    'Develop',
    'Design',
    'Build',
    'Construct'
]

def check(data):
    data['instruction'] = redecode(data['instruction'])
    data['response'] = redecode(data['response'])
    inst = data['instruction']
    resp = data['response']
    for prefix in prefixes:
        if inst.startswith(prefix):
            return True
        if inst.startswith(prefix.lower()):
            data['instruction'] = inst[0].upper() + inst[1:]
            return True
        if inst.startswith('Please ' + prefix.lower()):
            inst = inst[7:]
            data['instruction'] = inst[0].upper() + inst[1:]
            return True
        if inst.startswith('please ' + prefix.lower()):
            inst = inst[7:]
            data['instruction'] = inst[0].upper() + inst[1:]
            return True
    return False 

path = 'dataset/oss-instruct/data-oss_instruct-decontaminated-ins-rep.jsonl'

with open(path) as f:
    lines = list(f.readlines())
    dataset = []
    for line in lines:
        line = eval(line)
        if check(line):
            dataset.append({'instruction': line['instruction'], 'response': line['response']})
    with jsonlines.open(f'{path}.good_inst', mode='w') as writer:
        for line in dataset:
            writer.write(line)

