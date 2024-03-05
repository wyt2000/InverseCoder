import jsonlines                                                                                       
                                                                                                       
with open('magicoder_data/codevol-wizardcoder-summary-0305.jsonl') as f:                               
	instructions = [eval(line)['response'].strip() for line in f.readlines()]                      
with open('magicoder_data/codevol-wizardcoder-0304.jsonl') as f:                                       
	responses = [eval(line)['response'].strip() for line in f.readlines()]                         
                                                                                                       
with jsonlines.open('magicoder_data/codevol-wizardcoder-instruction-data-0305.jsonl', mode='w') as writer:                                                                                                    
	for inst, resp in zip(instructions, responses):                                                    
		line = {                                                                                       
			'instruction' : f'Write code that satisfies the following purposes:\n{inst}',              
			'response' : resp                                                                          
		}                                                                                              
	writer.write(line)                                                                             
