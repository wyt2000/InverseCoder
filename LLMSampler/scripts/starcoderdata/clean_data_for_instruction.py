import jsonlines

with open('magicoder_data/starcoderdata_cleaned_with_stars.jsonl') as f:
    data_with_stars = list(f.readlines())

with open('magicoder_data/starcoderdata_evol_by_wizardcoder_gpt4_0314.jsonl') as f:
    data_with_evol = list(f.readlines())

stars = set()
no_stars = set()

for star_data, evol_data in zip(data_with_stars, data_with_evol):
    star_data = eval(star_data)
    evol_data = eval(evol_data)
    evol_code = evol_data['response']
    if star_data['stars'] > 100:
        stars.add(evol_code)
    else:
        no_stars.add(evol_code)

with open('magicoder_data/starcoderdata_evol_by_wizardcoder_gpt4_instructed_by_wizardcoder_reversed_0306_0314_cleaned.jsonl') as f:
    final_data = list(f.readlines())

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with jsonlines.open('magicoder_data/starcoderdata_evol_by_wizardcoder_gpt4_instructed_by_wizardcoder_reversed_0306_0314_cleaned_filtered_by_stars_100.jsonl', mode='w') as writer:
    for data in final_data:
        data = eval(data)
        if data['response'] in stars:
            data['instruction'] = redecode(data['instruction'])
            data['response'] = redecode(data['response'])
            writer.write(data)

