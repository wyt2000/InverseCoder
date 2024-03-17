import jsonlines

with open('magicoder_data/starcoderdata_cleaned_with_stars.jsonl') as f:
    data_with_stars = list(f.readlines())

with open('magicoder_data/starcoderdata_evol_by_wizardcoder_gpt4_0314.jsonl') as f:
    data_with_evol = list(f.readlines())

with jsonlines.open('magicoder_data/starcoderdata_evol_by_wizardcoder_gpt4_0314_filtered_by_stars.jsonl', mode='w') as writer:
    for star_data, evol_data in zip(data_with_stars, data_with_evol):
        star_data = eval(star_data)
        evol_data = eval(evol_data)
        if star_data['stars'] > 0:
            writer.write(evol_data)