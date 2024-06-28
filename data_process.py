import json
data_path = '/mnt/bn/bohanzhainas1/bohan/gpt4v_famliy/LLaVA_versions/llava_gpt4v_generate_v1.json'

data = json.load(open(data_path))

for d in data:
    if 'image' in d:
        for i in range(len(d['conversations'])):
            conv = d['conversations'][i]['value'].replace('<image>', '').strip()
            conv = conv.replace('\\n', '').strip()
            conv = conv.replace('\\n<image>', '').strip()
            d['conversations'][i]['value'] = conv
        
        d['conversations'][0]['value'] = '<image>\n' + d['conversations'][0]['value']

with open("/mnt/bn/bohanzhainas1/bohan/gpt4v_famliy/LLaVA_versions/llava_gpt4v_generate_v1_3.json", 'w') as outfile:
    json.dump(data, outfile, indent=4)
