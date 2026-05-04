import json
import re

path = 'output/AlphaEdit_qwen3-4b_MQuAKE-CF-3k-v2-clke_en_metrics.json'

with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 逐条解析，跳过损坏的case
data = []
# 找所有case_id位置
positions = [m.start() for m in re.finditer(r'\{\s*"case_id"', content)]

for i, pos in enumerate(positions):
    end = positions[i+1] if i+1 < len(positions) else len(content)
    chunk = content[pos:end].rstrip(',\n ]')
    try:
        case = json.loads(chunk)
        data.append(case)
    except:
        pass

print(f'Successfully parsed cases: {len(data)}')

if data:
    hops = sum(x for d in data for pair in d['post']['hop_wise'] for x in pair)
    total = sum(len(pair) for d in data for pair in d['post']['hop_wise'])
    acc = sum(x for d in data for x in d['post']['accuracy'])
    total_acc = sum(len(d['post']['accuracy']) for d in data)
    print(f'Hop-wise (single-hop): {hops/total:.3f}')
    print(f'Accuracy (multi-hop):  {acc/total_acc:.3f}')
