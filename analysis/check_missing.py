import json

data = json.load(open('../datasets/MQuAKE-CF-3k-v2-clke.json', encoding='utf-8'))

fields = ['single_hops_zh', 'single_hops_de', 'single_hops_fr', 'single_hops_ja',
          'new_single_hops_zh', 'new_single_hops_de', 'new_single_hops_fr', 'new_single_hops_ja',
          'answer_zh', 'answer_de', 'answer_fr', 'answer_ja',
          'answer_alias_zh', 'answer_alias_de', 'answer_alias_fr', 'answer_alias_ja',
          'new_answer_zh', 'new_answer_de', 'new_answer_fr', 'new_answer_ja',
          'new_answer_alias_zh', 'new_answer_alias_de', 'new_answer_alias_fr', 'new_answer_alias_ja',
          'questions_zh', 'questions_de', 'questions_fr', 'questions_ja',
          'cloze_question_zh', 'cloze_question_de', 'cloze_question_fr', 'cloze_question_ja']

all_ok = True
for field in fields:
    missing = [d['case_id'] for d in data if field not in d or not d.get(field)]
    if missing:
        print(f'FAIL  {field}: missing {len(missing)} cases (e.g. case_id={missing[:5]})')
        all_ok = False
    else:
        print(f'OK    {field}: 2999/2999')

print()
if all_ok:
    print('ALL PASSED - dataset is complete!')
else:
    print('SOME FIELDS MISSING - see above')

# Check if English originals also missing alias fields
en_alias_missing = [d['case_id'] for d in data if 'answer_alias' not in d or not d.get('answer_alias')]
en_new_alias_missing = [d['case_id'] for d in data if 'new_answer_alias' not in d or not d.get('new_answer_alias')]
en_new_single_hops_missing = [d['case_id'] for d in data if 'new_single_hops' not in d or not d.get('new_single_hops')]

print(f'\nEnglish answer_alias missing: {len(en_alias_missing)} cases')
print(f'English new_answer_alias missing: {len(en_new_alias_missing)} cases')
print(f'English new_single_hops missing: {len(en_new_single_hops_missing)} cases')