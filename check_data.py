import json

data = json.load(open('./datasets/MQuAKE-CF-3k-v2-clke.json', 'r', encoding='utf-8'))

count = 0
for d in data:
    for lang in ['', '_zh', '_de', '_fr', '_ja']:
        # Check new_single_hops
        key = f'new_single_hops{lang}'
        for h in d.get(key, []):
            if 'answer_alias' not in h:
                print(f"Missing answer_alias in single_hops: case_id={d['case_id']}, lang={lang or 'en'}")
                count += 1
            if 'answer' not in h:
                print(f"Missing answer in single_hops: case_id={d['case_id']}, lang={lang or 'en'}")
                count += 1
            if 'cloze' not in h:
                print(f"Missing cloze in single_hops: case_id={d['case_id']}, lang={lang or 'en'}")
                count += 1
            if 'question' not in h:
                print(f"Missing question in single_hops: case_id={d['case_id']}, lang={lang or 'en'}")
                count += 1

        # Check top-level fields
        for field in [f'new_answer{lang}', f'new_answer_alias{lang}', f'cloze_question{lang}', f'questions{lang}']:
            if field not in d:
                print(f"Missing {field}: case_id={d['case_id']}")
                count += 1

print(f"\nTotal missing fields: {count}")
print("Done")
