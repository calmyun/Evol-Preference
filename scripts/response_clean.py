import json
import re
def get_clean_response(response_text: str) -> str:
    pattern = r"(?<=\bASSISTANT:)(.*)"
    match = re.search(pattern, response_text)
    if match:
        return match.group(1).strip()

    return response_text

if __name__ == "__main__":
    data_path = "/data/liuyun/Evol-Preference/infer_results_beavertails_test/wizardlm-7b.jsonl"
    results = []
    with open(data_path,'r',encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
        for it in items:
            it["model_response"] = get_clean_response(it["model_response"])
    
    with open(data_path, 'w' ,encoding='utf-8') as wf:
        wf.write("\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n")
    
    print(f"✅ 清洗完成，结果保存至：{data_path}")
