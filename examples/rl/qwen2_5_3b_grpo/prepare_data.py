import json
from pathlib import Path

from datasets import load_dataset


SYSTEM_PROMPT = """Respond in the following format:
<think>
step-by-step reasoning about the problem
</think>
answer
"""


def extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return text.strip()
    return text.split("####", 1)[1].strip()


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "gsm8k_train.jsonl"

    print("Loading openai/gsm8k:main train split...")
    ds = load_dataset("openai/gsm8k", "main")["train"]

    print(f"Writing {len(ds)} records to {output_path} ...")
    with output_path.open("w") as f:
        for row in ds:
            record = {
                "system_prompt": SYSTEM_PROMPT,
                "prompt": row["question"],
                "answer": extract_hash_answer(row["answer"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
