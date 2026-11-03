import re
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

SYSTEM_PROMPT = (
    "Bạn là chuyên gia trích xuất thông tin hành chính từ văn bản quy phạm pháp luật Việt Nam. "
    "Hãy đọc kỹ văn bản và trích xuất thông tin thành duy nhất một chuỗi JSON hợp lệ với đầy đủ 7 trường: "
    "loai_van_ban, so_ky_hieu, co_quan_ban_hanh, ngay_ban_hanh, trich_yeu, nguoi_ky, cong_bao."
)


class QwenFinetuneInference:
    def __init__(self, lora_path: str, base_model_id: str = BASE_MODEL_ID):
        print(f"[*] Đang tải Base Model: {base_model_id}")
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            load_in_4bit=True,
            device_map="auto",
        )

        adapter_path = Path(lora_path)
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Không tìm thấy adapter_config.json tại: {adapter_path}")

        print(f"[*] Đang nạp LoRA Adapter từ: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.eval()
        print(f"[✓] Sẵn sàng trên: {self.model.device}")

    def predict(self, messages: List[Dict[str, Any]], max_new_tokens: int = 1024) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return text[0].strip()

    def run_on_text(self, text_content: str, max_new_tokens: int = 1024) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": text_content.strip()}]},
        ]
        return self.predict(messages, max_new_tokens=max_new_tokens)


def parse_json_from_response(raw_text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```json\s*|```$", "", raw_text.strip(), flags=re.MULTILINE | re.IGNORECASE).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_response": raw_text, "error": "JSON parse error"}


def run(input_path: str, lora_path: str, output_dir: str,
        base_model_id: str = BASE_MODEL_ID, max_new_tokens: int = 1024):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = QwenFinetuneInference(lora_path=lora_path, base_model_id=base_model_id)

    if input_path.is_file() and input_path.suffix.lower() == ".jsonl":
        out_jsonl = output_dir / f"{input_path.stem}_finetune_pred.jsonl"
        with open(input_path, encoding="utf-8") as f_in, open(out_jsonl, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                item = json.loads(line)
                infer_msgs = [m for m in item.get("messages", []) if m.get("role") != "assistant"]
                raw = runner.predict(infer_msgs, max_new_tokens=max_new_tokens)
                item["prediction"] = parse_json_from_response(raw)
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[✓] Kết quả: {out_jsonl}")
        return

    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        txt_files = [input_path]
    elif input_path.is_dir():
        txt_files = sorted(input_path.glob("*.txt"))
    else:
        txt_files = []

    if not txt_files:
        print(f"[!] Không tìm thấy file .txt tại: {input_path}")
        return

    for idx, tf in enumerate(txt_files, 1):
        print(f"[{idx}/{len(txt_files)}] {tf.name}")
        content = tf.read_text(encoding="utf-8")
        raw = runner.run_on_text(content, max_new_tokens=max_new_tokens)
        parsed = parse_json_from_response(raw)
        out_file = output_dir / f"{tf.stem}_result.json"
        out_file.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[✓] Hoàn tất {len(txt_files)} file -> {output_dir}")


run(
    input_path=r"E:\SGOD\results\data_qwen\2026_557_383_2026_NĐ-CP_context.txt",
    lora_path=r"E:\SGOD\vietnamese-document-ai\module\IE\checkpoint\checkpoint-45",
    output_dir=r"E:\SGOD\vietnamese-document-ai\module\IE\outputs\Lora_Ck45",
)