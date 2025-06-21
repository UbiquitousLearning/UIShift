import csv
import json
import os
import random

def convert_csv_to_jsonl(
    csv_path: str,
    output_jsonl_path: str
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    jsonl_data = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            try:
                index = int(row['index'])
                image_file_1 = f"epi_{row['episode_id']}_ss_{index}.jpg"
                image_file_2 = f"epi_{row['episode_id']}_ss_{index + 1}.jpg"
                width = row['screenshot_width']
                height = row['screenshot_height']
                action_groundtruth = json.loads(row['action_groundtruth'])

                if action_groundtruth.get("action_type") == "click":
                    raw_bbox = json.loads(row['matched_bbox'])
                    assert len(raw_bbox) == 4
                    action_groundtruth["bbox"] = raw_bbox

                # prompt used in UIShift with k=3
                # prompt = (
                #     "<image>\n"
                #     "You are given two GUI screenshots:\n"
                #     "- Screenshot 1: the screen *before* any actions.\n"
                #     "- Screenshot 3: the screen *after* two consecutive actions.\n\n"
                #     "These two screenshots correspond to the first and third steps in a GUI interaction sequence.\n"
                #     "Your task is to infer the first GUI action that transformed Screenshot 1 into an intermediate Screenshot 2.\n\n"
                #     "Return exactly one JSON-formatted action inside <answer>{...}</answer>.\n\n"
                #     "Available action types:\n"
                #     "- Click: {\"action_type\": \"click\", \"x\": <coordinate_x>, \"y\": <coordinate_y>}\n"
                #     "- Input text: {\"action_type\": \"input_text\", \"text\": <text>}\n"
                #     "- Navigate back: {\"action_type\": \"navigate_back\"}\n"
                #     "- Open app: {\"action_type\": \"open_app\", \"app_name\": <app_name>}\n"
                #     "- Scroll: {\"action_type\": \"scroll\", \"direction\": <up/down/left/right>} \n\n"
                #     f"Screenshot width: {width}\n"
                #     f"Screenshot height: {height}\n"
                #     "# Now, based on the transition from Screenshot 1 to Screenshot 3, infer the first action that was executed.\n"
                # )
                
                # prompt used in UIShift with k=1
                prompt = (
                    "<image>\n"
                    "You are given two GUI screenshots:\n"
                    "- Screenshot 1: the screen *before* the action.\n"
                    "- Screenshot 2: the screen *after* the action.\n\n"
                    "Your task is to infer the single GUI action that caused this transition.\n\n"
                    "Return exactly one JSON-formatted action inside <answer>{...}</answer>.\n\n"
                    "Available action types:\n"
                    "- Click: {\"action_type\": \"click\", \"x\": <coordinate_x>, \"y\": <coordinate_y>}\n"
                    "- Input text: {\"action_type\": \"input_text\", \"text\": <text>}\n"
                    "- Navigate back: {\"action_type\": \"navigate_back\"}\n"
                    "- Open app: {\"action_type\": \"open_app\", \"app_name\": <app_name>}\n"
                    "- Scroll: {\"action_type\": \"scroll\", \"direction\": <up/down/left/right>} \n\n"
                    f"Screenshot width: {width}\n"
                    f"Screenshot height: {height}\n"
                    "# Now, describe the action that led from Screenshot 1 to Screenshot 2.\n"
                )
                json_obj = {
                    "task": "ui transition prediction",
                    "image": [image_file_1, image_file_2],
                    "conversations": [
                        {"role": "user", "value": prompt},
                        {"role": "assistant", "value": action_groundtruth}
                    ]
                }

                jsonl_data.append(json_obj)

            except Exception as e:
                if verbose:
                    print(f"[Row {idx}] Skipped due to error: {e}")
                continue

    random.shuffle(jsonl_data)

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for obj in jsonl_data:
            jsonl_file.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Output saved to: {output_jsonl_path}")
    print(f"Total valid entries: {len(jsonl_data)}")


if __name__ == "__main__":
    metadata_csv_path = f"/path/to/androidcontrol_metadata_with_bbox.csv"
    output_jsonl_path = f"/path/to/uishift_trainingset.jsonl"
    convert_csv_to_jsonl_for_ui_transition(metadata_csv_path, output_jsonl_path)
