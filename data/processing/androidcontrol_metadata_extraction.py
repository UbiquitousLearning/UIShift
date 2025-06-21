import json
import os
import csv
import random
import pandas as pd

def metadata_extraction(test_dir, output_csv):
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ['episode_id', 'index', 'task instruction', 'step instruction','screenshot_width', 'screenshot_height', 'action_groundtruth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for subdir, _, files in os.walk(test_dir):
            if subdir == test_dir:
                continue
            episode_id = os.path.basename(subdir).replace("episode_", "")
            print(subdir)
            metadata_path = os.path.join(subdir, f"metadata_epi_{episode_id}.json")
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            task_instruction = metadata["goal"]
            screenshot_widths = metadata["screenshot_widths"]
            screenshot_heights = metadata["screenshot_heights"]
            step_instruction = metadata["step_instructions"]
            actions = metadata["actions"]
            
            for file in files:
                if file.endswith("_simplified.json"):
                    simplified_json_path = os.path.join(subdir, file)
                    with open(simplified_json_path, "r") as f:
                        simplified_data = json.load(f)
                    
                    ui_elements = simplified_data.get("UI elements", [])
                    if 0 < len(ui_elements) < 50:
                        index = int(file.split("_")[3])
                        writer.writerow({
                            'episode_id': episode_id,
                            'index': index,
                            'task instruction': task_instruction,
                            'step instruction': step_instruction[index] if index!=len(screenshot_widths)-1 else "final state",
                            'screenshot_width': screenshot_widths[index],
                            'screenshot_height': screenshot_heights[index],
                            'action_groundtruth': json.dumps(actions[index]) if index<len(screenshot_widths)-1 else "task end"
                        })

def filter_action_types(csv_path):
    df = pd.read_csv(csv_path)
    original_len = len(df)

    # 去除 "final state"
    df = df[df["step instruction"] != "final state"]

    # 去除 action_type == "wait" or "long_press" or "navigate_home"
    def is_filtered_action(action_str):
        try:
            action = json.loads(action_str)
            is_wait = action.get("action_type") == "wait"
            is_long_press = action.get("action_type") == "long_press"
            is_navigate_home = action.get("action_type") == "navigate_home"
            is_filtered = is_wait or is_long_press or is_navigate_home
            return is_filtered 
        except:
            return False

    df = df[~df["action_groundtruth"].apply(is_filtered_action)]

    df.to_csv(csv_path, index=False)
    print(f"Filtered {original_len - len(df)} rows with 'final state' or 'wait' or 'long_press' action. Remaining: {len(df)}")


if __name__ == "__main__":
    dataset_dir = "/path/to/androidcontrol_dataset"
    output_matadata_csv = "../path/to/androidcontrol_metadata.csv"
    
    metadata_extraction(dataset_dir, output_matadata_csv)
    # filter_action_types(output_matadata_csv)