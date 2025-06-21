import os
import tensorflow as tf
import cv2
import json
import numpy as np
import re
from google.protobuf.json_format import MessageToDict
from android_env.proto.a11y import android_accessibility_forest_pb2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

input_dir = "/path/to/androidcontrol_tfrecord_folder"
output_dir = "/path/to/androidcontrol_parsed_folder"
os.makedirs(output_dir, exist_ok=True)

record_pattern = re.compile(r"android_control-(\d{5})-of-(\d{5})")
tfrecord_files = sorted([f for f in os.listdir(input_dir) if f.startswith("android_control")])

total_episodes = 0
total_screenshots = 0
saved_screenshots = 0
saved_trees = 0

for tf_file in tfrecord_files:
    record_match = record_pattern.search(tf_file)
    if not record_match:
        print(f"Skipping unrecognized file: {tf_file}")
        continue
    
    record_id = record_match.group(1)
    tf_file_path = os.path.join(input_dir, tf_file)
    print(f"Processing {tf_file} (Record ID: {record_id})...")
    
    raw_dataset = tf.data.TFRecordDataset(tf_file_path, compression_type="GZIP")
    for episode_index, raw_record in enumerate(raw_dataset):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        episode_id = example.features.feature["episode_id"].int64_list.value[0]
        goal = example.features.feature["goal"].bytes_list.value[0].decode()
        screenshot_widths = list(example.features.feature["screenshot_widths"].int64_list.value)
        screenshot_heights = list(example.features.feature["screenshot_heights"].int64_list.value)
        
        total_episodes += 1
        total_screenshots += len(screenshot_widths)
        
        episode_dir = os.path.join(output_dir, f"episode_{episode_id}")
        os.makedirs(episode_dir, exist_ok=True)
        screenshots = example.features.feature["screenshots"].bytes_list.value
        
        for i, screenshot_bytes in enumerate(screenshots):
            try:
                img_array = np.frombuffer(screenshot_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Screenshot {i} in {episode_id} is corrupted.")
                    continue
                
                screenshot_path = os.path.join(episode_dir, f"epi_{episode_id}_ss_{i}.jpg")
                cv2.imwrite(screenshot_path, img)
                saved_screenshots += 1
            except Exception as e:
                print(f"Error saving screenshot {i}: {str(e)}")
        
        accessibility_trees = android_accessibility_forest_pb2.AndroidAccessibilityForest()
        for i in range(0, len(screenshots)):
            try:
                accessibility_trees.ParseFromString(example.features.feature["accessibility_trees"].bytes_list.value[i])
                tree_path = os.path.join(episode_dir, f"epi_{episode_id}_tree_{i}.json")
                with open(tree_path, "w") as f:
                   json.dump(MessageToDict(accessibility_trees), f, indent=4)
                saved_trees += 1
            except Exception as e:
                print(f"Error saving screenshot {i}: {str(e)}")
        
        actions = []
        try:
            actions = [json.loads(action.decode()) for action in example.features.feature["actions"].bytes_list.value]
        except Exception as e:
            print(f"Error extracting actions: {str(e)}")
        
        step_instructions = []
        try:
            step_instructions = [instr.decode() for instr in example.features.feature["step_instructions"].bytes_list.value]
        except Exception as e:
            print(f"Error extracting step_instructions: {str(e)}")
            
        episode_metadata = {
            "record_id": record_id,
            "episode_id": episode_id,
            "goal": goal,
            "screenshot_widths": screenshot_widths,
            "screenshot_heights": screenshot_heights,
            "step_instructions": step_instructions,
            "actions": actions
        }

        metadata_path = os.path.join(episode_dir, f"metadata_epi_{episode_id}.json")
        try:
            with open(metadata_path, "w") as f:
                json.dump(episode_metadata, f, indent=4)
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")

print("Finish!")
print(f"Processed {total_episodes} episodes")
print(f"Extract {saved_screenshots}/{total_screenshots} screenshots")
print(f"Extract {saved_trees}/{total_screenshots} trees")