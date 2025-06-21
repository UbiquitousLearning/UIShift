import pandas as pd
import json
import os

# UI 检索
def parse_bounds(bounds):
    left = bounds.get("left", 0)
    top = bounds.get("top", 0)
    right = bounds.get("right", 0)
    bottom = bounds.get("bottom", 0)
    return left, top, right, bottom

def retrieve_related_UI(raw_json_file, x_coord_ref, y_coord_ref):
    try:
        with open(raw_json_file) as f:
            data = json.load(f)
        matching_nodes = []
        for window in data.get("windows", []):
            if "tree" in window and "nodes" in window["tree"]:
                for node in window["tree"]["nodes"]:
                    if "boundsInScreen" in node:
                        left, top, right, bottom = parse_bounds(node["boundsInScreen"])
                        if left <= x_coord_ref <= right and top <= y_coord_ref <= bottom:
                            matching_nodes.append(node)
        if not matching_nodes:
            return None

        # 找最小包含区域
        smallest_node = None
        smallest_area = float('inf')
        for node in matching_nodes:
            left, top, right, bottom = parse_bounds(node["boundsInScreen"])
            area = abs(right - left) * abs(bottom - top)
            if area < smallest_area:
                smallest_area = area
                smallest_node = node

        if smallest_node:
            left, top, right, bottom = parse_bounds(smallest_node["boundsInScreen"])
            return [left, top, right, bottom]
        return None

    except Exception as e:
        print(f"Error in {raw_json_file}: {e}")
        return None

# 主要处理
def process_csv_and_add_bbox(csv_path, output_path, dataset_dir):
    df = pd.read_csv(csv_path)
    bbox_list = []

    for i, row in df.iterrows():
        episode_id = row["episode_id"]
        index = row["index"]

        raw_json_file = os.path.join(
            dataset_dir, f"episode_{episode_id}", f"epi_{episode_id}_tree_{index}.json"
        )

        try:
            action = json.loads(row["action_groundtruth"])
            x, y = action.get("x", None), action.get("y", None)
            if x is not None and y is not None:
                bbox = retrieve_related_UI(raw_json_file, x, y)
                bbox_list.append(str(bbox) if bbox else "")
            else:
                bbox_list.append("")
        except json.JSONDecodeError:
            print(f"Invalid JSON at row {i}")
            bbox_list.append("")

    df["matched_bbox"] = bbox_list
    df.to_csv(output_path, index=False)
    print(f"处理完成，结果保存在：{output_path}")
    
# 统计分析
def analyze_action_types(output_path):
    df = pd.read_csv(output_path)
    print("\n 动作类型统计")
    total = len(df)
    print(f"总共数据条数: {total}")

    type_counts = {}
    for action_str in df["action_groundtruth"]:
        try:
            action = json.loads(action_str)
            action_type = action.get("action_type", "unknown")
        except json.JSONDecodeError:
            action_type = "invalid_json"
        type_counts[action_type] = type_counts.get(action_type, 0) + 1

    for action_type, count in type_counts.items():
        print(f"动作类型 {action_type}: {count} 条")

def analyze_matched_bbox(output_path):
    df = pd.read_csv(output_path)
    print("\n bbox 匹配结果")
    matched = df["matched_bbox"].apply(lambda x: isinstance(x, str) and x.strip().startswith("["))
    matched_count = matched.sum()
    print(f"成功匹配 bbox 的条数: {matched_count} / {len(df)}")

    for i, row in df.iterrows():
        try:
            action = json.loads(row["action_groundtruth"])
            action_type = action.get("action_type", "unknown")
        except json.JSONDecodeError:
            continue

        bbox = row["matched_bbox"]
        has_bbox = isinstance(bbox, str) and bbox.strip().startswith("[")

        if action_type == "click" and not has_bbox:
            print(f"[未匹配 click] episode {row['episode_id']}, index {row['index']}")
        elif action_type != "click" and has_bbox:
            print(f"[异常匹配非click] episode {row['episode_id']}, index {row['index']}, action_type={action_type}, bbox={bbox}")

def main():
    dataset_dir = "/path/to/androidcontrol_dataset"
    metadata_csv_path = "/path/to/androidcontrol_matadata.csv"
    output_path = "/path/to/androidcontrol_metadata_with_bbox.csv"

    process_csv_and_add_bbox(csv_path, output_path, dataset_dir)
    analyze_action_types(output_path)
    analyze_matched_bbox(output_path)

if __name__ == "__main__":
    main()
