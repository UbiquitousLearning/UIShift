# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from datasets import Dataset
from transformers import Qwen2VLForConditionalGeneration

from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from transformers.utils import logging

logger = logging.get_logger(__name__)

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GUI Automation GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    val_split_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of validation split, default 0.1"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["action_reward", "format_reward"],
        metadata={"help": "List of reward functions. Possible values: 'action_reward', 'format_reward'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

def format_reward(completions, **kwargs):
    """Reward function that only checks if the completion has the basic <answer></answer> format."""
    pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "gui_grounding_format_log.txt")
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]

def is_point_in_bbox(point_x, point_y, bbox):
    """Check if a point is inside a bounding box."""
    x1, y1, x2, y2 = bbox
    return x1 <= point_x <= x2 and y1 <= point_y <= y2

def action_reward(completions, solution, **kwargs):
    """Reward function that checks if the action parameters match the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        answer = ""
        action_type = ""
        # Try to parse the solution
        try:        
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            answer = content_match.group(1).strip() if content_match else content.strip()
            
            try: 
                gt_action = json.loads(sol)
            except Exception as e:
                rewards.append(0.0)
                continue

            # Try to parse the model's output
            try:
                # Extract JSON from the completion if it's wrapped in backticks
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', answer, re.DOTALL)
                if json_match:
                    answer = json_match.group(1)
                
                answer = answer.replace('\\n', '').strip()
                pred_action = json.loads(answer)
                if not isinstance(pred_action, dict):
                    if os.getenv("DEBUG_MODE") == "true":
                        with open("pred_action_debug.txt", "a", encoding="utf-8") as f:
                            f.write(f"Error Parse RAW pred_action content: {answer}\n")
                    rewards.append(0.0)
                    continue
            except Exception as e:
                rewards.append(0.0)
                continue
                
            # Get the action type
            if "action_type" not in pred_action:
                rewards.append(0.0)
                continue
                
            # Check if action types match
            if pred_action["action_type"].lower() != gt_action["action_type"].lower():
                rewards.append(0.0)
                continue
                
            action_type = pred_action["action_type"]
            if action_type == "click":
                if "x" not in pred_action or "y" not in pred_action:
                    rewards.append(0.0)
                    continue
                    
                pred_x = float(pred_action["x"])
                pred_y = float(pred_action["y"])
                
                # For click, check if x and y are within the bounding box
                if "bbox" in gt_action:
                    # If a bounding box is provided, check if the click point is inside
                    bbox = [float(x) for x in gt_action["bbox"]]
                    if is_point_in_bbox(pred_x, pred_y, bbox):
                        reward = 1.0
                else:
                    # If no bounding box, use exact coordinates
                    gt_x = float(gt_action["x"])
                    gt_y = float(gt_action["y"])
                    if pred_x == gt_x and pred_y == gt_y:
                        reward = 1.0
            
            elif action_type == "scroll":
                # For scroll, check if direction matches
                if "direction" in pred_action and pred_action["direction"].strip().lower() == gt_action["direction"].strip().lower():
                    reward = 1.0
            
            elif action_type == "open_app":
                # For open_app, check if app_name matches
                if "app_name" in pred_action and pred_action["app_name"].strip().lower() == gt_action["app_name"].strip().lower():
                    reward = 1.0
            
            elif action_type == "navigate_back":
                # For navigate_back, just check if action type matches (already done)
                reward = 1.0
            
            elif action_type == "input_text":
                # For input_text, check if text matches
                if "text" in pred_action and pred_action["text"].strip().lower() == gt_action["text"].strip().lower():
                    reward = 1.0
                    
        except Exception as e:
            # Catch any exceptions during verification and assign 0.0 reward
            reward = 0.0
            
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "gui_automation_action_log.txt")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
        
    return rewards

def get_vlm_module(model_name_or_path):
    """Get the appropriate VLM module based on the model name."""
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InternVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type="uishift_no_reasoning")
    print("question_prompt:", question_prompt)
    # Set up reward functions
    reward_funcs_registry = {
        "action_reward": action_reward,
        "format_reward": format_reward,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL dataset
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'image' in item:
                    if isinstance(item['image'], str):
                        # Store image path instead of loading the image
                        item['image_path'] = [os.path.join(image_folder, item['image'])]
                        del item['image']  # remove the image column so that it can be loaded later
                    elif isinstance(item['image'], list):
                        # if the image is a list, then it is a list of images (for multi-image input)
                        item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                        del item['image']  # remove the image column so that it can be loaded later
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
                # Remove immediate image loading
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                
                # Handle action data
                action_data = item['conversations'][1]['value']
                if isinstance(action_data, str):
                    # If it's a string, check if it's JSON
                    try:
                        action_json = json.loads(action_data)
                        item['solution'] = json.dumps(action_json)
                    except:
                        item['solution'] = action_data.replace('<answer>', '').replace('</answer>', '').strip()
                else:
                    # If it's already a dict or list, convert to string
                    item['solution'] = json.dumps(action_data)
                
                del item['conversations']
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        """Format example as conversation."""
        return {
            'image_path': [p for p in example['image_path']],
            'problem': example['problem'],
            'solution': example['solution'], 
            'prompt': [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                    {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                ]
            }]
        }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    # Split dataset for validation
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Initialize the GRPO trainer
    trainer = VLMGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)