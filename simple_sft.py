#!/usr/bin/env python3

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import json
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os

class NexarSFTDataset(Dataset):
    def __init__(self, json_path, media_path, processor, max_length=4096):
        self.processor = processor
        self.max_length = max_length
        self.media_path = media_path
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load video with OpenCV
        video_path = os.path.join(self.media_path, item['videos'][0])
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / 4)
        
        frames = []
        count = 0
        while len(frames) < 40:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            count += 1
        cap.release()
        
        # Build conversation
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "video", "video": frames, "fps": 4, "max_pixels": 768*768},
                {"type": "text", "text": item['conversations'][0]['value']}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": item['conversations'][1]['value']}]}
        ]
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                inputs[key] = inputs[key].squeeze(0)
        
        # Labels are the same as input_ids for causal LM
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs

def main():
    # Load model
    model_name = "nvidia/Cosmos-Reason2-2B"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.train()
    
    # Dataset
    dataset = NexarSFTDataset(
        "/teamspace/studios/this_studio/nexar_data/sft/train.json",
        "./nexar_data/sample_videos",
        processor
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/cosmos_reason2/sft_output",
        num_train_epochs=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        learning_rate=1e-4,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model("/tmp/cosmos_reason2/sft_output/final")

if __name__ == "__main__":
    main()
