# Minesweeper LLM Competition - GRPO Training

## 🎯 Goal
Finetune an LLM with LoRA using GRPO (Group Relative Policy Optimization) to play Minesweeper optimally. The model learns to:
- **Input**: JSON game state (board configuration)
- **Output**: JSON action (reveal or flag a cell)

## 📋 Table of Contents
- [Project Structure](#project-structure)
- [Competition Overview](#competition-overview)
- [Training Approach](#training-approach)
- [Setup](#setup)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Model Submission](#model-submission)
- [Scoring System](#scoring-system)
- [Optimization Tips](#optimization-tips)

## 📁 Project Structure

```
workspace/
├── minesweeper_grpo.ipynb      # Main training notebook with GRPO implementation
├── demo_eval.py                # Evaluation script (simulates competition environment)
├── minesweeper_config.yaml     # Generation configuration for inference
├── agents/
│   ├── minesweeper_agent.py    # Agent interface for playing Minesweeper
│   ├── minesweeper_model.py    # Model loader (customize for your trained model)
│   └── __init__.py
└── README.md                   # This file
```

## 🏆 Competition Overview

### Objective
Train the best Minesweeper-playing LLM that maximizes win rate and score across diverse board configurations.

### Rules
- **Model**: Use models from `/root/.cache/huggingface/hub` directory only
- **Method**: GRPO, SFT, or any RL policy (not restricted to GRPO)
- **Framework**: Unsloth (2-6x faster, 70% less VRAM)
- **Hardware**: AMD GPU (ROCm)
- **Disqualification**: Using models from outside the allowed directory

### Input/Output Format

**Input (Game State)**:
```json
{
  "board": [
    ["1", ".", ".", "."],
    [".", ".", ".", "."],
    [".", ".", "2", "."],
    [".", ".", ".", "."]
  ],
  "rows": 4,
  "cols": 4,
  "mines": 3,
  "flags_placed": 0,
  "cells_revealed": 2
}
```

**Output (Action)**:
```json
{"type": "reveal", "row": 2, "col": 3}
```
or
```json
{"type": "flag", "row": 1, "col": 4}
```

## 🧠 Training Approach

### Model Configuration
- **Base Model**: Llama-3.1-8B-Instruct (or GPT-OSS 20B)
- **Method**: GRPO with LoRA adapters
- **LoRA Rank**: 16 (adjustable based on task complexity)
- **Max Sequence Length**: 6000 tokens
- **Precision**: bfloat16

### LoRA Target Modules
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Curriculum Learning Strategy

**Phase 1: Small Boards (6x6, 8x8)**
- 100 training steps
- Focus on learning basic Minesweeper rules
- Simpler reasoning required

**Phase 2: Large Boards (10x10, 15x15, 20x20, 25x25)**
- 300 training steps
- Advanced strategic planning
- Longer max completion length (512 tokens)

## 🛠️ Setup

### Prerequisites
```bash
# Install required packages
pip install unsloth
pip install transformers
pip install datasets
pip install trl
pip install pyyaml
pip install torch
```

### Environment Configuration
```python
import os
os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface"
```

## 🚀 Training Pipeline

### 1. Load Model with Unsloth
```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/.cache/huggingface/models--Unsloth--Llama-3.1-8B-Instruct/...",
    load_in_4bit = False,
    max_seq_length = 6000,
    torch_dtype = torch.bfloat16,
)
```

### 2. Add LoRA Adapters
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

### 3. Define Reward Functions

**Valid JSON Reward**: Ensures model outputs proper JSON format
```python
def valid_json_reward(completions, **kwargs):
    # +0.2 for valid JSON, -0.5 for invalid
```

**Gameplay Scores**: Rewards based on game outcome
```python
def gameplay_scores(completions, **kwargs):
    # +5.0 for win
    # -0.5 for hitting mine
    # +0.5 for valid reveal
    # +0.25 for valid flag
    # Penalties for invalid moves
```

### 4. Generate Training Dataset

Creates diverse game states with varying board sizes:
```python
ds1 = generate_game_states(num_samples=1000, rows=6, cols=6, num_mines=5)
ds2 = generate_game_states(num_samples=1000, rows=10, cols=10, num_mines=20)
ds3 = generate_game_states(num_samples=1000, rows=15, cols=15, num_mines=45)
ds4 = generate_game_states(num_samples=1000, rows=20, cols=20, num_mines=80)
ds5 = generate_game_states(num_samples=1000, rows=25, cols=25, num_mines=125)
```

### 5. Configure GRPO Trainer

**Phase 1 Configuration**:
```python
training_args_p1 = GRPOConfig(
    output_dir = "minesweeper_curriculum_p1_llama",
    temperature = 1.0,
    learning_rate = 1e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    num_generations = 4,
    max_steps = 100,
    save_steps = 50,
)
```

### 6. Train the Model
```python
trainer_p1 = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [valid_json_reward, gameplay_scores, log_completions],
    args = training_args_p1,
    train_dataset = dataset_p1,
    callbacks = [MinesweeperEvalCallback(eval_every_steps=20, num_games=3)],
)
trainer_p1.train()
```

## 📊 Evaluation

### Using Demo Eval Script
```bash
# Test with built-in sample game state
python demo_eval.py

# Test with custom game state
python demo_eval.py --game_state_file my_state.json
```

### Programmatic Evaluation
```python
def play_full_game(model, tokenizer, rows=6, cols=6, num_mines=5, seed=None, max_moves=50):
    """Play a complete Minesweeper game with the model"""
    game = MinesweeperGame(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
    # ... game loop ...
    return game, moves
```

### Evaluation Callback
The training includes automatic evaluation every N steps:
```python
class MinesweeperEvalCallback(TrainerCallback):
    def __init__(self, eval_every_steps=50, num_games=5):
        # Plays sample games during training
        # Reports win rate
```

## 💾 Model Submission

### Save LoRA Adapters
```python
model.save_pretrained("my_minesweeper_model")
tokenizer.save_pretrained("my_minesweeper_model")
```

### Save Merged Model (Optional)
```python
model.save_pretrained_merged(
    "my_minesweeper_model_merged",
    tokenizer,
    save_method = "merged_16bit"
)
```

### Update Model Path
Edit [agents/minesweeper_model.py](agents/minesweeper_model.py):
```python
model_name = "/workspace/minesweeper_curriculum_p2_llama/checkpoint-150"
```

## 🎯 Scoring System

### Positive Rewards
- **Win Game**: +100 points (flag all mines + reveal all safe cells)
- **Flag Mine**: +15 points (correctly identified mine)
- **Reveal Safe Cell (Logical)**: +15 points (deduced from clues)
- **Reveal Safe Cell (Guess)**: +10 points (lucky guess)

### Penalties
- **Reveal Mine**: -25 points (Game Over)
- **Flag Incorrectly**: -10 points (flagged a safe cell)
- **Redundant Move**: -12/-8 points (reveal/flag same cell)
- **Invalid Move**: -15/-10 points (out of bounds/malformed JSON)
- **Excessive Flags**: -10 points (more flags than remaining mines)

### Strategy
1. **Prioritize Safety**: Revealing a mine ends the game with heavy penalty
2. **Maximize Logical Deductions**: +15 vs +10 for guessing
3. **Avoid Redundancy**: Track previous moves
4. **Valid JSON**: Essential for action execution

