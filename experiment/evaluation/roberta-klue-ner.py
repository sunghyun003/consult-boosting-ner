import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from module_ner.config import env

log_path = os.path.join(env.PROJECT_ROOT_DIR, "models/roberta-klue-ner/checkpoint-13130/trainer_state.json")

if os.path.exists(log_path):
    with open(log_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["log_history"])
    fig, ax1 = plt.subplots(figsize=(12, 6))

    if 'loss' in df.columns:
        train_loss = df.dropna(subset=['loss'])
        ax1.plot(train_loss['epoch'], train_loss['loss'], label='Train Loss', color='blue', alpha=0.4, linestyle='--')

    eval_df = df.dropna(subset=['eval_loss'])

    ax1.plot(eval_df['epoch'], eval_df['eval_loss'], label='Eval Loss', color='red', marker='o', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(eval_df['epoch'], eval_df['eval_f1'], label='Eval F1', color='green', marker='s', linewidth=2)
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1.0)  # F1 스코어 범위를 0~1로 고정하여 가독성 향상
    ax2.legend(loc='upper right')

    plt.title("NER Training Progress (Success!)")
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.show()
