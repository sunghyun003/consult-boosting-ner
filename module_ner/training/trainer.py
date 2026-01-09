import numpy as np

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from module_ner.config import env

model_id = 'klue/roberta-base'
DATA_DIR = f'{env.PROJECT_ROOT_DIR}/data/external/klue/ner'
BASE_MODEL_DIR = f'{env.PROJECT_ROOT_DIR}/models/roberta-large'
OUTPUT_MODEL_DIR = f'{env.PROJECT_ROOT_DIR}/models/roberta-large-klue-ner'


def train_ner():
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    # Data Preparation
    dataset = load_dataset("parquet", data_files={
            'train': f"{DATA_DIR}/klue_ner_train_roberta_aligned.parquet",
            'validation': f"{DATA_DIR}/klue_ner_val_roberta_aligned.parquet",
        })

    label_list = dataset['train'].features['labels'].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    print(f"* Label List: {label_list}")
    print(f"* id2label: {id2label}")
    print(f"* label2id: {label2id}")

    # Model & Configuration
    config = AutoConfig.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        # Dropout
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    # * hidden_dropout_prob=0.1,
    #   - 1. Self-Attention의 결과가 나오고 다음 층으로 전달되기 직전 (Residual Connection 전)
    #   - 2. FFN(Feed-Forward Network)의 두 번째 Linear 레이어 통과 직후
    #   - 3. 최종 NER 분류기(Classifier Head) 직전
    # * attention_probs_dropout_prob=0.1,
    #   - 어텐션 맵(Attention Map), SoftMax 함수 처리 이후 10%

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_DIR,
        config=config,
    )

    # Training Argument
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        learning_rate=2e-5,
        # Regularization - Micro batch
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # Regularization - L2
        weight_decay=0.01,
        load_best_model_at_end=True,
        num_train_epochs=10,
        metric_for_best_model="f1",
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        report_to="tensorboard",
    )

    # Trainer API
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 3회 이상 성능 안 오르면 종료
    )

    # Training
    trainer.train()

    #