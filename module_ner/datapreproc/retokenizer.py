import os

from transformers import AutoTokenizer
from datasets import (
    load_dataset,
    Features,
    Sequence,
    ClassLabel,
    Value,
)

from module_ner.config import env

model_id = 'klue/roberta-base'
MODEL_DIR = f'{env.PROJECT_ROOT_DIR}/models/roberta-base'
DATA_DIR = f'{env.PROJECT_ROOT_DIR}/data/external/klue/ner'


def align_labels_to_inference_tokenization(examples, tokenizer):
    sentences = ["".join(t) for t in examples["tokens"]]

    tokenized_inputs = tokenizer(
        sentences,
        truncation=True,
        return_offsets_mapping=True,
    )
    # padding='longest' -> DataCollator

    labels = []
    new_tokens_list = []
    for i, label in enumerate(examples["ner_tags"]):
        # for labels (new ner_tags)
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        offsets = tokenized_inputs["offset_mapping"][i]

        label_ids = []
        for word_idx, (start, end) in zip(word_ids, offsets):
            if word_idx is None or start == end:  # [CLS], [SEP], 특수 기호 등 특수 토큰 (빈 토큰)
                label_ids.append(-100)            # e.g., `None 0 0`
            else:
                # * 해당 서브워드가 시작되는 원본 음절의 라벨을 가져옴
                # - '영동'이라는 토큰이 원본 문장의 3번 인덱스에서 시작한다면, label[3]을 부여
                label_ids.append(label[start])
        labels.append(label_ids)

        # for tokens
        input_ids = tokenized_inputs["input_ids"][i]
        new_tokens_list.append(tokenizer.convert_ids_to_tokens(input_ids))

    tokenized_inputs["labels"] = labels
    tokenized_inputs["tokens"] = new_tokens_list  # [추가] 문자열 토큰 컬럼
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    dataset = load_dataset("parquet", data_files={
        'train': f"{DATA_DIR}/train-*.parquet",
        'validation': f"{DATA_DIR}/validation-*.parquet",
    })

    label_list = dataset['train'].features['ner_tags'].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    print(f"* Label List: {label_list}")
    print(f"* id2label: {id2label}")
    print(f"* label2id: {label2id}")

    processed_dataset = dataset.map(
        lambda x: align_labels_to_inference_tokenization(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    new_features = Features({
        "tokens": Sequence(Value("string")),
        "input_ids": Sequence(Value("int32")),
        "token_type_ids": Sequence(Value("int8")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(ClassLabel(names=label_list)),
    })
    processed_dataset = processed_dataset.cast(new_features)

    train_save_path = os.path.join(DATA_DIR, "klue_ner_train_roberta_aligned.parquet")
    val_save_path = os.path.join(DATA_DIR, "klue_ner_val_roberta_aligned.parquet")

    os.makedirs(DATA_DIR, exist_ok=True)
    processed_dataset['train'].to_parquet(train_save_path)
    processed_dataset['validation'].to_parquet(val_save_path)

    if os.path.exists(train_save_path) and os.path.exists(val_save_path):
        print("Parquet has been created.")
    else:
        print("No parquet has been created.")


if __name__ == "__main__":
    main()
