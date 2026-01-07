# 1. Data Preprocessing
<img src="docs/img/data-preparation-1-1.png" />

* klue/klue의 Subset `ner` 활용
    - NER 데이터셋 특징
        1. 음절 단위 토큰화
            - tokens : `["특", "히", " ", "영", "동", "고", "속", "도", "로", ... ]`
        2. 태깅/라벨링 (BIO Format)
            ```
            {
              "0": "B-DT", # 날짜 (Date) - 시작
              "1": "I-DT", # 날짜 - 내부
              "2": "B-LC", # 지명 (Location) - 시작
              "3": "I-LC", # 지명 - 내부
              "4": "B-OG", # 조직 (Organization) - 시작
              "5": "I-OG", # 조직 - 내부
              "6": "B-PS", # 인명 (Person) - 시작
              "7": "I-PS", # 인명 - 내부
              "8": "B-QT", # 수량 (Quantity) - 시작
              "9": "I-QT", # 수량 - 내부
              "10": "B-TI", # 시간 (Time) - 시작
              "11": "I-TI", # 시간 - 내부
              "12": "O" # 개체명 아님 (Outside)
            }
            ```
            - https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
    - Example
        <img src="docs/img/data-preparation-1-5.png" />


* 문제점
    - 사용 베이스 모델 `klue/roberta-base` (한국어 특화 & 가벼운 모델)
        + 샘플 데이터 셋의 token과 tag가 음절 단위로 구성되어 있음.
        + 하지만 사용하려는 base 모델의 토큰나이즈 방식은 형태소/서브워드 방식(WordPiece)임.
            * tokens : `["특히", "영동", "##고속도로", ...]` (공백 토큰화 (X))
        + 우리가 학습/추론할 때, 베이스 모델은 음절단위로 토큰화하지 않기 때문에, 결국 해당 데이터과 동일한 룰셋으로 학습/추론 상황을 만들 수 없음.
        + 해당 샘플을 그대로 활용할 수는 없어, 해당 데이터셋을 기준으로 WordPiece 방식 재토큰화 및 재태깅을 진행함.
          <img src="docs/img/data-preparation-1-7.png" />

<br>

# 2. Model Training
베이스 모델 `klue/roberta-base`에는 인코더만 있음. 이에 따라, NER 용 헤드 추가학습 필요

## 2.1. Hyperparameter
```python
config = AutoConfig.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

...

training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
...
```

## 2.2. Evaluation
```
/Users/sunghyun03/git/consult-boosting-ner/.venv/lib/python3.13/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.0168, 'grad_norm': 0.15941525995731354, 'learning_rate': 1.1623000761614625e-05, 'epoch': 4.19}                                                                                                             
{'loss': 0.0161, 'grad_norm': 1.3545631170272827, 'learning_rate': 1.1242193450114243e-05, 'epoch': 4.38}                                                                                                              
{'loss': 0.0185, 'grad_norm': 0.13429899513721466, 'learning_rate': 1.0861386138613863e-05, 'epoch': 4.57}                                                                                                             
{'loss': 0.0163, 'grad_norm': 0.2579185366630554, 'learning_rate': 1.0480578827113481e-05, 'epoch': 4.76}                                                                                                              
{'loss': 0.0172, 'grad_norm': 0.8366368412971497, 'learning_rate': 1.0099771515613102e-05, 'epoch': 4.95}                                                                                                              
{'eval_loss': 0.11231765151023865, 'eval_precision': 0.8708879884623308, 'eval_recall': 0.8914586994727592, 'eval_f1': 0.8810532897936497, 'eval_accuracy': 0.9765978944623709, 'eval_runtime': 21.7204, 'eval_samples_per_second': 230.199, 'eval_steps_per_second': 28.775, 'epoch': 5.0}                                                                                                                                                   
{'train_runtime': 1843.3681, 'train_samples_per_second': 113.965, 'train_steps_per_second': 14.246, 'train_loss': 0.04501724390246354, 'epoch': 5.0}                                                                   
 50%|██████████████████████████████████████████████████████████████████████████████████████ 
```

<img src="docs/img/eval-1-1.png" />

| Epoch | Train Loss | Eval Loss | Eval Precision | Eval Recall | Eval F1 | Eval Accuracy | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1.0 | 0.0725 | 0.0812 | 0.8787 | 0.8885 | 0.8836 | 0.9768 | |
| **2.0** | 0.0485 | **0.0778** | **0.8854** | **0.9001** | **0.8927** | **0.9782** | **Best Model** |
| 3.0 | 0.0327 | 0.0905 | 0.8748 | 0.8904 | 0.8825 | 0.9774 | 과적합 시작 |
| 4.0 | 0.0257 | 0.0985 | 0.8846 | 0.8940 | 0.8893 | 0.9779 | |
| 5.0 | 0.0172 | 0.1123 | 0.8708 | 0.8914 | 0.8810 | 0.9765 | Early Stop |

* Best Model
    - 2번째 Epoch에서 Eval Loss가 가장 낮고, 대체적으로 모든 성능 지표들이 가장 높게 측정
* Overfitting
    - 3번째 Epoch부터 Train Loss는 계속 감소하지만, Eval Loss가 다시 상승하는 전형적인 과적합 현상 발생
* Early Stopping
    - 성능 개선이 이루어지지 않아 5번째 Epoch에서 학습 중단

<br>

# 3. Model Inference
## 3.1. APIs
* [POST] http://0.0.0.0:8000/infer
    - Request Body
        ```json
        {
            "text": "서울 마곡 LG유플러스 AX솔루션개발팀은 훌륭한 구성원들로 모여있는 팀이다."
        }
        ```
    - Response Body
        ```json
        {
            "input": "서울 마곡 LG유플러스 AX솔루션개발팀은 훌륭한 구성원들로 모여있는 팀이다.",
            "result": [
                {
                    "entity_group": "LC",
                    "score": 0.8558929562568665,
                    "word": "서울",
                    "start": 0,
                    "end": 2
                },
                {
                    "entity_group": "OG",
                    "score": 0.5763113498687744,
                    "word": "마곡",
                    "start": 3,
                    "end": 5
                },
                {
                    "entity_group": "OG",
                    "score": 0.9307456612586975,
                    "word": "LG유플러스 AX솔루션개발팀",
                    "start": 6,
                    "end": 21
                }
            ]
        }
        ```

<br>

# 4. Model Evaluation (TBD, 개선 예정)
## 4.1. Accuracy
* Confusion Matrix : Precision, Recall, F1-Score
```
from seqeval.metrics import classification_report

y_true = [["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
          ["B-PER", "I-PER", "O"]]
y_pred = [["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
          ["B-PER", "I-PER", "O"]]
print(classification_report(y_true, y_pred))
```
```
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
```

## 4.2. Generalization
* Train-Validation Split 후 태그 균형 체크
    - 5GPLN, MVAS, ORG 빈도 분포가 대체로 각 분할에서 동일한지 검토
    - 따라서 이 검증 세트와 테스트 세트는 NER 태그의 일반화 능력을 평가

```
from seqeval.metrics import classification_report

y_true = [["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
          ["B-PER", "I-PER", "O"]]
y_pred = [["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
          ["B-PER", "I-PER", "O"]]
print(classification_report(y_true, y_pred))
```

<br>

# 5. Future Works
1. 보안 클라우드에서 학습, 추론
2. 성능평가 개선

<br>

---

### [ References ]
* GitHub : https://github.com/sunghyun003/consult-boosting-ner
  
<br>