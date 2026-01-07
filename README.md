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
    - Case 1) 짧은 문장
        + Request Body
            ```json
            {
                "text": "서울 마곡 LG유플러스 AX솔루션개발팀은 훌륭한 구성원들로 모여있는 팀이다."
            }
            ```
        + Response Body
            ```json
            {
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
    - Case 2) 긴 문장 - 일반
        + Request Body
            ```json
            {
                "text": "지난 2025년 10월 24일 오전 9시 정각에 테슬라의 일론 머스크 대표는 미국 텍사스 기가팩토리에서 새로운 자율주행 트럭 500대를 공개하는 대규모 쇼케이스를 열었습니다. 이 행사에는 전 세계 200여 개의 언론사와 약 3,000명의 관계자들이 참석했으며, 현대자동차와 같은 글로벌 경쟁 기업의 분석팀원들도 현장 분위기를 살피기 위해 집결했습니다. 발표가 이어진 2시간 동안 테슬라의 주가는 15% 이상 급등했고, 현장에서 즉석 예약된 수량만 12,000건에 달하며 역대 최대 기록을 경신했습니다. 다가오는 2026년 1월에는 한국 서울 코엑스에서 아시아 시장을 겨냥한 후속 행사를 가질 예정이라며 마케팅 팀장인 김철수 이사가 공식 인터뷰를 통해 포부를 밝혔습니다. 마지막으로 행사를 축하하기 위해 삼성전자 측에서 지원한 스마트 워치 5,000세트가 참석자 전원에게 배포되었으며, 뜨거웠던 현장의 열기는 밤 11시가 넘어서야 겨우 식기 시작했습니다."
            }
            ```
            * "지난 2025년 10월 24일 오전 9시 정각에 테슬라의 일론 머스크 대표는 미국 텍사스 기가팩토리에서 새로운 자율주행 트럭 500대를 공개하는 대규모 쇼케이스를 열었습니다. 이 행사에는 전 세계 200여 개의 언론사와 약 3,000명의 관계자들이 참석했으며, 현대자동차와 같은 글로벌 경쟁 기업의 분석팀원들도 현장 분위기를 살피기 위해 집결했습니다. 발표가 이어진 2시간 동안 테슬라의 주가는 15% 이상 급등했고, 현장에서 즉석 예약된 수량만 12,000건에 달하며 역대 최대 기록을 경신했습니다. 다가오는 2026년 1월에는 한국 서울 코엑스에서 아시아 시장을 겨냥한 후속 행사를 가질 예정이라며 마케팅 팀장인 김철수 이사가 공식 인터뷰를 통해 포부를 밝혔습니다. 마지막으로 행사를 축하하기 위해 삼성전자 측에서 지원한 스마트 워치 5,000세트가 참석자 전원에게 배포되었으며, 뜨거웠던 현장의 열기는 밤 11시가 넘어서야 겨우 식기 시작했습니다."
        + Response Body
            ```json
            {
                "result": [
                    {
                        "entity_group": "DT",
                        "score": 0.8937010765075684,
                        "word": "지난 2025년 10월 24일",
                        "start": 0,
                        "end": 16
                    },
                    {
                        "entity_group": "TI",
                        "score": 0.9922220706939697,
                        "word": "오전 9시 정각",
                        "start": 17,
                        "end": 25
                    },
                    {
                        "entity_group": "OG",
                        "score": 0.9850203990936279,
                        "word": "테슬라",
                        "start": 27,
                        "end": 30
                    },
                    {
                        "entity_group": "PS",
                        "score": 0.996773898601532,
                        "word": "일론 머스크",
                        "start": 32,
                        "end": 38
                    },
                    {
                        "entity_group": "LC",
                        "score": 0.9824407696723938,
                        "word": "미국 텍사스 기가팩토리",
                        "start": 43,
                        "end": 55
                    },
                    {
                        "entity_group": "QT",
                        "score": 0.9958693385124207,
                        "word": "500대",
                        "start": 70,
                        "end": 74
                    },
                    {
                        "entity_group": "QT",
                        "score": 0.9868373870849609,
                        "word": "200여 개의",
                        "start": 110,
                        "end": 117
                    },
                    {
                        "entity_group": "QT",
                        "score": 0.9895029664039612,
                        "word": "3, 000명",
                        "start": 125,
                        "end": 131
                    },
                    {
                        "entity_group": "OG",
                        "score": 0.9898701310157776,
                        "word": "현대자동차",
                        "start": 146,
                        "end": 151
                    },
                    {
                        "entity_group": "TI",
                        "score": 0.9495131373405457,
                        "word": "2시간 동안",
                        "start": 205,
                        "end": 211
                    },
                    {
                        "entity_group": "OG",
                        "score": 0.9854955673217773,
                        "word": "테슬라",
                        "start": 212,
                        "end": 215
                    },
                    {
                        "entity_group": "QT",
                        "score": 0.9944543838500977,
                        "word": "15 %",
                        "start": 221,
                        "end": 224
                    },
                    {
                        "entity_group": "QT",
                        "score": 0.9967505931854248,
                        "word": "12, 000건",
                        "start": 250,
                        "end": 257
                    },
                    {
                        "entity_group": "DT",
                        "score": 0.9744736552238464,
                        "word": "2026년 1월",
                        "start": 286,
                        "end": 294
                    },
                    {
                        "entity_group": "LC",
                        "score": 0.9784884452819824,
                        "word": "한국 서울 코엑스",
                        "start": 297,
                        "end": 306
                    },
                    {
                        "entity_group": "LC",
                        "score": 0.9515602588653564,
                        "word": "아시아",
                        "start": 309,
                        "end": 312
                    },
                    {
                        "entity_group": "PS",
                        "score": 0.9980403184890747,
                        "word": "김철수",
                        "start": 345,
                        "end": 348
                    },
                    {
                        "entity_group": "OG",
                        "score": 0.9834198951721191,
                        "word": "삼성전자",
                        "start": 393,
                        "end": 397
                    },
                    {
                        "entity_group": "QT",
                        "score": 0.9870311617851257,
                        "word": "5, 000세트",
                        "start": 413,
                        "end": 420
                    },
                    {
                        "entity_group": "TI",
                        "score": 0.9922183156013489,
                        "word": "밤 11시",
                        "start": 452,
                        "end": 457
                    }
                ]
            }
            ```
    - Case 3) 긴 문장 - 통신 도메인
        + Request Body
            ```json
            {
                "text": "안녕하세요, 김아무개 고객님! LG유플러스 AICC 지능형 상담사입니다. 5G 프리미엄 요금제의 핵심 혜택인 유튜브 프리미엄 연동 문제로 연락 주셨군요. 즐거운 영상 시청 중에 흐름이 끊겨 많이 답답하셨을 텐데, 제가 빠르게 해결 도와드리겠습니다! 많이 번거로우셨죠. 고객님의 소중한 권리를 되찾아드리기 위해 현재 상태를 체크해 보니, [계정 미연동] 상태로 확인됩니다. 보통 이 문제는 세 가지 경우에 발생하는데요. 바로 해결 가능한 '치트키'를 안내해 드릴게요. 고객님, 지금 바로 고객님의 휴대폰으로 '원클릭 연동 링크'를 발송해 드렸습니다. 이 링크를 누르시면 복잡한 절차 없이 구글 계정 선택만으로 연동이 완료됩니다. 성공하셨다니 정말 다행입니다! 이제 광고 없는 쾌적한 영상 시청 즐기시길 바랍니다. 추가로 5G 프리미엄 요금제에 포함된 다른 미디어 혜택(VOD 쿠폰 등)도 아직 미사용 중이신데, 관련 안내 문자를 함께 보내드릴까요? 별씀을요! 김아무개 고객님, 오늘도 프리미엄한 하루 보내세요. 지금까지 AICC 상담사였습니다!"
            }
            ```
            * "안녕하세요, 김아무개 고객님! LG유플러스 AICC 지능형 상담사입니다. 5G 프리미엄 요금제의 핵심 혜택인 유튜브 프리미엄 연동 문제로 연락 주셨군요. 즐거운 영상 시청 중에 흐름이 끊겨 많이 답답하셨을 텐데, 제가 빠르게 해결 도와드리겠습니다! 많이 번거로우셨죠. 고객님의 소중한 권리를 되찾아드리기 위해 현재 상태를 체크해 보니, [계정 미연동] 상태로 확인됩니다. 보통 이 문제는 세 가지 경우에 발생하는데요. 바로 해결 가능한 '치트키'를 안내해 드릴게요. 고객님, 지금 바로 고객님의 휴대폰으로 '원클릭 연동 링크'를 발송해 드렸습니다. 이 링크를 누르시면 복잡한 절차 없이 구글 계정 선택만으로 연동이 완료됩니다. 성공하셨다니 정말 다행입니다! 이제 광고 없는 쾌적한 영상 시청 즐기시길 바랍니다. 추가로 5G 프리미엄 요금제에 포함된 다른 미디어 혜택(VOD 쿠폰 등)도 아직 미사용 중이신데, 관련 안내 문자를 함께 보내드릴까요? 별씀을요! 김아무개 고객님, 오늘도 프리미엄한 하루 보내세요. 지금까지 AICC 상담사였습니다!"
        + Response Body
            ```json
            {
                "result": [
                    {
                        "entity_group": "PS",
                        "score": 0.9185789227485657,
                        "word": "김아",
                        "start": 7,
                        "end": 9
                    },
                    {
                        "entity_group": "OG",
                        "score": 0.9643535614013672,
                        "word": "LG유플러스 AICC",
                        "start": 17,
                        "end": 28
                    },
                    {
                        "entity_group": "QT",
                        "score": 0.7189666628837585,
                        "word": "5G",
                        "start": 41,
                        "end": 43
                    },
                    {
                        "entity_group": "QT",
                        "score": 0.9896253347396851,
                        "word": "세 가지",
                        "start": 217,
                        "end": 221
                    },
                    {
                        "entity_group": "QT",
                        "score": 0.7113736271858215,
                        "word": "##G",
                        "start": 403,
                        "end": 404
                    },
                    {
                        "entity_group": "PS",
                        "score": 0.903061032295227,
                        "word": "김아",
                        "start": 480,
                        "end": 482
                    },
                    {
                        "entity_group": "DT",
                        "score": 0.9103196859359741,
                        "word": "오늘",
                        "start": 490,
                        "end": 492
                    },
                    {
                        "entity_group": "DT",
                        "score": 0.9374863505363464,
                        "word": "하루",
                        "start": 500,
                        "end": 502
                    },
                    {
                        "entity_group": "OG",
                        "score": 0.9354147911071777,
                        "word": "AICC",
                        "start": 514,
                        "end": 518
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