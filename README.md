# gpt-2.0-sherlock

Andrej Karpathy의 [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)를 기반으로,
character-level 언어 모델을 가장 단순한 **Bigram**에서 시작해 **MLP → Self-Attention → Tiny GPT**까지
단계적으로 직접 구현한 프로젝트입니다.

수업에서 다룬 코드를 토대로, 학습 데이터를 **셰익스피어(tinyshakespeare)에서 셜록홈즈로 교체**하고
최종 모델(6번)에 **temperature/top-k 샘플링**과 **어텐션 시각화**를 추가했습니다.

---

## 파일별 설명

### 1. `1_Bigram_Language_Model.ipynb` — Bigram (가장 단순한 모델)

가장 기본이 되는 모델입니다. **직전 한 글자만 보고 다음 글자를 예측**합니다.

- 데이터: `names.txt` (사람 이름 목록). 각 이름을 글자 시퀀스로 만들고, 시작/끝을 `.` 토큰으로 표시
- 구조: 현재 글자를 one-hot으로 만든 뒤, 학습 가능한 가중치 행렬 `W`(vocab × vocab)와 곱해 다음 글자의 logits를 얻음
- 핵심: 사실상 "어떤 글자 다음에 어떤 글자가 오는가"의 **빈도 통계를 경사하강법으로 학습하는 룩업 테이블**
- 학습 후 새 이름을 한 글자씩 샘플링해 생성

가장 단순하지만, 이후 모든 모델이 공유하는 **"다음 토큰 예측 + cross-entropy 손실 + 샘플링"** 이라는 언어 모델의 기본 골격을 여기서 잡습니다.

### 2. `2_MLP_Character_Model.ipynb` — MLP 문자 모델

Bigram의 한계(한 글자만 봄)를 넘어, **여러 글자의 문맥**을 보도록 확장합니다. (Bengio 2003 스타일)

- 데이터: `names.txt`, 문맥 길이(block_size) 3
- 구조: `Embedding`으로 각 글자를 벡터로 변환 → 직전 3글자 벡터를 이어붙임 → `Linear → Tanh → Linear`로 다음 글자 예측
- 핵심: 글자를 **임베딩 공간의 벡터**로 다루기 시작. 비슷한 역할의 글자가 가까운 벡터를 갖도록 학습됨
- Bigram보다 더 그럴듯한 이름을 생성

> 1~2번은 **사람 이름을 생성**하는 모델이라 데이터 구조(짧은 단어 단위)가 달라 `names.txt`를 그대로 사용합니다.

### 3. `3_mlp_sherlock.ipynb` — MLP를 긴 텍스트에 적용 (+ 데이터 교체)

같은 MLP 구조를 **이름 목록이 아닌 긴 산문(셜록홈즈)** 에 적용합니다.
이 단계부터 학습 데이터를 셰익스피어에서 셜록홈즈로 교체했습니다.

#### 데이터 교체: 셰익스피어 → 셜록홈즈

이 프로젝트의 핵심 변경점입니다.

- **원본**: 수업에서는 `tinyshakespeare`(셰익스피어 희곡 모음, 약 1MB)를 사용. `ROMEO:` 같은 화자 표기가 포함됨
- **교체**: 저작권이 만료된 퍼블릭 도메인 작품 **《The Adventures of Sherlock Holmes》**(Arthur Conan Doyle, 1892)로 변경. [Project Gutenberg eBook #1661](https://www.gutenberg.org/ebooks/1661)에서 받아옴

```python
URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
```

- **왜 가능한가**: 모델이 **문자 단위(character-level)** 로 동작하기 때문. 입력 텍스트의 등장 문자로 vocabulary를 만들고 "다음 글자"를 예측하므로, 텍스트 파일이기만 하면 어떤 글로도 학습 가능. **모델 코드는 그대로 두고 데이터 로딩만** 교체하면 됨
- **전처리**: Gutenberg 원문은 본문 앞뒤에 라이선스 안내문이 붙어 있어, 마커 기준으로 잘라내 **본문만** 학습 (안 그러면 모델이 법률 문구까지 흉내 냄)

```python
s = raw.find("*** START OF THE PROJECT GUTENBERG EBOOK")
e = raw.find("*** END OF THE PROJECT GUTENBERG EBOOK")
text = raw[raw.find("\n", s) + 1 : e].strip()
```

#### 이 단계의 의미
- 데이터: 셜록홈즈, 문맥 길이 16
- `CharSequenceNextCharDataset`: 연속된 16글자를 보고 다음 1글자를 예측 (슬라이딩 윈도우)
- 핵심: MLP가 이름 같은 짧은 단어뿐 아니라 **실제 문장/문단 단위의 긴 텍스트에도 확장**됨을 확인. vocab과 임베딩·은닉층 크기를 키움

### 4. `4_sequence_model_sherlock.ipynb` — GPT 스타일 데이터 + 최소 시퀀스 모델

GPT가 쓰는 **데이터 포맷과 위치 정보**를 도입합니다. (어텐션은 아직 없음)

- 데이터: 셜록홈즈, 문맥 길이 32. 입력 `x`와 정답 `y`가 모두 시퀀스이며, **`y`는 `x`를 한 칸 민 것** (각 위치에서 바로 다음 글자가 정답)
- 구조: `token_embedding`(글자 의미) + `position_embedding`(글자 위치)을 더한 뒤 `Linear`로 예측
- 핵심:
  - 한 번의 forward로 **모든 위치의 다음 글자를 동시에 예측** (출력이 `(B, T, V)`)
  - **위치 임베딩** 도입 — 트랜스포머는 순서를 내장하지 않으므로 위치 정보를 따로 더해줌
- 다만 아직 위치들이 서로 정보를 주고받지 못함. 각 위치가 **독립적으로** 예측

### 5. `5_single_head_attention_sherlock.ipynb` — 마스킹된 Single-Head Self-Attention

트랜스포머의 핵심인 **self-attention**을 한 개의 head로 도입합니다.

- 구조: 각 위치에서 `key`·`query`·`value` 벡터를 만들고, query와 key의 내적으로 "어디에 주목할지" 가중치(attention weight)를 계산해 value를 가중합
- **causal mask** (`tril`): 각 위치가 **자기보다 앞선 글자만** 참조하도록 미래를 가림 (언어 모델은 다음 글자를 미리 보면 안 되므로)
- 핵심: 4번에서 독립적이던 위치들이 **드디어 서로의 문맥을 참조**하기 시작. "필요한 이전 글자에 집중"하는 능력이 생김
- scaled dot-product (`* C**-0.5`)로 값이 너무 커지는 것을 방지

### 6. `6_tiny_gpt_sherlock.ipynb` — Tiny GPT (최종 모델 + 활용)

앞 단계들을 모아 **작은 GPT**를 완성합니다. ("Attention is All You Need" / GPT-2 구조의 축소판)

- 구조:
  - **Multi-Head Attention** — 여러 개의 어텐션 head를 병렬로 두어 다양한 관계를 동시에 포착
  - **FeedForward** — 위치별 비선형 변환
  - **residual connection** + **LayerNorm** — 깊게 쌓아도 학습이 안정되도록
  - 이 `Block`을 여러 층 쌓음 (기본 4 head × 4 layer, 문맥 길이 64)
- 이게 사실상 GPT의 핵심 아키텍처 전부입니다.

#### 추가한 활용

**(1) temperature / top-k 샘플링** — 생성의 다양성·일관성 조절
- `temperature`: 낮을수록(0.5) 안정적·반복적, 높을수록(1.2) 다양·산만
- `top-k`: 확률 상위 k개 글자만 후보로 두어 헛소리를 억제

```python
generate(model, ..., temperature=0.8, top_k=40)
```

**(2) 어텐션 시각화** — `Head`가 attention weight를 저장하게 하고 히트맵으로 그림
세로축(query)의 각 글자가 가로축(key)의 어떤 이전 글자에 주목하는지 보여주며,
causal mask 때문에 오른쪽 위 삼각형은 비어 있습니다. (얕은 층 vs 깊은 층의 패턴 차이를 비교)

---

## 결과

**학습 곡선 (6번)**
```
epoch  0 | train loss 2.6428
...
epoch 49 | train loss 1.2860
```

**temperature별 생성 예시 (시작어: "Sherlock Holmes")**
- `temperature=0.5` :

Sherlock Holmes came against to the singular side. I had not seemed to be described that I have
thought a case of the sevent which was standing as the country. It was perhaps the
danger was a long time, and a deal sensation which may be a long bring heart and
so provided that he was made a little record which she had been lost in the street in the matter,
the morning of the great streets of the son of the stable

- `temperature=0.8` : 

Sherlock Holmes seeing at the window which purposes and little remarkable and
search shrough the weave, dreeping in the pense and deadling off his
friends of Larmchaire of the Southam so. But its anything the allar
about the difficty said his askine. I should not be denience. Just how successes communications when I could not
see a little little pointed in a very such in which we peach was
a lady strong and a lo

- `temperature=1.2` :

Sherlock Holmes came at trains why—I have gives. She curious, was
as so rushed her household.’

“It is in your client, though put his church ads. Graveful belober
singular way so nofdening days, ill-box, who have are neatenhall astoner that
a save flame-position large man, with his specked seen upon the dresses, future and sun.
Holmes that I only would such that you had premised to in him. He
is see, begged up I


**어텐션 히트맵**
<img width="868" height="790" alt="image" src="https://github.com/user-attachments/assets/87593a8f-f67d-4115-b5db-3f52f0ae5c80" />


> 참고: 작은 char-level 모델이라 완벽한 문장은 생성되지 않습니다.
> 셜록홈즈 문체의 단어·철자·구두점 패턴을 흉내 내는 수준을 확인하는 것이 목적입니다.

---

### Open in Colab

| 단계 | 링크 |
|------|------|
| 1. Bigram | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chloe8407/gpt-2.0-sherlock/blob/main/1_Bigram_Language_Model.ipynb) |
| 2. MLP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chloe8407/gpt-2.0-sherlock/blob/main/2_MLP_Character_Model.ipynb) |
| 3. MLP (셜록) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chloe8407/gpt-2.0-sherlock/blob/main/3_mlp_sherlock.ipynb) |
| 4. Sequence (셜록) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chloe8407/gpt-2.0-sherlock/blob/main/4_sequence_model_sherlock.ipynb) |
| 5. Attention (셜록) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chloe8407/gpt-2.0-sherlock/blob/main/5_single_head_attention_sherlock.ipynb) |
| 6. Tiny GPT (셜록) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chloe8407/gpt-2.0-sherlock/blob/main/6_tiny_gpt_sherlock.ipynb) |

---

### 핵심 개념 정리
최종 모델을 한 문장으로 요약하면 다음과 같습니다.

글자를 의미 벡터로 바꾸고(임베딩) → 글자끼리 정보를 주고받게 하고(어텐션) → 그 블록을 여러 층 쌓아 → 다음 글자를 예측한다.

각 구성 요소가 "무엇을, 왜" 하는지 정리합니다.
토큰화와 임베딩
모델은 글자를 직접 다루지 못하므로, 글자마다 번호를 매겨 정수 시퀀스로 바꿉니다(토큰화).
하지만 번호 자체에는 의미가 없습니다. 임베딩은 글자마다 학습되는 벡터를 부여하는 표로,
학습이 진행되면 비슷한 역할의 글자가 비슷한 벡터를 갖게 됩니다.
즉 "글자를 의미를 담은 벡터로 바꾸는 단계"입니다.
위치 임베딩
어텐션은 모든 글자를 동시에 처리하기 때문에 순서 정보를 갖지 못합니다.
그래서 "몇 번째 위치인지"를 나타내는 벡터를 따로 만들어 글자 벡터에 더해줍니다.
이로써 모델은 글자의 정체와 위치를 함께 알게 됩니다.
Self-Attention
각 글자가 다음 글자를 예측하기 위해 앞의 어떤 글자를 참조할지 스스로 정하는 메커니즘입니다.

Query: 현재 글자가 찾는 정보
Key: 각 글자가 가진 정보
Value: 실제로 전달되는 내용

Query와 Key를 비교해 주목할 정도를 정한 뒤, 그 비율로 Value를 가중합합니다.
결과적으로 각 글자는 문맥에서 필요한 이전 글자의 정보를 끌어옵니다.
Causal Mask
이 모델의 목적은 다음 글자 예측이므로, 각 위치가 미래(자기 뒤) 글자를 보면
정답을 미리 보는 셈이 됩니다. 따라서 자기보다 뒤쪽은 참조하지 못하도록 가립니다.
어텐션 가중치 히트맵이 아래쪽 삼각형 형태가 되는 이유입니다.
Multi-Head Attention
어텐션 하나는 한 가지 관점만 학습합니다. 여러 head를 병렬로 두면
서로 다른 종류의 관계(예: 바로 앞 글자, 따옴표 짝, 단어 경계 등)를 동시에 포착해 종합할 수 있습니다.
FeedForward
어텐션이 위치 간 정보를 교환하는 단계라면, FeedForward는
각 위치가 받아온 정보를 독립적으로 한 번 더 가공하는 단계입니다.
어텐션(소통)과 FFN(개별 처리)이 한 쌍으로 동작합니다.
층 쌓기 (Multi-Layer)
어텐션과 FFN 한 묶음(블록)을 위로 여러 층 쌓습니다.
얕은 층은 단순한 패턴(철자 조합)을, 깊은 층은 더 추상적인 패턴(단어·문장 구조)을 학습합니다.

05번(single-head, 1층)과 06번의 핵심 차이는 두 종류의 "여러 개"입니다.
같은 층 안에서 어텐션을 여러 개(멀티헤드) 두고, 그 블록을 위로 여러 층(멀티레이어) 쌓았다는 점입니다.

Residual Connection & LayerNorm

Residual: 각 단계에서 입력을 그대로 더해주는 지름길. 층이 깊어져도 정보와 학습 신호가
사라지지 않게 하여 안정적인 학습을 돕습니다.
LayerNorm: 단계를 거치며 값이 너무 커지거나 들쭉날쭉해지는 것을 막아, 값 분포를
일정하게 정규화합니다.

다음 글자 예측과 생성
마지막에 각 위치의 벡터를 "다음에 올 글자 점수"로 변환하고(lm_head),
실제 다음 글자와 비교해 손실(cross-entropy)을 계산하며 학습합니다.
생성 시에는 한 글자를 뽑아 입력 뒤에 붙이고 다시 넣는 과정을 반복합니다(자기회귀).
이때 temperature로 다양성을, top-k로 후보 범위를 조절합니다.
데이터 교체가 가능한 이유
이 모델은 글자 단위(character-level)로 학습하므로, 텍스트만 교체하면
등장 글자로 vocabulary가 자동으로 다시 구성되고 임베딩도 거기 맞춰집니다.
모델 구조는 데이터에 무관하므로, 셰익스피어를 셜록홈즈로 바꾸는 데에는 데이터 로딩 부분만 수정하면 됩니다.

---

## 참고

- 데이터: [The Adventures of Sherlock Holmes — Project Gutenberg #1661](https://www.gutenberg.org/ebooks/1661) (Public Domain)
