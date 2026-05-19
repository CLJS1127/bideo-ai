# 🧠 BIDEO AI — 영상 작품 마켓플레이스 머신러닝 서버

> BIDEO 의 큐레이션, 가격 예측, 콘텐츠 보호를 담당하는
> **FastAPI 기반 단일 ML 추론 서버**.
>
> 분류·회귀·RAG·워터마크 — 네 가지 모델을 하나의 API 로 통합 제공합니다.

🔗 **Web 리포지토리**: [bideo-web](https://github.com/your-org/bideo-web) — Spring Boot 본 서비스

---

## 📌 왜 만들었나

영상 작품 마켓플레이스에서 데이터가 의미를 갖는 순간은 다음 세 가지입니다.

| 사용자 시점 | 필요한 AI |
|---|---|
| "어떤 작품이 잘 팔릴까?" | **분류 모델** — 경매 낙찰 확률 예측 |
| "이 작가는 얼마나 성장할까?" | **회귀 모델** — 팔로워 성장 곡선 예측 |
| "내 취향에 맞는 작품 추천해줘" | **RAG** — 벡터 검색 + LLM 추천 |
| "내 작품이 무단 사용되면 어쩌지?" | **워터마크** — 보이지 않는 저작권 마킹 |

BIDEO AI 서버는 위 네 모델을 통합해 Spring 본 서버의 HTTP 호출에 응답합니다.

---

## 🧠 모델 라인업

### 1. **분류 모델 (Classification) — 경매 낙찰 예측**

| 항목 | 내용 |
|---|---|
| 목적 | 진행 중인 경매가 낙찰될 확률 점수화 (0.0 ~ 1.0) |
| 입력 Feature | 작가 팔로워, 시작가, 카테고리, 작품 조회수, 좋아요, 등록 후 경과 시간 등 |
| 모델 | Gradient Boosting (XGBoost / LightGBM) |
| 임계값 | `score ≥ 0.5294` 시 "낙찰 가능" 라벨 |
| 사용처 | 메인 페이지 AI 큐레이션 Top-K, 작품 상세 페이지 예측 배지 |

**API**: `GET /api/predictions/curation?k=10` / `POST /api/predictions/predict`

---

### 2. **회귀 모델 (Regression) — 작가 팔로워 성장 예측**

| 항목 | 내용 |
|---|---|
| 목적 | 작가의 향후 N일 후 팔로워 수 예측 |
| 입력 Feature | 현재 팔로워, 최근 일주일 작품 업로드 수, 평균 좋아요, 가입 경과일 등 |
| 모델 | Linear Regression / Random Forest Regressor |
| 출력 | 7일 / 30일 / 90일 후 예상 팔로워 수 + 신뢰구간 |
| 사용처 | 작가 대시보드 성장 차트, 자기 작가 검증·승급 판단 |

**API**: `POST /api/growth/forecast` / `GET /api/growth/me`

---

### 3. **RAG (Retrieval-Augmented Generation) — 의미 기반 작품 추천**

| 항목 | 내용 |
|---|---|
| 목적 | 사용자 쿼리·취향 기반 작품 추천을 자연어로 설명 |
| 임베딩 | 작품 제목·설명·태그를 sentence-transformers 로 벡터화 |
| Vector Store | FAISS (또는 Chroma) — 빠른 cosine similarity 검색 |
| LLM | OpenAI GPT-4o-mini / 로컬 LLM (Llama 3.1 등) |
| 흐름 | 쿼리 임베딩 → Top-K 검색 → 컨텍스트와 함께 LLM 프롬프트 → 자연어 추천 응답 |
| 사용처 | "비슷한 작가 추천", 메인 "오늘의 큐레이션 코멘트", 작품 검색 |

**API**: `GET /api/recommend/creators/similar/{id}` / `GET /api/recommend/creators/me`

---

### 4. **워터마크 (Watermarking) — 작품 저작권 보호**

| 항목 | 내용 |
|---|---|
| 목적 | 작가의 작품에 **육안으로 보이지 않는 워터마크** 자동 삽입 → 도용 시 추적 가능 |
| 방식 | DCT(Discrete Cosine Transform) 기반 주파수 도메인 워터마킹 |
| 입력 | 원본 이미지 + 작가 ID·작품 ID 인코딩 비트열 |
| 출력 | 시각적으로 동일한 이미지 + 워터마크 메타데이터 |
| 검증 | 추후 의심 이미지에서 워터마크 추출 → 원작자 식별 |
| 사용처 | 작품 등록 시 자동 적용, 도용 신고 시 원작자 검증 |

**API**: `POST /api/watermark/embed` / `POST /api/watermark/extract`

---

## 🛠 기술 스택

| 분류 | 사용 기술 |
|---|---|
| **Framework** | FastAPI (Python 3.11+) |
| **ML 분류·회귀** | scikit-learn, XGBoost, LightGBM, pandas, numpy |
| **RAG** | sentence-transformers, FAISS, LangChain, OpenAI API (or Llama 로컬) |
| **워터마크** | OpenCV, PyWavelets, NumPy (DCT 도메인 처리) |
| **데이터** | PostgreSQL (BIDEO DB 직접 read-only 조회) |
| **배포** | Docker, **Cloudflare Tunnel** (로컬 GPU PC → 외부 노출) |
| **연동** | Spring Boot 본 서버가 `ML_API_BASE_URL` 환경변수로 호출 |

### 시스템 흐름

```
[Spring Boot (EC2)]
      ↓ HTTP
[Cloudflare Tunnel: bideo-ml.trycloudflare.com]
      ↓
[로컬 GPU PC : FastAPI :8000]
      ├─→ 분류 모델 (낙찰 예측)
      ├─→ 회귀 모델 (성장 예측)
      ├─→ RAG 파이프라인 (FAISS + LLM)
      └─→ 워터마크 모듈 (DCT)
            ↑
      [BIDEO PostgreSQL (read-only)]
```

---

## 📂 프로젝트 구조

```
bideo-ai/
├── app/
│   ├── main.py                     # FastAPI entrypoint
│   ├── routers/
│   │   ├── predictions.py          # 분류 (낙찰 예측)
│   │   ├── growth.py               # 회귀 (성장 예측)
│   │   ├── recommend.py            # RAG 추천
│   │   └── watermark.py            # 워터마크
│   ├── models/
│   │   ├── classifier.pkl          # 학습된 분류 모델
│   │   ├── regressor.pkl           # 학습된 회귀 모델
│   │   └── faiss_index.bin         # 벡터 인덱스
│   ├── services/
│   │   ├── feature_engineering.py
│   │   ├── rag_pipeline.py
│   │   └── watermark_dct.py
│   └── db.py                       # PostgreSQL 연결
├── train/
│   ├── train_classifier.ipynb
│   ├── train_regressor.ipynb
│   └── build_faiss_index.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 데이터 & 학습 파이프라인

### 학습 데이터
- **출처**: BIDEO 자체 DB (`tbl_auction`, `tbl_work`, `tbl_member`, `tbl_follow` 등)
- **샘플**: 작품 20,000개 / 경매 3,000건 / 회원 5,000명 / 팔로우 150,000건 (시드 데이터 기준)

### 분류·회귀 파이프라인
```
PostgreSQL → pandas DataFrame
  → Feature Engineering (정규화, 결측치, 카테고리 인코딩)
  → train/test split (8:2)
  → 모델 학습 (GridSearch CV)
  → 모델 평가 (Accuracy / AUC / RMSE)
  → pickle 저장 → FastAPI 가 시작 시 로드
```

### RAG 인덱스 빌드
```
작품 데이터 SELECT
  → 제목·설명·태그 concat
  → sentence-transformers 임베딩 (768d)
  → FAISS IndexFlatIP 빌드
  → 디스크 저장 (faiss_index.bin)
  → 신규 작품은 nightly job 으로 incremental update
```

### 워터마크 적용
```
작품 업로드 시 Spring 이 /api/watermark/embed 호출
  → 원본 이미지 + (memberId, workId) 비트열 입력
  → OpenCV 로 DCT 변환 → 중간 주파수 대역에 비트 삽입
  → IDCT → 시각적으로 동일한 출력
  → S3 업로드
```

---

## 🚀 실행 방법

### 로컬 실행
```bash
git clone https://github.com/your-org/bideo-ai.git
cd bideo-ai

# 가상환경
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 의존성
pip install -r requirements.txt

# 환경변수 (.env)
DATABASE_URL=postgresql://bideo:****@localhost:5432/bideo
OPENAI_API_KEY=sk-...

# 서버 실행
uvicorn app.main:app --reload --port 8000
```

### Docker 실행
```bash
docker build -t bideo-ai .
docker run -d -p 8000:8000 --env-file .env --name bideo-ai bideo-ai
```

### Cloudflare Tunnel 로 외부 노출
```bash
cloudflared tunnel --url http://localhost:8000
```
출력된 `*.trycloudflare.com` URL 을 Spring 본 서버의 `ML_API_BASE_URL` 환경변수에 등록.

---

## 🔧 트러블슈팅

### 1. Cloudflare Tunnel 재시작 시 URL 변경

**증상**: Quick Tunnel 재실행 시 URL 이 매번 바뀌어 EC2 환경변수 매번 갱신 필요

**해결**: Cloudflare Zero Trust 대시보드에서 **Named Tunnel** 생성 → 본인 도메인의 서브도메인(`ml.bideo.ai.kr`)에 영구 매핑

---

### 2. ML 호출 시 Connection refused

**증상**: Spring 측 로그에 `[ML] curation 호출 실패: Connection refused`

**원인**: 로컬 FastAPI 가 꺼져있거나 cloudflared 터널 종료

**해결**: 두 프로세스 모두 살아있어야 함을 인지. 시연 전 두 cmd 창 켜둠
```bash
# 창 1
uvicorn app.main:app --port 8000

# 창 2
cloudflared tunnel --url http://localhost:8000
```

---

### 3. FAISS 인덱스 크기로 인한 메모리 부담

**증상**: 작품 수가 늘어나면서 FAISS 인덱스 메모리 사용량 급증

**대처**: IVF (Inverted File Index) 도입 → `IndexFlatIP` → `IndexIVFFlat` 로 변경, 검색 정확도 99% 유지하면서 메모리 1/10 감소

---

### 4. LLM 응답 지연

**증상**: GPT-4o-mini 호출이 1~2초 지연되어 메인 페이지 로딩 체감 느림

**해결**:
- 큐레이션 결과를 **Redis 에 1시간 캐싱** (Spring 측)
- RAG 응답은 **stream 으로 토큰 단위 점진 전달** (FastAPI `StreamingResponse`)

---

## 📈 모델 성능 (예시 수치)

| 모델 | 지표 | 값 |
|---|---|---|
| 분류 (낙찰 예측) | ROC-AUC | 0.847 |
| 분류 (낙찰 예측) | Accuracy | 79.3% |
| 회귀 (성장 예측) | RMSE | ±42 followers |
| 회귀 (성장 예측) | R² | 0.71 |
| RAG (Top-5 추천) | nDCG@5 | 0.682 |
| 워터마크 | 추출 정확도 | 99.4% (JPEG 압축 70% 이내) |

---

## 🎓 회고

### 잘한 점
- **단일 서버에 4개 도메인 모델 통합**: FastAPI 의 router 구조로 모델별 책임 분리, 유지보수 용이
- **로컬 GPU 활용 + Cloudflare Tunnel**: 클라우드 GPU 비용 없이 시연 환경 구축
- **워터마크까지 직접 구현**: 단순 ML 추천을 넘어 콘텐츠 보호 영역까지 확장

### 아쉬운 점
- **모델 버전 관리 부재**: MLflow 같은 도구 없이 pickle 파일로만 관리. 다음엔 실험 트래킹 필요
- **재학습 자동화 미구축**: 신규 데이터 누적 시 수동으로 재학습. Airflow / Prefect 같은 워크플로우 도구로 nightly retrain 구축 필요

### 배운 점
- **ML 모델은 0.1% 의 끝이고 99.9% 가 데이터·인프라**: 모델 자체보다 feature 설계 · 데이터 정제 · 배포 환경이 결과 품질을 좌우
- **AI 와 서비스의 경계 설계**: AI 서버는 stateless 추론만 담당하고, 도메인 로직은 Spring 본 서버에 둬야 양쪽 모두 유지 가능

---

🔗 **Web 리포지토리**: [bideo-web](https://github.com/your-org/bideo-web)
