# 🤖 BIDEO AI — 모델 설계 & 운영 문서

> 본 문서는 **BIDEO 의 ML 모델 (분류 + 회귀)** 을 어떻게 설계 / 학습 / 서빙했는지 코드와 함께 정리한 자료입니다.
>
> 웹 페이지 단위 기능은 별도 [README.md](../../../aws/workspace/bideo/README.md) 참조.

---

## 🙋 담당 모델 한눈에

| 모델 | 분류 | 핵심 책임 |
|---|---|---|
| [1. 경매 낙찰 분류](#1-경매-낙찰-분류-classification) | Classification (이진) | 메인 "AI PICK" 큐레이션 Top-K 점수 |
| [2. 작가 팔로워 성장 회귀](#2-작가-팔로워-성장-회귀-regression) | Regression | 마이페이지 N 주 후 팔로워 예측 |
| [3. 모델 서빙](#3-모델-서빙-fastapi) | Infra | FastAPI + asyncpg + Redis 캐시 |
| [4. Spring 연동](#4-spring-연동) | Infra | RestTemplate + DTO 매핑 |

```
[Spring (web)] → /api/prediction/* → [FastAPI (ML)] → joblib 모델 + Postgres
                                          ↑
                                  Redis cache (TTL 30s)
```

---

## 1. 경매 낙찰 분류 (Classification)

### 무엇을 예측?

종료된 경매의 `SOLD vs EXPIRED` 라벨로 학습한 이진 분류 모델. 진행 중(`ACTIVE`) 경매에
"낙찰 확률 점수" 를 매겨 메인페이지 **"서두르세요! 곧 낙찰될 것 같아요!"** 슬롯의 Top-K 노출에 사용.

### 학습 데이터 — BIDEO DB 직접 조회 (`train_real.py`)

기존 `train.py` 는 100% 합성 데이터로 학습되어 실제 BIDEO 분포와 어긋났음.
실데이터 기반으로 전환.

```sql
SELECT a.id AS auction_id, a.starting_price,
       extract(epoch FROM (a.closing_at - a.started_at))/3600.0 AS duration_hours,
       extract(hour FROM a.started_at) AS started_hour,
       extract(dow  FROM a.started_at) AS started_dow,
       w.category, w.view_count, w.like_count, w.save_count, w.comment_count,
       m.creator_verified,
       (SELECT COUNT(*) FROM tbl_follow WHERE following_id = a.seller_id) AS creator_followers,
       -- 작가 이력 (누설 방지: 현재 경매 id 이전만)
       (SELECT COUNT(*) FROM tbl_auction p
          WHERE p.seller_id = a.seller_id AND p.id < a.id
            AND p.status IN ('SOLD','EXPIRED','CLOSED'))  AS prior_total,
       (SELECT COUNT(*) FROM tbl_auction p
          WHERE p.seller_id = a.seller_id AND p.id < a.id
            AND p.status = 'SOLD')                         AS prior_sold,
       (SELECT COUNT(*) FROM tbl_work w
          WHERE w.member_id = a.seller_id AND w.deleted_datetime IS NULL) AS seller_n_works,
       CASE WHEN a.status = 'SOLD' THEN 1 ELSE 0 END AS is_won
  FROM tbl_auction a
  LEFT JOIN tbl_work w ON w.id = a.work_id
  LEFT JOIN tbl_member m ON m.id = a.seller_id
 WHERE a.status IN ('SOLD','EXPIRED','CLOSED')
   AND a.closing_at > a.started_at;
```

종료된 경매 ~1,800건 / 양성률 ~38%.

### 피처

#### 기본
- `price_log` — 가격(log1p, 낮을수록 낙찰 ↑)
- `view_log`, `like_log`, `bookmark_log` — 작품 인기
- `follower_log` — 작가 영향력
- `creator_verified` — 인증 작가 0/1
- `is_weekend`, `started_hour`, `started_dow` — 시간 파생
- `work_category` (OneHot 22종)

#### 신규 파생 (`train_real.py`)
- `prior_total_log`, `prior_sold_log`, `prior_sold_rate` — 작가의 직전 경매 이력 (누설 X)
- `seller_n_works_log` — 활성 작품 수
- `save_per_view`, `comment_per_view`, `like_per_view` — 인게이지먼트 비율

### 모델

DT (`GridSearchCV`) + RF (`n_estimators=500`) + GB (`n_estimators=400, lr=0.05, depth=4`) 3종 학습 후
**AUC 최고 모델 자동 선택**.

```python
metrics = {'rf': {...}, 'gb': {...}}
best_name = max(['rf','gb'], key=lambda k: metrics[k]['default_0_5']['auc'])
joblib.dump({
    'model_dt':     dtc,
    'model_rf':     rfc if best_name == 'rf' else gbc,   # predictor 호환을 위해 동일 키
    'features':     feature_columns,
    'rf_threshold': best_thresh,
    'cat_encoder':  cat_encoder,
    'best_model':   best_name,
}, out_path)
```

### 결과

| 단계 | AUC | F1 | acc |
|---|---:|---:|---:|
| 합성 모델을 BIDEO 시드에 그대로 적용 | 0.47 (랜덤 미만) | 0.65 (majority class) | 0.54 |
| 실데이터 학습 (시드 약시그널) | 0.68 | 0.52 | 0.67 |
| **+ 시드 강화 + 파생 피처 + GB** | **0.79** | **0.62** | **0.74** |

### 🔧 트러블슈팅 — "큐레이션 점수가 의미 없이 0.5 근처에서 흔들림"

**증상**: 메인 큐레이션 Top-10 에 들쭉날쭉한 점수의 작품이 나옴. 모델의 변별력이 없어 보임.

**원인 1**: 합성 학습 시 가정한 분포와 실제 BIDEO 시드 분포 불일치.
시그널이 있는 시드인 줄 알았으나 실측해보니 `SOLD/EXPIRED` 라벨이 `view/like/follower/verified` 와 거의 무상관.

| 피처 | not-SOLD 평균 | SOLD 평균 | lift |
|---|---:|---:|---:|
| starting_price | 78,299 | 78,051 | 1.00x |
| view_count | 26.5 | 22.9 | **0.87x (역방향)** |
| follower | 40.4 | 39.6 | 0.98x |
| bid_count | 0.0 | 14.4 | ∞ (라벨 누설 — 사용 금지) |

→ **재학습으로 해결 불가. 시드 자체를 고쳐야 했음.**

**원인 2**: 시드 SQL 의 `quality_raw` 공식이 분산을 너무 좁게 만듦.

```sql
-- 기존 (deploy_setup.sql 원본)
quality_raw = ln(v_cnt+1) + 1.5*ln(s_cnt+1) + 2.0*ln(f_cnt+1)
SOLD 확률 = 0.15 + 0.70 * min(1, quality_raw / 18.0)
```

대부분의 작품이 quality 13~15 → 확률 0.65~0.75 로 몰림 → 사실상 무작위.

**해결**: 시드 INSERT 로직을 **다중 피처 + 강한 가중치 + 약한 노이즈** 로 재설계.

```sql
update _auction_cand
   set quality_logit =
         -0.50
         + 0.50 * (ln(v_cnt + 1) - 3.0)      -- view
         + 0.65 * (ln(s_cnt + 1) - 2.0)      -- save
         + 0.75 * (ln(f_cnt + 1) - 3.5)      -- follower
         + 2.50 * verified                    -- creator_verified
         - 1.00 * (ln(sp / 50000.0))         -- price ↓ → 낙찰 ↑
         + case when category in ('SF','CINEMATIC','3D_ANIMATION',
                                  'AI_GENERATED','디자인')   then  0.80
                when category in ('포트폴리오','일상','패션') then -0.80
                else 0.0 end                  -- 카테고리 효과
         + case when extract(dow from started_at_t) >= 5 then 0.35 else 0.0 end
         + r_noise;                           -- N(0, 0.17)
update _auction_cand
   set final_status = case
     when r_state < 0.35 then 'ACTIVE'
     when random() < (1.0 / (1.0 + exp(-quality_logit))) then 'SOLD'
     else 'EXPIRED'
   end;
```

**결과**: AUC **0.47 → 0.79**.

### 🔧 트러블슈팅 — 큐레이션에 마감된 경매가 노출됨

**증상**: 메인 페이지 "곧 낙찰될 것 같아요!" 슬롯에 진행 중이 아닌 작품이 섞여 보임.

**원인**: ML 서버 큐레이션 쿼리가 `WHERE status='ACTIVE'` 만 보고 `closing_at` 시간 체크를 안 함.
Spring 의 `AuctionClosureService` 가 10초 주기로 `closing_at <= now()` 인 ACTIVE 를 CLOSED 로 바꾸지만,
그 사이의 좀비 경매가 큐레이션에 끼어듦.

**해결**: 쿼리에 시간 필터 추가.

```python
# api/repository/auction_repository.py
WHERE a.status = 'ACTIVE'
  AND a.closing_at > now()      # 좀비 차단
```

### 추론 파이프라인

```
Spring                                   FastAPI                           Postgres
  │                                         │
  └─ GET /api/prediction/curation?k=10 ─▶  │
                                            │
                                            ├─ Redis cache (TTL 30s) ─ hit? 즉시 반환
                                            │
                                            ├─ AuctionRepository.find_active_auctions
                                            │     → 500건 + 모든 피처 일괄 join (1 쿼리)
                                            │
                                            ├─ predictor.featurize_batch
                                            │     → log1p / 비율 / OneHot 변환
                                            │
                                            ├─ model_rf.predict_proba_batch
                                            │     → 500건 < 1s
                                            │
                                            └─ Top-K 정렬 후 반환
```

500건 추론도 1초 미만으로 끝나도록 **단건 N번이 아닌 배치 1번**으로 설계.

```python
# 단순 단건 N번 (느림)
[predictor.predict_proba(featurize(r)) for r in rows]

# 배치 1번 (운영 채택)
X = predictor.featurize_batch(rows)
scores = predictor.predict_proba_batch(X)
```

---

## 2. 작가 팔로워 성장 회귀 (Regression)

### 무엇을 예측?

마이페이지에서 **"12주 후 당신의 팔로워 수"** 를 보여주는 회귀 모델.

### 🔧 트러블슈팅 — "0 작품 / 0 팔로워 신규 가입자에게 +46명 예측"

**증상**: 작품 한 개도 안 올린 신규 사용자가 마이페이지에 들어가니 "12주 후 +46명" 표시.

**원인**: 합성 학습 모델의 `forecast()` 가 **시간(week)과 등급(grade)** 만 보고 예측.

```python
# 기존 (train_growth.py + follower_growth_predictor.py)
def forecast(grade, current_week, weeks=12):
    m = models[grade]                              # grade 는 follower_count 만 보고 결정
    pred = m.predict([[current_week + weeks]])[0]  # X = [target_week] 뿐!
    return ...
```

- 사용자의 활동 (작품 수, 인게이지먼트) 을 전혀 보지 않음
- "NEWCOMER 평균 성장 곡선" 을 누구에게나 동일 적용 → 활동 0 인 사람도 +46명

→ **모델 디자인 문제**. 입력 자체를 재설계.

### 해결 — 활동 피처 추가 + 실데이터 학습 (`train_growth_real.py`)

#### 새 입력
- `current_week_offset` — **첫 팔로워 이후** 경과 주 (가입 → 데뷔 기준 변경)
- `current_followers`
- `n_works` — 업로드 작품 수
- `recent_velocity_4w` — 최근 4주간 팔로워 증감 (음수 가능)
- `creator_verified`
- `weeks_ahead` — 예측 horizon (4/8/12 주)

#### 학습 데이터 — 실제 BIDEO follow 이벤트 재구성

`tbl_follow.created_datetime` 으로 임의 시점 T 의 누적 follower 수를 정확히 계산 가능.

```python
def cum_count(events, until):
    """정렬된 timestamp 리스트에서 until 이하 개수 (이분탐색)."""
    lo, hi = 0, len(events)
    while lo < hi:
        mid = (lo + hi) // 2
        if events[mid] <= until: lo = mid + 1
        else:                    hi = mid
    return lo
```

작가별로 데뷔(첫 팔로워) 후 4주 ~ (데이터끝 − ahead주) 구간에서 3개씩 스냅샷 샘플링 × ahead ∈ {4, 8, 12}
→ **45,909 row** 학습 데이터셋.

#### 모델

`GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05)`.

### 결과

| 항목 | 합성 (등급별 다항) | **실데이터 (GB)** |
|---|---:|---:|
| R² | ~0.19 ~ 0.85 (등급별) | **0.99** |
| MAE | 수십 followers | **1.82** |
| resid_std | 30 ~ 100 | **2.90** |

피처 중요도 — `current_followers_log` 가 0.94 로 가장 큼. BIDEO 시드에서 follow 이벤트가 작품 활동과
거의 독립이라 `n_works` 의 직접 효과는 작음. 그래도 **0 팔로워 → 거의 0 성장** 으로 정상 예측됨.

```
0 작품 / 0 팔로워        →  +6명     (이전 모델: +46명)
5 작품 / 10 팔로워       →  +10명
30 작품 / 100 팔로워 / verified → +45명
100 작품 / 500 팔로워 / verified → +90명
```

### API 컨트랙트 변경

모델 입력이 늘었으므로 FastAPI Pydantic + Spring DTO + Spring 서비스 모두 갱신.

```python
# api/domain/follower_growth.py
class FollowerGrowthRequest(BaseModel):
    current_week_offset: int
    current_followers:   int
    weeks_ahead:         int = 12
    n_works:             int = 0   # NEW
    recent_velocity_4w:  int = 0   # NEW
    creator_verified:    int = 0   # NEW
```

Spring 측 `FollowerGrowthService.forecastForMember(memberId)` 가 DB 에서 활동 피처를 채워 호출:

```java
LocalDateTime debutAt = memberRepository.findFirstFollowAt(memberId).orElse(null);
int weekOffset = debutAt == null ? 0 : weeksBetween(debutAt, now);
int nWorks   = memberRepository.countActiveWorksByMemberId(memberId);
int velocity = followers - memberRepository.countFollowersBefore(memberId, now.minusWeeks(4));
int verified = Boolean.TRUE.equals(member.getCreatorVerified()) ? 1 : 0;

return apiClient.forecast(FollowerGrowthRequestDTO.builder()
        .currentWeekOffset(weekOffset)
        .currentFollowers(followers)
        .weeksAhead(12)
        .nWorks(nWorks)
        .recentVelocity4w(velocity)
        .creatorVerified(verified)
        .build());
```

---

## 3. 모델 서빙 (FastAPI)

### 구조

```
api/
├── main.py                 # FastAPI app + lifespan (모델 load/unload)
├── database.py             # asyncpg pool
├── cache.py                # Redis 래퍼
├── router/
│   ├── prediction.py       # /api/predictions/{predict, curation, ...}
│   └── follower_growth.py  # /api/follower-growth/{forecast, forecast-curve}
├── service/
│   ├── prediction_service.py
│   └── follower_growth_service.py
├── repository/
│   └── auction_repository.py
├── ml/
│   ├── predictor.py                  # joblib load + featurize/predict
│   └── follower_growth_predictor.py
└── domain/                # Pydantic 입출력 스키마
```

### Lifespan 으로 모델 한 번만 로드

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()         # auction_classifier_v1.pkl
    growth_predictor.load()  # creator_follower_growth_v1.pkl
    await db.connect()
    yield
    await db.disconnect()
```

→ 매 요청마다 18MB pkl 다시 안 읽음. 모델 교체 시 FastAPI 재시작이면 충분.

### Redis 캐시

큐레이션은 30초 TTL — 같은 사용자가 페이지 이동해도 ML 추론 부담 없음.

```python
CURATION_CACHE_TTL = 30
async def get_curation(self, k=10, scan_limit=500):
    key = f"curation:k={k}:scan={scan_limit}"
    if cached := cache.get(key):
        return CurationResponse(**json.loads(cached))
    # ... 추론 ...
    cache.setex(key, json.dumps(resp.model_dump()), ttl=CURATION_CACHE_TTL)
```

단일 `/predict` 는 입력 해시(MD5)를 키로 — 같은 경매 다시 물어도 즉시 반환.

---

## 4. Spring 연동

`PredictionApiClient` (분류) + `FollowerGrowthApiClient` (회귀) 가 RestTemplate 으로 ML 서버 호출.

```java
@Component
public class PredictionApiClient {
    @Qualifier("mlApiRestTemplate") private final RestTemplate restTemplate;
    @Value("${ml.api.base-url}") private String baseUrl;

    public CurationResponseDTO getCuration(int k) {
        String url = UriComponentsBuilder.fromHttpUrl(baseUrl + "/api/predictions/curation")
                .queryParam("k", k).queryParam("scan_limit", 100)
                .toUriString();
        return restTemplate.getForObject(url, CurationResponseDTO.class);
    }
}
```

### 🔧 트러블슈팅 — Spring 과 FastAPI 가 서로 다른 DB 를 봄

**증상**: 큐레이션 카드를 클릭해 들어가니 "이 작품은 경매 중이 아닙니다" 표시. AI 가 추천한 작품이
실제로는 경매 중이 아닌 사례 다수.

**원인**: FastAPI 의 `api/.env` 에 `DATABASE_URL=postgresql://bideo:1234@localhost:5432/bideo`,
Spring 의 yaml 은 `jdbc:postgresql://bideo.ai.kr:5432/bideo`. **서로 다른 DB**.
FastAPI 로컬 DB 의 ACTIVE 가 EC2 DB 에선 이미 마감.

**해결**: FastAPI `.env` 의 DSN 을 Spring 과 동일한 EC2 DB 로 통일.

```diff
- DATABASE_URL=postgresql://bideo:1234@localhost:5432/bideo
+ DATABASE_URL=postgresql://bideo:1234@bideo.ai.kr:5432/bideo
```

---

## 🎓 회고

### 잘한 점
- **합성 → 실데이터** 로 전 과정 재정렬. AUC 0.47 → 0.79 / 회귀 "0작품 → +46명" 버그 → +6명으로 교정.
- **시드 데이터 자체의 한계**를 인정하고 시드 SQL 부터 고침. "모델만 잘하면 된다" 가 아니었음.
- **누설 없는 작가 이력 피처** (`prior_total/prior_sold` 의 `p.id < a.id` 조건) 로 데이터 누설 방지.
- **모델 교체 = pkl 교체 + 재시작** — 인프라 변경 없이 운영 가능한 구조 유지.

### 아쉬운 점
- BIDEO 시드 데이터가 본질적으로 합성이라 모델의 "최선" 이 시드 식을 역추론하는 수준에 머무름.
  진짜 좋아지려면 production 사용자 로그가 누적된 후 다시 학습해야 함.
- `n_works` 의 importance 가 거의 0 — 시드에서 `tbl_follow` 가 `tbl_work` 와 독립적으로 부여돼
  "작품 많은 작가가 팔로워도 많다" 라는 자연 상관이 안 만들어짐. 시드 보강 여지.
- LightGBM / XGBoost 미도입 — sklearn GB 만으로 AUC 0.79. 라이브러리 추가 시 1~3% 더 짤 가능성.

### 배운 점
- **모델은 데이터를 못 이긴다** — 합성 시그널이 약하면 RF 든 GB 든 다 0.5 근처에서 흔들림.
  먼저 데이터부터 분석.
- **누설(Leakage) 은 lift 의 모양으로 드러난다** — `bid_count` lift = ∞ 같은 비현실적 값이 보이면 학습 X.
- **재학습은 모델 작업이 아니라 인프라 작업이기도 하다** — DTO, repository, predictor.featurize_batch,
  Spring 서비스까지 같이 손대야 production 에 진짜로 반영됨.
- **합성 데이터의 distribution 가정은 항상 의심해야** — view 가 1~3000 분포라고 학습한 모델이
  실제로는 평균 23 인 데이터를 만났을 때 학습 가정이 깨짐.

---

🔗 웹 페이지 단위 기능 문서: [README.md](../../../aws/workspace/bideo/README.md)
