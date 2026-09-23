# World's Hardest Game RL

`The World's Hardest Game`을 Gymnasium 환경으로 재구성하고, 픽셀 입력 기반 PPO 에이전트를 학습하기 위한 프로젝트입니다.

현재 학습 파이프라인은 다음 구성을 사용합니다.

- **Spatial path-progress reward shaping**
- **`legacy_exact` 픽셀 렌더링**
- **Cached NOOP reset**을 이용한 시작 phase 랜덤화
- **Spatial oracle physics fast-path**
- **Frame stacking + frame skip**
- 다중 환경 학습 시 **shared-memory observation IPC**
- Stable-Baselines3 **PPO**

기본 학습 대상은 **Level 3**이며, 필요하면 Level 1~30을 선택할 수 있습니다.

---

## 프로젝트 구조

```text
.
├── train.py                 # PPO 학습
├── evaluate.py              # 학습 모델 평가
├── watch.py                 # 학습된 모델을 화면으로 재생
├── play.py                  # 사람이 직접 플레이
├── whg_models.py            # PPO용 커스텀 CNN
├── requirements.txt
├── pyproject.toml
└── whg_swf_gym/
    ├── __init__.py
    ├── core.py              # 게임 로직 / 물리
    ├── env.py               # Gymnasium 환경
    ├── game_data.py         # SWF에서 추출한 레벨 데이터
    ├── navigation.py        # Spatial oracle 생성 및 탐색
    ├── pixel_rl.py          # Spatial reward shaping
    ├── render.py            # 픽셀 렌더러
    ├── swf.py               # SWF 파서
    └── assets/
        └── WHGOriginal.swf
```

---

## 요구 사항

- Python **3.10 이상**
- Windows / Linux / macOS
- GPU는 선택 사항
  - CUDA 환경이 있으면 PyTorch가 GPU를 사용할 수 있습니다.
  - CPU만으로도 실행할 수 있습니다.
- 다음 위치에 대상 SWF 파일이 있어야 합니다.

```text
whg_swf_gym/assets/WHGOriginal.swf
```

환경은 프로젝트가 대상으로 하는 SWF와 SHA-1이 일치하는지 확인합니다. 다른 버전의 SWF를 넣으면 실행 중 오류가 발생합니다.

> 공개 GitHub 저장소에 원본 게임 asset을 포함하려면 해당 파일을 재배포할 권리가 있는지 별도로 확인하세요.

---

# 설치

## 1. 저장소 받기

```bash
git clone https://github.com/YOUR_USERNAME/WorldsHardestGameRL.git
cd WorldsHardestGameRL
```

GitHub에 올리기 전이라면 프로젝트 폴더로 직접 이동하면 됩니다.

---

## 2. 가상환경 만들기

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. 패키지 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 먼저 게임이 정상적으로 실행되는지 확인하기

머신러닝 학습 전에 reconstructed environment가 정상적으로 동작하는지 확인하는 것이 좋습니다.

Level 3을 직접 플레이하려면:

```bash
python play.py --level 3
```

전체 30레벨 campaign으로 실행하려면:

```bash
python play.py
```

### 조작법

| 키 | 동작 |
| --- | --- |
| 방향키 / WASD | 이동 |
| `R` | 재시작 |
| `[` | 이전 레벨 |
| `]` | 다음 레벨 |
| `Esc` | 종료 |

---

# PPO 학습

기본 설정으로 학습하려면:

```bash
python train.py
```

현재 기본값은 다음과 같습니다.

```text
level             = 3
steps             = 6,000,000
parallel envs     = 128
observation       = 110 x 80 RGB
frame stack       = 3
frame skip        = 2
start delay       = 0..100 Flash frames
pixel pipeline    = legacy_exact
reward shaping    = spatial
learning rate     = 1e-4
PPO n_steps       = 128 / environment
batch size        = 2048
gamma             = 0.99
GAE lambda        = 0.9
entropy coef      = 0.03
```

기본 `128`개 병렬 환경은 CPU/RAM 사용량이 큰 설정입니다.

처음 테스트하거나 PC 사양이 충분하지 않다면 다음처럼 환경 수를 줄이는 것을 권장합니다.

```bash
python train.py --n-envs 16
```

또는:

```bash
python train.py --n-envs 32
```

---

## 짧게 학습 테스트하기

설정이 정상인지 확인할 때는 처음부터 수백만 step을 돌릴 필요가 없습니다.

```bash
python train.py \
  --steps 500000 \
  --n-envs 16 \
  --out runs/test
```

Windows CMD에서는 한 줄로 실행하면 됩니다.

```bat
python train.py --steps 500000 --n-envs 16 --out runs/test
```

---

## 다른 레벨 학습하기

```bash
python train.py --level 5 --out runs/level05
```

---

## GPU / CPU 지정

GPU 사용:

```bash
python train.py --device cuda
```

CPU 사용:

```bash
python train.py --device cpu
```

기본값은:

```text
--device auto
```

입니다.

---

# Cached reset과 Spatial oracle

실행 시 두 종류의 cache가 자동으로 생성됩니다.

## Reset cache

예:

```text
reset_cache_level03_legacy_exact_110x80_box/
```

랜덤 NOOP pre-roll 상태와 필요한 RGB frame을 미리 저장합니다.

따라서 매 episode reset마다 0~100 frame을 다시 시뮬레이션하고 렌더링할 필요가 없습니다.

---

## Spatial oracle

예:

```text
spatial_oracle_level03.npz
```

Spatial reward shaping과 wall physics fast-path에서 사용됩니다.

정상적인 경우 기존 cache는 자동으로 재사용됩니다.

### 환경 코드를 수정했다면

다음과 같은 부분을 변경한 경우 cache를 다시 만드는 것이 안전합니다.

- player movement
- wall collision
- navigation / spatial oracle
- rendering 크기나 방식
- reset과 관련된 core state

두 cache를 모두 다시 만들려면:

```bash
python train.py --rebuild-reset-cache --rebuild-spatial-cache
```

환경 내부를 수정한 뒤 결과가 이상하다면 우선 cache부터 rebuild하는 것을 권장합니다.

---

# 학습 결과

기본 output directory는:

```text
runs/whg_pixels_l3/
```

입니다.

대략 다음과 같은 파일들이 생성됩니다.

```text
runs/whg_pixels_l3/
├── best_model.zip
├── final_model.zip
├── config.json
├── checkpoints/
├── tb/
├── reset_cache_level03_legacy_exact_110x80_box/
└── spatial_oracle_level03.npz
```

### `best_model.zip`

주기적인 deterministic evaluation 결과가 가장 좋았던 모델입니다.

### `final_model.zip`

학습이 끝난 시점의 모델입니다.

PPO는 학습 후반에 성능이 다시 떨어질 수도 있으므로, 항상 `final_model.zip`이 가장 좋은 모델이라고 볼 수는 없습니다.

실제로 사용할 모델은 `best_model.zip`도 함께 비교하는 것이 좋습니다.

### `checkpoints/`

학습 중간 checkpoint가 저장됩니다.

### `config.json`

해당 run에 사용한 command-line 설정이 기록됩니다.

### `tb/`

TensorBoard log가 저장됩니다.

---

# TensorBoard 확인

학습 중 PPO 상태와 환경 통계를 확인하려면:

```bash
tensorboard --logdir runs/whg_pixels_l3/tb
```

일반적으로 브라우저에서 다음 주소를 열면 됩니다.

```text
http://localhost:6006
```

PPO가 갑자기 불안정해지는 경우 다음 지표를 함께 확인하는 것이 좋습니다.

```text
train/approx_kl
train/clip_fraction
train/entropy_loss
train/explained_variance
```

환경 쪽 success/progress metric도 함께 확인할 수 있습니다.

---

# 학습 이어서 하기

기존 PPO 모델에서 이어서 학습하려면:

```bash
python train.py \
  --resume runs/whg_pixels_l3/best_model.zip \
  --out runs/whg_pixels_l3
```

Windows CMD:

```bat
python train.py --resume runs/whg_pixels_l3/best_model.zip --out runs/whg_pixels_l3
```

저장된 모델의 입력 구조와 호환되어야 하므로 최소한 다음 값은 기존 모델과 동일하게 유지해야 합니다.

```text
--width
--height
--frame-stack
```

---

# 모델 평가

학습된 PPO 모델을 화면 출력 없이 평가하려면:

```bash
python evaluate.py runs/whg_pixels_l3/best_model.zip
```

기본적으로 Level 3에서 100 episode를 deterministic policy로 평가합니다.

500 episode 평가:

```bash
python evaluate.py \
  runs/whg_pixels_l3/best_model.zip \
  --episodes 500
```

출력되는 주요 값은 다음과 같습니다.

- Success rate
- Mean reward
- Mean policy decisions
- Mean Flash frames
- Mean randomized start delay

다른 레벨 모델이라면 반드시 level을 맞춰주세요.

```bash
python evaluate.py runs/level05/best_model.zip --level 5
```

모델을 기본값과 다른 observation 설정으로 학습했다면 평가 시에도 동일한 값을 사용해야 합니다.

예:

```bash
python evaluate.py model.zip \
  --frame-stack 4 \
  --frame-skip 2 \
  --width 110 \
  --height 80
```

> 현재 `evaluate.py`의 reward coefficient는 `train.py` 기본 reward 값에 맞춰져 있습니다. 학습 시 reward coefficient를 직접 변경했다면 success 여부와 policy 동작은 그대로 평가할 수 있지만, 출력되는 episode reward를 정확히 비교하려면 동일한 reward 설정을 사용해야 합니다.

---

# 학습된 모델 화면으로 보기

PPO가 실제로 어떻게 플레이하는지 보려면:

```bash
python watch.py runs/whg_pixels_l3/best_model.zip
```

기본값은 deterministic policy입니다.

stochastic action을 사용하려면:

```bash
python watch.py \
  runs/whg_pixels_l3/best_model.zip \
  --stochastic
```

episode 수 변경:

```bash
python watch.py \
  runs/whg_pixels_l3/best_model.zip \
  --episodes 50
```

재생 속도 변경:

```bash
python watch.py \
  runs/whg_pixels_l3/best_model.zip \
  --fps 30
```

종료하려면 `Esc`를 누르거나 창을 닫으면 됩니다.

`watch.py` 역시 학습과 같은 environment construction을 사용하므로, 모델을 다른 frame stack / resolution으로 학습했다면 동일한 값을 지정해야 합니다.

---

# 현재 환경의 중요한 특징

## Spatial reward shaping만 사용

기존 temporal reward path는 제거되었습니다.

현재 학습은 항상 `PathProgressReward` 기반 spatial shaping을 사용합니다.

---

## `legacy_exact` 렌더링만 사용

과거 wrapper 기반 resize path는 제거되었습니다.

학습 / 평가 / 모델 시청 모두 동일한 `legacy_exact` 픽셀 경로를 사용합니다.

---

## Cached reset만 사용

랜덤 시작 phase는 매 episode마다 NOOP을 다시 실행하는 대신 미리 계산된 reset cache에서 복원됩니다.

---

## Spatial oracle physics fast-path

Spatial oracle에 존재하는 상태에서는:

```text
(position, action) -> next position
```

transition lookup을 이용해 wall movement를 처리합니다.

Oracle에 없는 상태는 기존 physics path로 fallback할 수 있습니다.

---

## Shared-memory observations

두 개 이상의 training environment를 사용할 때 stacked RGB observation은 multiprocessing Pipe를 통해 매번 pickle하지 않고 shared memory를 이용할 수 있습니다.

기본적으로 활성화되어 있습니다.

A/B 테스트나 디버깅을 위해 끄려면:

```bash
python train.py --no-shared-observations
```

---

# 자주 발생하는 문제

## `ModuleNotFoundError`

가상환경이 활성화되어 있는지 확인한 뒤:

```bash
pip install -r requirements.txt
```

를 다시 실행하세요.

스크립트는 가능하면 프로젝트 루트에서 실행하세요.

---

## SWF hash mismatch

다음 위치의 파일을 확인하세요.

```text
whg_swf_gym/assets/WHGOriginal.swf
```

이 프로젝트는 특정 SWF build를 기준으로 재구성되어 있습니다.

---

## Model observation shape mismatch

학습된 CNN은 학습 당시 observation shape와 동일한 입력이 필요합니다.

기본 설정에서는 RGB frame 3장을 stack하므로 PPO에 들어가는 shape은:

```text
9 x 80 x 110
```

입니다.

`width`, `height`, `frame-stack`을 변경해서 학습했다면 evaluate/watch에서도 반드시 동일하게 지정하세요.

---

## 학습이 너무 무겁거나 RAM 사용량이 큼

가장 먼저 parallel environment 수를 줄여보세요.

```bash
python train.py --n-envs 16
```

짧은 테스트라면 total steps도 줄일 수 있습니다.

```bash
python train.py --steps 500000
```

---

## 환경 코드를 바꾼 뒤 결과가 이상함

기존 cache가 남아 있다면 우선 다시 생성하세요.

```bash
python train.py --rebuild-reset-cache --rebuild-spatial-cache
```

---

# 전체 CLI 확인

각 스크립트의 최신 옵션은 `--help`로 확인할 수 있습니다.

```bash
python train.py --help
python evaluate.py --help
python watch.py --help
python play.py --help
```

---

## Notes

이 프로젝트는 SWF에서 게임 동작을 재구성하여 강화학습 환경과 PPO 실험에 사용하는 것을 목적으로 합니다.

원작 게임 제작자 또는 배급사와 공식적으로 연관된 프로젝트가 아닙니다.
