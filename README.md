# World's Hardest Game RL

`The World's Hardest Game`을 Gymnasium 환경으로 재구성하고, 픽셀 입력 기반 PPO 에이전트를 학습하기 위한 프로젝트입니다.

The World's Hardest Game의 원본 SWF 파일은 아래 Speedrun.com 리소스 페이지에서 제공되는 파일을 사용했습니다.

https://www.speedrun.com/whg1/resources/mq46b



## 프로젝트 구조

```text
.
├── train.py                 # PPO 학습
├── evaluate.py              # 학습 모델 평가
├── watch.py                 # 학습된 모델을 화면으로 재생
├── play.py                  # 사람이 직접 플레이
├── whg_models.py            # PPO용 커스텀 CNN
├── requirements.txt
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


## 요구 사항

- Python 3.10 이상 (3.11 권장)
- Windows / Linux / macOS
- GPU는 선택 사항
  - CUDA 환경이 있으면 PyTorch가 GPU를 사용할 수 있습니다.
  - CPU만으로도 실행할 수 있습니다.



# 설치

## 1. 저장소 받기

```bash
git clone https://github.com/great0108/WorldsHardestGameRL.git
cd WorldsHardestGameRL
```



## 2. 가상환경 만들기

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```



## 3. 패키지 설치

먼저 `pip`을 업데이트합니다.

```bash
python -m pip install --upgrade pip
```

### CPU로 사용할 경우

그대로 프로젝트 의존성을 설치하면 됩니다.

```bash
pip install -r requirements.txt
```

### NVIDIA GPU로 학습할 경우

CUDA를 사용할 예정이라면 **프로젝트 의존성을 설치하기 전에 CUDA 지원 PyTorch를 먼저 설치하는 것을 권장합니다.**

PyTorch 공식 설치 페이지에서 자신의 OS와 GPU 환경에 맞게 다음 항목을 선택합니다.

- OS: Windows 또는 Linux
- Package: Pip
- Language: Python
- Compute Platform: 지원되는 CUDA 버전

공식 설치 페이지:

<https://pytorch.org/get-started/locally/>

페이지에 표시되는 명령을 실행한 뒤 나머지 프로젝트 의존성을 설치합니다. 예시는 다음과 같은 형태입니다.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
pip install -r requirements.txt
```

설치 후 PyTorch가 GPU를 인식하는지 확인합니다.

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

정상적인 예시는 다음과 같습니다.

```text
CUDA available: True
PyTorch CUDA: 12.x
GPU: NVIDIA ...
```

`CUDA available: False`가 나온다면 CPU 전용 PyTorch가 설치되었거나 NVIDIA 드라이버/CUDA 호환 문제가 있는 것입니다. 이 경우 PyTorch 공식 설치 페이지에서 CUDA 빌드 설치 명령을 다시 확인하세요.

이미 CPU 버전의 PyTorch가 설치된 상태라면 제거한 뒤 CUDA 빌드를 다시 설치할 수 있습니다.

```bash
pip uninstall torch torchvision torchaudio -y
```


# 게임 환경

학습 전에 환경이 정상적으로 동작하는지 확인하는 것이 좋습니다.

Level 1을 직접 플레이하려면:

```bash
python play.py --level 1
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



# PPO 학습

Level 1을 기본 설정으로 학습하려면:

```bash
python train.py --level 1 --out runs/level01
```

현재 기본값은 다음과 같습니다.

```text
steps             = 6,000,000
parallel envs     = 128
observation       = 110 x 80 RGB
frame stack       = 3
frame skip        = 2
start delay       = 0..100 Flash frames
learning rate     = 1e-4
PPO n_steps       = 128
batch size        = 2048
gamma             = 0.99
GAE lambda        = 0.9
entropy coef      = 0.03
```


# 학습 결과

학습 후에 대략 다음과 같은 파일들이 생성됩니다.

```text
runs/level01/
├── best_model.zip
├── final_model.zip
├── config.json
├── checkpoints/
├── tb/
├── reset_cache_level01_legacy_exact_110x80_box/
└── spatial_oracle_level01.npz
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



# TensorBoard 확인

학습 중 PPO 상태와 환경 통계를 확인하려면:

```bash
tensorboard --logdir runs/level01/tb
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



# 학습 이어서 하기

기존 PPO 모델에서 이어서 학습하려면:

```bash
python train.py \
  --resume runs/level01/best_model.zip \
  --out runs/level01
```



# 모델 평가

학습된 PPO 모델을 화면 출력 없이 평가하려면:

```bash
python evaluate.py runs/level01/best_model.zip --level 1
```

기본적으로 100 episode를 deterministic policy로 평가합니다.

출력되는 주요 값은 다음과 같습니다.

- Success rate
- Mean reward
- Mean policy decisions
- Mean Flash frames
- Mean randomized start delay



# 학습된 모델 화면으로 보기

PPO가 실제로 어떻게 플레이하는지 보려면:

```bash
python watch.py runs/level01/best_model.zip --level 1
```

기본값은 deterministic policy입니다.

종료하려면 `Esc`를 누르거나 창을 닫으면 됩니다.



# 전체 CLI 확인

각 스크립트의 옵션은 `--help`로 확인할 수 있습니다.

```bash
python train.py --help
python evaluate.py --help
python watch.py --help
python play.py --help
```

# NGU-lite 탐험 보너스 (선택 기능)

`train.py`에는 기존 extrinsic reward를 그대로 둔 채 학습 중에만 episodic novelty를 더하는 NGU-inspired 옵션이 있습니다.

구성은 다음과 같습니다.

- inverse-dynamics encoder: 연속 관측에서 어떤 action이 실행됐는지 예측하여 플레이어가 통제할 수 있는 시각 변화에 민감한 embedding을 학습
- target encoder: embedding 공간이 너무 빨리 움직이지 않도록 EMA로 천천히 갱신
- episodic k-NN memory: 각 vector environment가 이번 episode에서 방문한 embedding을 저장
- intrinsic reward: memory에서 멀리 떨어진 새로운 상태일수록 큰 `[0, 1]` novelty를 부여
- death/fade frame: 플레이어가 통제할 수 없는 화면 변화이므로 intrinsic reward와 inverse-dynamics 학습에서 제외

평가 환경에는 intrinsic reward가 들어가지 않습니다. 또한 NGU wrapper를 `VecMonitor` 바깥에 두기 때문에 TensorBoard의 기존 episode reward는 extrinsic reward 기준으로 유지됩니다.

레벨 6에서 시작해볼 예시는 다음과 같습니다.

```bash
python train.py \
  --level 6 \
  --intrinsic \
  --no-death-ends-episode \
  --intrinsic-beta 0.01 \
  --intrinsic-memory-size 512 \
  --intrinsic-k 10 \
  --frame-stack 3 \
  --frame-skip 1 \
  --delay-min 0 \
  --delay-max 83 \
  --eval-delay-min 0 \
  --eval-delay-max 83 \
  --out runs/level06_ngu
```

`--no-death-ends-episode`가 중요한 이유는 죽은 뒤 원래 게임의 checkpoint respawn을 계속 진행하면서 같은 episodic memory를 유지하기 위해서입니다. 그러면 이미 여러 번 시도한 직진 경로의 novelty는 낮아지고, 아직 가보지 않은 옆/우회 공간을 탐색할 동기가 생깁니다.

추가 TensorBoard 지표:

```text
intrinsic/novelty_mean
intrinsic/scaled_reward_mean
intrinsic/inverse_loss_mean
```

`intrinsic/novelty_mean`이 오랫동안 거의 `1.0`이면 embedding이 상태를 너무 쉽게 전부 다르게 보고 있을 가능성이 있습니다. 반대로 너무 빨리 `0`에 붙으면 `--intrinsic-beta`를 올리기 전에 embedding/inverse loss가 정상적으로 학습되는지 먼저 확인하는 것이 좋습니다.

NGU-lite 보조 모델은 `final_intrinsic.pt`에 저장되며, 100만 timestep마다 `checkpoints/ngu_*_steps.pt`도 저장됩니다. PPO를 이어서 학습할 때는 대응되는 보조 state를 함께 지정할 수 있습니다.

```bash
python train.py \
  --level 6 \
  --intrinsic \
  --resume runs/level06_ngu/checkpoints/level06_pixels_1000000_steps.zip \
  --intrinsic-state runs/level06_ngu/checkpoints/ngu_1000000_steps.pt \
  --out runs/level06_ngu
```
