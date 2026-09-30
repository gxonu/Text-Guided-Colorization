# 실행 가이드

[프로젝트 소개로 돌아가기](../README.md)

## 재현 범위

이 저장소에는 통합 추론 코드, 단계별 실행 기록, 입력 CSV와 이미지가 있습니다. **DDColor 소스와 모델 가중치는 별도로 준비해야 합니다.** 통합 추론을 새 환경에서 재실행해 검증한 설치 조합은 아직 없으며, 아래는 코드가 요구하는 준비 절차입니다.

`requirements.txt`의 패키지 대부분은 버전이 고정되어 있지 않습니다. FLUX 추론은 BF16과 CUDA generator를 사용하므로 호환되는 NVIDIA GPU와 PyTorch 환경이 필요합니다. CPU 실행은 검증하지 않았습니다.

## 1. 저장소와 환경 준비

```bash
git clone https://github.com/gxonu/Text-Guided-Colorization.git
cd Text-Guided-Colorization/APXGPu

python3 -m venv .venv
source .venv/bin/activate

# CUDA 환경에 맞는 PyTorch를 준비한 뒤 프로젝트 의존성을 설치합니다.
pip install -r requirements.txt
```

## 2. DDColor 소스 준비

통합 코드는 `APXGPu/DDColor/infer.py`를 호출합니다. 현재 저장소에는 이 폴더가 포함되어 있지 않습니다.

```bash
# 작업 디렉터리: APXGPu/
git clone https://github.com/piddnad/DDColor.git DDColor
```

DDColor 자체의 의존성도 [원본 설치 안내](https://github.com/piddnad/DDColor#installation)에 따라 준비해야 합니다. 로컬에 보관된 DDColor 체크아웃의 commit은 `3348cee1e5e9c4cf0d1c6e6e81cef45ed9a4f0f5`입니다. 대회 당시 사용 버전과 같은지까지 확인된 것은 아닙니다.

## 3. 모델 준비

| 모델 | 위치 또는 식별자 |
|---|---|
| DDColor | ModelScope `damo/cv_ddcolor_image-colorization` |
| FLUX | `black-forest-labs/FLUX.1-dev` |
| ControlNet | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` |
| CLIP | OpenCLIP `ViT-L-14`, pretrained `openai` |

FLUX 모델 접근에 필요한 계정 설정과 사용 조건 동의를 완료해야 합니다. 코드에 토큰을 넣을 필요는 없습니다. 모델이 캐시에 없으면 첫 실행에서 다운로드하므로 네트워크와 저장 공간이 필요합니다. 추론 자체는 로컬 모델로 수행합니다.

## 4. 입력 확인

기본 구조는 다음과 같습니다.

```text
APXGPu/
├── DDColor/infer.py
├── data/
│   ├── test.csv
│   └── test/input_image/
│       ├── TEST_001.png
│       └── ...
├── requirements.txt
└── run_inference.py
```

CSV는 `ID`, `input_img_path`, `caption` 열을 사용합니다. `input_img_path`는 입력 이미지 폴더를 기준으로 한 파일명입니다.

```csv
ID,input_img_path,caption
TEST_001,TEST_001.png,Description of the scene and its colors
```

위 행은 형식을 설명하는 예시입니다. 기본 CSV의 실제 설명문을 그대로 사용하면 됩니다. 데이터 경로를 바꾸려면 `run_inference.py` 상단의 `CFG`를 수정합니다.

## 5. 추론 및 결과 확인

```bash
# 작업 디렉터리: APXGPu/
python3 run_inference.py
```

```text
intermediate_results/
├── ddcolor_images/       DDColor 이미지와 임베딩 CSV
└── flux_images/          FLUX 이미지와 임베딩 CSV
submission/
├── best_images/          CLIP으로 선택한 이미지
├── embed_submission.csv ID와 768차원 이미지 임베딩
└── submission.zip       임베딩 CSV와 선택 이미지
```

후보 중 어느 한쪽에 이미지가 없으면 현재 코드는 해당 ID를 건너뜁니다. 제출 전에는 입력 CSV의 ID 수와 최종 이미지 및 CSV 행 수가 일치하는지 확인해야 합니다. 재실행할 때는 이전 결과가 섞이지 않도록 새로운 출력 폴더를 지정하는 편이 좋습니다.

기존 README의 실행 시간은 과거 기록입니다. 이 문서 작성 과정에서는 GPU 추론이나 실행 시간 재측정을 수행하지 않았습니다.
