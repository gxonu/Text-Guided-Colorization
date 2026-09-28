# Text-Guided Colorization

흑백 이미지와 텍스트 설명을 받아 설명에 맞는 컬러 이미지를 생성하는 프로젝트입니다.
**2025 Text-Guided Colorization AI 챌린지 학부생 부문 우수상(BK21 교육연구단장상)**을 받은 최종 제출 방법과 추론 코드를 정리했습니다.

김건우는 팀 리더로 문제 정의, 모델 구성, 결과 비교, 앙상블과 최종 제출까지 전 과정을 주도했습니다.

## 최종 제출 방법

1. **DDColor**로 이미지의 컬러 복원 후보를 생성합니다.
2. **FLUX.1-dev와 ControlNet**으로 입력 이미지와 텍스트 설명을 반영한 후보를 생성합니다.
3. **CLIP ViT-L/14**로 두 후보와 텍스트 설명의 유사도를 비교합니다.
4. 이미지마다 유사도가 높은 후보를 선택하고 제출 이미지 및 임베딩 파일을 만듭니다.

현재 공개된 최종 파이프라인은 **DDColor + FLUX 후보 생성과 CLIP 기반 이미지별 선택**입니다.

## 코드와 자료

| 자료 | 경로 |
|---|---|
| 통합 추론 및 제출 파일 생성 | [APXGPu/run_inference.py](APXGPu/run_inference.py) |
| 기존 실행 안내 | [APXGPu/README.md](APXGPu/README.md) |
| DDColor 단계별 노트북 | [APXGPu/Original/DDColor.ipynb](APXGPu/Original/DDColor.ipynb) |
| FLUX 추론 | [APXGPu/Original/flux.py](APXGPu/Original/flux.py) |
| 앙상블 노트북 | [APXGPu/Original/Ensemble.ipynb](APXGPu/Original/Ensemble.ipynb) |
| 의존 패키지 | [APXGPu/requirements.txt](APXGPu/requirements.txt) |

## 실행 준비

진입점은 `APXGPu/run_inference.py`이며 작업 디렉터리는 `APXGPu/`입니다.

```bash
cd APXGPu
pip install -r requirements.txt
python3 run_inference.py
```

실행 전 `DDColor/infer.py`를 포함한 DDColor 소스, 입력 이미지와 CSV, 사전학습 모델의 접근 권한 및 실행 환경이 필요합니다.
모델 파일이 캐시에 없으면 코드에서 ModelScope와 Hugging Face를 통해 내려받습니다.
최종 출력 경로는 `submission/best_images/`, `submission/embed_submission.csv`와 `submission/submission.zip`입니다.

이 README 작성 과정에서는 전체 추론을 다시 실행하지 않았습니다.
