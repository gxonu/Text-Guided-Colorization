# APXGPu: Text-Guided Image Colorization (DDColor + Flux Ensemble)

이 저장소는 **DDColor**와 **Flux** 모델의 추론 및 **앙상블
파이프라인**을 하나로 통합한 프로젝트입니다.\
기존에는 Colab 및 Jupyter 환경에서 단계별로 나누어 실행하던 작업을
**하나의 스크립트(run_inference.py)**로 자동화하였습니다.

## 🚀 파이프라인 구성

### 1. DDColor 추론

-   환경: Google Colab\
-   GPU: A100

### 2. Flux 추론

-   환경: Jupyter Notebook\
-   GPU: A40

### 3. 앙상블 및 제출 파일 생성

-   환경: Google Colab\
-   GPU: A100

> 개별 실행 스크립트는 모두 `Original/` 디렉토리에 포함되어 있습니다.

## 🧩 규정 준수

-   모든 가중치는 **2025-06-23 이전 공개 모델만 사용**\
-   **외부 API 호출 없음**\
-   **모든 연산은 로컬 및 로컬 GPU 환경에서 직접 수행**

## ⚡ 전체 실행 시간

-   A40 40GB 기준 약 **46분**

## 📁 폴더 구조

    APXGPu/
    ├── DDColor/                # DDColor 원본 레포
    ├── Original/               # DDColor, Flux, Ensemble 개별 스크립트
    │
    ├── data/
    │   ├── test/               # 테스트 이미지
    │   └── test.csv            # 이미지 ID + caption
    │
    ├── requirements.txt        # 의존성 리스트
    └── run_inference.py        # 전체 파이프라인 통합 실행 스크립트

## ▶ 실행 방법

### 1) 이동

``` bash
cd APXGPu
```

### 2) 패키지 설치

``` bash
pip install -r requirements.txt
```

### 3) 전체 파이프라인 실행

``` bash
python3 run_inference.py
```

## 🏁 결과물

실행이 완료되면:

-   모델별 추론 결과\
-   앙상블 최종 산출물\
-   제출용 파일

이 자동으로 생성됩니다.
