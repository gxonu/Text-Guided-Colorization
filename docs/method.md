# 구현 설명

[프로젝트 소개로 돌아가기](../README.md)

## 입력에서 출력까지

| 단계 | 입력 | 출력 | 구현 |
|---|---|---|---|
| DDColor | 흑백 이미지 | 컬러 후보 A | `run_ddcolor_step()` |
| FLUX + ControlNet | 흑백 이미지, 설명문 | 컬러 후보 B | `run_flux_step()` |
| CLIP 선택 | 두 후보, 설명문 | 선택 이미지와 임베딩 | `run_ensemble_step()` |

위 함수들은 모두 [`APXGPu/run_inference.py`](../APXGPu/run_inference.py)에 있습니다. 처리 순서는 DDColor → FLUX → 선택입니다. 개념도에서 두 분기는 후보를 만드는 두 경로를 뜻하며, 실제 코드는 순차 실행합니다.

## 두 생성 경로

DDColor는 이미지에서 색을 복원합니다. 별도로 준비하는 DDColor 구현은 원본의 Lab 밝기 채널 L을 유지하고 예측한 색 채널 ab와 결합합니다. 설명문은 사용하지 않습니다.

FLUX는 설명문을 `prompt`, 입력 이미지를 `control_image`로 전달합니다. 사용 모델은 `black-forest-labs/FLUX.1-dev`와 `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`입니다.

| FLUX 설정 | 코드 값 |
|---|---|
| 생성 크기 | 512 × 512 |
| 추론 단계 | 30 |
| Guidance scale | 3.5 |
| ControlNet conditioning scale | 0.7 |
| Control guidance end | 1 |
| 이미지별 seed | 42 |

## CLIP 선택

`ViT-L-14`, `pretrained="openai"`를 사용합니다. 각 후보 이미지의 임베딩과 설명문 임베딩을 L2 정규화한 뒤 내적합니다. 정규화된 벡터의 내적은 코사인 유사도와 같습니다.

```python
score_dd = (dd_emb * cap_emb).sum(-1)
score_fx = (fx_emb * cap_emb).sum(-1)
use_flux = score_fx >= score_dd
```

이미지마다 선택 결과가 달라질 수 있습니다. FLUX가 선택되면 FLUX 이미지와 임베딩을, 그렇지 않으면 DDColor 이미지와 임베딩을 저장합니다. 임베딩이나 픽셀을 평균하지 않습니다.

## 해석할 때 알아둘 점

- CLIP은 전체 이미지와 설명문의 정합성을 비교합니다. 물체마다 지정한 색이 정확히 반영됐는지 별도로 검증하지는 않습니다.
- 입력 이미지의 원래 색을 유일하게 복원하는 문제가 아니라, 설명문을 반영한 컬러 결과를 만드는 문제입니다.
- 현재 공개된 통합 코드의 최종 선택 기준은 CLIP 유사도입니다. HSV 보정이나 새로 학습한 DiT는 이 진입점에 포함되어 있지 않습니다.
- 후보별 성능 차이와 선택 비율을 보여주는 비교표는 실제 실험 결과가 있어야 작성할 수 있습니다. 현재 문서에는 해당 수치를 추정해 넣지 않았습니다.
