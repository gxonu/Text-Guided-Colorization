# -*- coding: utf-8 -*-
"""
본 코드는 세 단계로 나누어 DDColor와 Flux 모델의 추론 및 앙상블을 개별 수행하던 작업을 하나로 통합한 스크립트입니다.

1. DDColor 추론: Colab, A100
2. Flux 추론: Jupyter, A40
3. 앙상블 및 제출 파일 생성: Colab, A100
-> 해당 개별 실행 스크립트는 original 폴더에 보관되어 있습니다.

-> 모든 가중치는 2025-06-23 이전 공개 모델만 사용하여 대회 규정을 준수합니다.
-> 실행 시간 : A40 40 GB 단일 GPU 기준 전체 약 46 분. 
-> 외부 API : 호출 없음. 모든 연산은 로컬에서 수행됩니다.

[제출 폴더 구조]
APXGPu/
├── DDColor/                # DDColor 원본 레포
├── Original/               # Flux, DDColor, Ensemble 개별 스크립트

├── data/
│   ├── test/               # 테스트 이미지
│   └── test.csv            # 이미지 ID 및 caption 정보
├── requirements.txt        # 의존 패키지 목록
└── run_inference.py        # 통합 실행 스크립트

[실행 방법]

1. 'cd APXGPu' 경로 이동
2. 'pip install -r requirements.txt' 실행
3. 'python3 run_inference.py' 명령어 실행

"""
# 0. 기본 모듈 임포트 및 설정
import os
import random
import zipfile
import json
import shutil
import subprocess
import warnings

import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# --- 라이브러리 임포트 (추론 단계별로 필요한 시점에 호출) ---
# 이 스크립트는 단계별로 실행되므로, 필요한 라이브러리를 각 함수 내에서 호출할 수 있으나
# 가독성을 위해 상단에 명시합니다.
from modelscope.hub.snapshot_download import snapshot_download
import open_clip
from diffusers import FluxControlNetPipeline, FluxControlNetModel

warnings.filterwarnings('ignore')

# 1. 환경설정 (Configuration)

# --- 로컬 환경에 맞는 상대 경로 사용 ---
CFG = {
    # --- 입력 경로 ---
    "CSV_TEST": "./data/test.csv",
    "DIR_TEST_IMG": "./data/test/input_image",

    # --- 중간 결과물 저장 경로 (자동 생성) ---
    "DIR_INTERMEDIATE": "./intermediate_results",
    "DIR_DD_IMG_OUT": "./intermediate_results/ddcolor_images",
    "DIR_FX_IMG_OUT": "./intermediate_results/flux_images",

    # --- 최종 결과물 저장 경로 (자동 생성) ---
    "DIR_SUBMISSION": "./submission",
    "DIR_BEST_IMG": "./submission/best_images",

    # --- 모델 및 실행 설정 ---
    "DD_MODEL_CACHE": "./models/modelscope", # ModelScope 모델 캐시 경로
    "FLUX_BASE_MODEL": "black-forest-labs/FLUX.1-dev",
    "FLUX_CONTROLNET": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
    "CLIP_MODEL": "ViT-L-14",
    "CLIP_PRETRAIN": "openai",
    "BATCH_SIZE": 32,
    "SEED": 42,
}

# 2. 헬퍼 함수 (Helper Functions)

def seed_everything(seed):
    """모든 랜덤 시드를 고정하는 함수"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

@torch.no_grad()
def create_image_embeddings(image_dir, df, clip_model, preprocess, device, id_to_path_map):
    """이미지 폴더를 순회하며 CLIP 임베딩을 생성하는 함수"""
    ids, vecs = [], []
    for row in tqdm(df.itertuples(), total=len(df), desc=f"Creating embeddings for {os.path.basename(image_dir)}"):
        img_id = row.ID
        # 이미지 파일명은 ID.png 형식이라고 가정
        img_path = os.path.join(image_dir, f"{img_id}.png")

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            emb = clip_model.encode_image(img_tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
            vecs.append(emb.cpu().numpy().reshape(-1))
            ids.append(img_id)

    cols = [f"vec_{i}" for i in range(vecs[0].shape[0])]
    embed_df = pd.DataFrame(vecs, columns=cols)
    embed_df.insert(0, "ID", ids)
    return embed_df

# 3. 단계별 실행 함수

def run_ddcolor_step(cfg, device):
    """1단계: DDColor 추론 및 임베딩 생성"""
    print("--- Starting Step 1: DDColor Inference ---")

    # 1.1) DDColor 모델 다운로드
    print("Downloading DDColor model from ModelScope...")
    model_path = os.path.join(cfg["DD_MODEL_CACHE"], "damo/cv_ddcolor_image-colorization/pytorch_model.pt")
    if not os.path.exists(model_path):
        snapshot_download('damo/cv_ddcolor_image-colorization', cache_dir=cfg["DD_MODEL_CACHE"])
    else:
        print("DDColor model already downloaded.")

    # 1.2) DDColor 추론 실행
    print("Running DDColor inference on test images...")
    # infer.py는 디렉토리 단위로 작동하므로, subprocess로 실행
    command = [
        "python3", "./DDColor/infer.py",
        "--model_path", model_path,
        "--input", cfg["DIR_TEST_IMG"],
        "--output", cfg["DIR_DD_IMG_OUT"]
    ]
    subprocess.run(command, check=True)
    print("✅ DDColor inference finished.")

    # 1.3) DDColor 결과물 임베딩
    print("Generating CLIP embeddings for DDColor results...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg["CLIP_MODEL"], pretrained=cfg["CLIP_PRETRAIN"])
    clip_model.to(device).eval()
    test_df = pd.read_csv(cfg["CSV_TEST"])
    id_to_path_map = {row.ID: row.input_img_path for row in test_df.itertuples()}

    # DDColor는 파일명을 원본과 동일하게 생성하므로, 파일명 변경이 필요함
    for row in test_df.itertuples():
        original_basename = os.path.basename(row.input_img_path)
        new_name = f"{row.ID}.png"
        try:
            os.rename(os.path.join(cfg["DIR_DD_IMG_OUT"], original_basename.replace('.jpg', '.png')),
                      os.path.join(cfg["DIR_DD_IMG_OUT"], new_name))
        except FileNotFoundError:
            continue
            
    dd_embed_df = create_image_embeddings(cfg["DIR_DD_IMG_OUT"], test_df, clip_model, clip_preprocess, device, id_to_path_map)
    dd_embed_df.to_csv(os.path.join(cfg["DIR_DD_IMG_OUT"], "embed_submission.csv"), index=False)
    print("✅ DDColor embeddings saved.")
    print("--- Step 1: DDColor Finished ---\n")


def run_flux_step(cfg, device):
    """2단계: Flux 추론 및 임베딩 생성"""
    print("--- Starting Step 2: Flux Inference ---")
    
    # 2.1) Flux 모델 로드
    print("Loading Flux ControlNet model...")
    controlnet = FluxControlNetModel.from_pretrained(cfg["FLUX_CONTROLNET"], torch_dtype=torch.bfloat16).to(device)
    pipe = FluxControlNetPipeline.from_pretrained(cfg["FLUX_BASE_MODEL"], controlnet=controlnet, torch_dtype=torch.bfloat16).to(device)
    print("✅ Flux model loaded.")

    # 2.2) Flux 추론 실행
    print("Running Flux inference...")
    test_df = pd.read_csv(cfg["CSV_TEST"])
    out_imgs, out_img_names = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating Flux images"):
        full_path = os.path.join(CFG["DIR_TEST_IMG"], row["input_img_path"])
        input_img = Image.open(full_path).convert("RGB")
        caption = row["caption"]

        output_img = pipe(
            prompt=caption, control_image=input_img, width=512, height=512,
            controlnet_conditioning_scale=0.7, num_inference_steps=30,
            control_guidance_end=1,
            guidance_scale=3.5, generator=torch.Generator(device="cuda").manual_seed(cfg["SEED"]),
        ).images[0]

        # 생성된 이미지를 바로 저장
        output_path = os.path.join(cfg["DIR_FX_IMG_OUT"], f"{row['ID']}.png")
        output_img.save(output_path)
    print("✅ Flux inference finished.")

    # 2.3) Flux 결과물 임베딩
    print("⏳ Generating CLIP embeddings for Flux results...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(cfg["CLIP_MODEL"], pretrained=cfg["CLIP_PRETRAIN"])
    clip_model.to(device).eval()
    id_to_path_map = {row.ID: row.input_img_path for row in test_df.itertuples()}
    
    fx_embed_df = create_image_embeddings(cfg["DIR_FX_IMG_OUT"], test_df, clip_model, clip_preprocess, device, id_to_path_map)
    fx_embed_df.to_csv(os.path.join(cfg["DIR_FX_IMG_OUT"], "embed_submission.csv"), index=False)
    print("✅ Flux embeddings saved.")
    print("--- Step 2: Flux Finished ---\n")


def run_ensemble_step(cfg, device):
    """3단계: 앙상블 및 최종 제출 파일 생성"""
    print("--- Starting Step 3: Ensemble and Final Submission ---")

    # 3.1) CLIP 모델 및 데이터 로드
    print("Loading models and data for ensemble...")
    clip_model, _, _ = open_clip.create_model_and_transforms(cfg["CLIP_MODEL"], pretrained=cfg["CLIP_PRETRAIN"])
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg["CLIP_MODEL"])

    def load_embed_csv(path):
        df = pd.read_csv(path, header=0)
        ids = df.iloc[:, 0].astype(str)
        vec = df.iloc[:, 1:].astype(np.float32).to_numpy()
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        return {k: v for k, v in zip(ids, vec)}

    dd_map = load_embed_csv(os.path.join(cfg["DIR_DD_IMG_OUT"], "embed_submission.csv"))
    fx_map = load_embed_csv(os.path.join(cfg["DIR_FX_IMG_OUT"], "embed_submission.csv"))
    test_df = pd.read_csv(cfg["CSV_TEST"])
    print(f"✅ DD({len(dd_map)}) / Flux({len(fx_map)}) embeddings loaded.")

    # 3.2) CLIP Score 기반 앙상블 실행
    print("Comparing models and selecting the best images...")
    chosen_rows = []
    batch_caps, batch_ids = [], []

    @torch.no_grad()
    def encode_text(texts):
        tok = tokenizer(texts).to(device)
        emb = clip_model.encode_text(tok)
        return emb / emb.norm(dim=-1, keepdim=True)
        
    def flush_batch():
        nonlocal chosen_rows, batch_caps, batch_ids
        if not batch_ids: return
        cap_emb = encode_text(batch_caps)
        dd_emb = torch.tensor(np.stack([dd_map[i] for i in batch_ids]), device=device)
        fx_emb = torch.tensor(np.stack([fx_map[i] for i in batch_ids]), device=device)

        score_dd = (dd_emb * cap_emb).sum(-1)
        score_fx = (fx_emb * cap_emb).sum(-1)

        for idx, img_id in enumerate(batch_ids):
            use_flux = score_fx[idx] >= score_dd[idx]
            src_dir = cfg["DIR_FX_IMG_OUT"] if use_flux else cfg["DIR_DD_IMG_OUT"]
            src_img = os.path.join(src_dir, img_id + ".png")
            dst_img = os.path.join(cfg["DIR_BEST_IMG"], img_id + ".png")
            shutil.copyfile(src_img, dst_img)

            emb_vec = fx_emb[idx] if use_flux else dd_emb[idx]
            chosen_rows.append([img_id, *emb_vec.cpu().numpy()])
        batch_caps.clear(); batch_ids.clear()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="🔍 Selecting best"):
        img_id = str(row["ID"])
        if img_id not in dd_map or img_id not in fx_map:
            print(f"⚠️ Image ID {img_id} missing from results, skipping."); continue
        batch_caps.append(row["caption"])
        batch_ids.append(img_id)
        if len(batch_ids) >= cfg["BATCH_SIZE"]:
            flush_batch()
    flush_batch()
    
    # 3.3) 최종 결과물 저장
    print("Saving final submission files...")
    cols = ["ID"] + [f"vec_{i}" for i in range(768)]
    final_csv_path = os.path.join(cfg["DIR_SUBMISSION"], "embed_submission.csv")
    pd.DataFrame(chosen_rows, columns=cols).to_csv(final_csv_path, index=False)
    
    # 3.4) 최종 제출용 ZIP 파일 생성
    zip_path = os.path.join(cfg["DIR_SUBMISSION"], "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(final_csv_path, arcname="embed_submission.csv")
        for fname in os.listdir(cfg["DIR_BEST_IMG"]):
            fpath = os.path.join(cfg["DIR_BEST_IMG"], fname)
            z.write(fpath, arcname=fname)
            
    print(f"Final submission file created successfully at: {zip_path}")
    print("--- Step 3: Ensemble Finished ---")

# 4. 메인 실행 블록 (Main Execution Block)
if __name__ == '__main__':
    # 0. 준비 단계
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 필요한 모든 디렉토리 생성
    for path in [
        CFG["DIR_INTERMEDIATE"], CFG["DIR_DD_IMG_OUT"], CFG["DIR_FX_IMG_OUT"],
        CFG["DIR_SUBMISSION"], CFG["DIR_BEST_IMG"], CFG["DD_MODEL_CACHE"]
    ]:
        os.makedirs(path, exist_ok=True)
        
    # 1. DDColor 단계 실행
    run_ddcolor_step(CFG, DEVICE)

    # 2. Flux 단계 실행
    seed_everything(CFG['SEED'])
    run_flux_step(CFG, DEVICE)
    
    # 3. 앙상블 및 제출 단계 실행
    run_ensemble_step(CFG, DEVICE)
    
    print("\n✅ All processes completed successfully.")