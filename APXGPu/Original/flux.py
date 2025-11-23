from tqdm import tqdm
import random
import os
import zipfile
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
import open_clip
import warnings
warnings.filterwarnings('ignore')

from diffusers import FluxControlNetPipeline, FluxControlNetModel
from transformers import pipeline
import re
import time
import PIL

start_time = time.time()
base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0'

def preprocess_caption(caption: str) -> str:
    # 1. 질문 형식의 문장 제거
    # '?'로 끝나거나 특정 질문 단어로 시작하는 문장 제거
    sentences = re.split(r'(?<=[.!?])\s+', caption.strip()) # 마침표, 물음표, 느낌표 기준으로 문장 분리

    filtered_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if s.endswith('?') or \
           s_lower.startswith('what is') or \
           s_lower.startswith('do you see') or \
           s_lower.startswith('are there') or \
           s_lower.startswith('which piece') or \
           s_lower.startswith('is the'):
            continue # 질문 문장은 건너뛰기

        # 2. 불필요한 시작 구문 제거 (단, 문장 시작 부분에만 적용)
        if s_lower.startswith('in this image i can see'):
            s = s[len('in this image i can see'):].strip()
            # 첫 글자가 소문자면 대문자로 변경 (문법적으로)
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
        elif s_lower.startswith('i can see'): # 다른 유사한 불필요 구문이 있다면 추가
            s = s[len('i can see'):].strip()
            if s and s[0].islower():
                s = s[0].upper() + s[1:]

        filtered_sentences.append(s.strip())

    return " ".join(filtered_sentences).strip()

def apply_color(image, color_map):
    # Convert input images to LAB color space
    image_lab = image.convert('LAB')
    color_map_lab = color_map.convert('LAB')

    # Split LAB channels
    l, a, b = image_lab.split()
    _, a_map, b_map = color_map_lab.split()

    # Merge LAB channels with color map
    merged_lab = PIL.Image.merge('LAB', (l, a_map, b_map))

    # Convert merged LAB image back to RGB color space
    result_rgb = merged_lab.convert('RGB')
    
    return result_rgb



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CFG = {
    'SUB_DIR' : './submission',
    'SEED' : 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED']) # Seed 고정

controlnet = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16).to(device)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16).to(device)

    
test_df = pd.read_csv('./test.csv')

out_imgs = []
out_img_names = []
num = 0
for img_id, img_path, caption in zip(test_df['ID'], test_df['input_img_path'], test_df['caption']):
    input_img  = Image.open(img_path).convert("RGB")

    #control_image = preprocess_for_controlnet(input_img, detector_type="canny")
    #caption = preprocess_caption(caption)

    full_prompt = f"{caption}"
    output_img = pipe(
        full_prompt, 
        control_image=input_img,
        width=512,
        height=512,
        controlnet_conditioning_scale=0.7,
        control_guidance_end=1,
        num_inference_steps=28, 
        guidance_scale=3.5,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    x = apply_color(input_img, output_img)
    out_imgs.append(output_img)
    out_img_names.append(img_id)
    num += 1

    if num == 5:
        break

    print(num)
    
print('✅ Test 데이터셋에 대한 모든 이미지 생성 완료.')


# 추론 결과물 디렉토리 생성
os.makedirs(CFG['SUB_DIR'], exist_ok=True)

# **중요** 추론 이미지 평가용 Embedding 추출 모델
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai") # 모델명을 반드시 일치시켜야합니다.

clip_model.to(device)
# 평가 제출을 위해 추론된 이미지들을 ViT-L-14 모델로 임베딩 벡터(Feature)를 추출합니다.
feat_imgs = []
for output_img, img_id in tqdm(zip(out_imgs, out_img_names)):
    path_out_img = CFG['SUB_DIR'] + '/' + img_id + '.png' 
    output_img.save(path_out_img)
    # 평가용 임베딩 생성 및 저장
    output_img = clip_preprocess(output_img).unsqueeze(0).cuda()
    with torch.no_grad():
        feat_img = clip_model.encode_image(output_img)
        feat_img /= feat_img.norm(dim=-1, keepdim=True) # L2 정규화 필수

    feat_img = feat_img.detach().cpu().numpy().reshape(-1)
    feat_imgs.append(feat_img)

feat_imgs = np.array(feat_imgs)
vec_columns = [f'vec_{i}' for i in range(feat_imgs.shape[1])]
feat_submission = pd.DataFrame(feat_imgs, columns=vec_columns)
feat_submission.insert(0, 'ID', out_img_names)

feat_submission.to_csv(CFG['SUB_DIR']+'/embed_submission.csv', index=False)

# 최종 제출물 (ZIP) 생성 경로
# 제출물 (ZIP) 내에는 디렉토리(폴더)가 없이 구성해야합니다.
zip_path = './submission.zip'

# zip 파일 생성
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_name in os.listdir(CFG['SUB_DIR']):
        file_path = os.path.join(CFG['SUB_DIR'], file_name)

        # 일반 파일이며 숨김 파일이 아닌 경우만 포함
        if os.path.isfile(file_path) and not file_name.startswith('.'):
            zipf.write(file_path, arcname=file_name)

print(f"✅ 압축 완료: {zip_path}")

end_time = time.time()
print(f"\n✅ 전체 추론 시간: {end_time - start_time:.2f}초")
