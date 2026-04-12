import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
from PIL import Image

def prepare_blip_model(cache_dir="./model_cache"):
    """
    作用：下载 BLIP 模型到指定本地路径，并验证加载
    """
    model_id = "Salesforce/blip-image-captioning-base"
    
    # 确保本地缓存目录存在
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"创建目录: {cache_dir}")

    print(f"正在从 Hugging Face 下载/加载模型: {model_id}")
    print(f"目标本地路径: {os.path.abspath(cache_dir)}")

    try:
        # 下载并保存 processor 和 model 到本地路径
        # 第一次运行会耗时，之后会直接从本地读取
        processor = BlipProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = BlipForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir)
        
        print("\n" + "="*30)
        print("✅ 模型准备就绪！")
        print(f"模型已缓存至: {cache_dir}")
        print("="*30)
        
        return True
    except Exception as e:
        print(f"\n❌ 模型下载失败: {e}")
        print("提示：请确认是否运行了 'source /etc/network_turbo' 和设置了镜像源。")
        return False

def prepare_clip_model(cache_dir="./clip_model"):
    """
    作用：下载 CLIP 模型到指定本地路径，并验证加载
    """
    model_id = "openai/clip-vit-base-patch32"
    
    # 确保本地缓存目录存在
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"创建目录: {cache_dir}")

    print(f"正在从 Hugging Face 下载/加载模型: {model_id}")
    print(f"目标本地路径: {os.path.abspath(cache_dir)}")

    try:
        # 下载并保存 processor 和 model 到本地路径
        # 第一次运行会下载，之后直接从本地读取
        processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
        
        print("\n" + "="*30)
        print("✅ CLIP 模型准备就绪！")
        print(f"模型已缓存至: {cache_dir}")
        print("="*30)
        
        return True
    except Exception as e:
        print(f"\n❌ 模型下载失败: {e}")
        print("提示：请确认是否运行了 'source /etc/network_turbo' 和设置了镜像源。")
        return False

BLIP_CACHE = "./blip_base_model"
CLIP_CACHE = "./clip_model"
BLIP_ID = "Salesforce/blip-image-captioning-base"
CLIP_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEEPSEEK_KEY = "sk-3df8f7c51eb8481d96c8969ecac4fb54" 

def run_prediction(img_path):
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    raw_image = Image.open(img_path).convert('RGB')

    # --- 1. BLIP 描述 ---
    blip_proc = BlipProcessor.from_pretrained(BLIP_ID, cache_dir=BLIP_CACHE)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_ID, cache_dir=BLIP_CACHE).to(DEVICE)
    blip_inputs = blip_proc(raw_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(**blip_inputs, max_new_tokens=50)
    caption = blip_proc.decode(out[0], skip_special_tokens=True)
    print(f"\n[1. BLIP 描述]: {caption}")

    # --- 2. DeepSeek 提议 (LLM 提议) ---
    candidate_prompt = f"The image is described as '{caption}'. Predict 4 specific materials it might be (e.g., stainless steel, aluminum, chrome-plated steel). Return ONLY names separated by commas."
    cand_resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": candidate_prompt}]
    )
    candidates = [c.strip() for c in cand_resp.choices[0].message.content.split(',')]
    print(f"[2. 候选名单]: {candidates}")

    # --- 3. CLIP 筛选 (视觉校验) ---
    clip_proc = CLIPProcessor.from_pretrained(CLIP_ID, cache_dir=CLIP_CACHE)
    clip_model = CLIPModel.from_pretrained(CLIP_ID, cache_dir=CLIP_CACHE).to(DEVICE)
    clip_inputs = clip_proc(text=candidates, images=raw_image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)
        probs = clip_outputs.logits_per_image.softmax(dim=1)
    
    best_material = candidates[probs.argmax().item()]
    print(f"[3. CLIP 定案]: {best_material}")

    # --- 4. DeepSeek 定案 (JSON 格式) ---
    print(f"[4. 生成物理参数 JSON...]")
    # 强制要求 JSON 格式，不解释
    final_prompt = f"""
    The object is confirmed to be '{best_material}'. 
    Provide physical parameters for simulation in the following JSON format:
    {{
      "material": "{best_material}",
      "density_kg_m3": float,
      "static_friction": float,
      "restitution": float
    }}
    Return ONLY the raw JSON string without any explanation or markdown backticks.
    """
    
    final_resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a physics data server. You only output valid JSON."},
            {"role": "user", "content": final_prompt}
        ]
    )
    return final_resp.choices[0].message.content

# if __name__ == "__main__":
#     prepare_blip_model("./blip_base_model")
#     prepare_clip_model()