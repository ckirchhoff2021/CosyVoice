import gradio as gr
import requests
import torch
import torchaudio
import numpy as np
import os
import uuid

# 默认参数
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

def tts_inference(text, mode="sft", spk_id="中文女", prompt_text="", prompt_wav=None, instruct_text=""):
    """
    调用TTS服务进行语音合成
    """
    try:
        # 生成唯一的文件名
        unique_id = str(uuid.uuid4())[:8]
        output_wav = f"tts_output_{unique_id}.wav"
        
        url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/inference_{mode}"
        
        if mode == 'sft':
            payload = {
                'tts_text': text,
                'spk_id': spk_id
            }
            response = requests.request("GET", url, data=payload, stream=True)
        elif mode == 'zero_shot':
            payload = {
                'tts_text': text,
                'prompt_text': prompt_text
            }
            # 如果提供了参考音频文件
            if prompt_wav is not None:
                files = [('prompt_wav', ('prompt_wav', open(prompt_wav, 'rb'), 'application/octet-stream'))]
                response = requests.request("GET", url, data=payload, files=files, stream=True)
            else:
                # 使用默认的参考音频
                payload['prompt_text'] = "希望你以后能够做的比我还好呦。" if not prompt_text else prompt_text
                files = [('prompt_wav', ('prompt_wav', open('asset/zero_shot_prompt.wav', 'rb'), 'application/octet-stream'))]
                response = requests.request("GET", url, data=payload, files=files, stream=True)
        elif mode == 'cross_lingual':
            payload = {
                'tts_text': text,
            }
            if prompt_wav is not None:
                files = [('prompt_wav', ('prompt_wav', open(prompt_wav, 'rb'), 'application/octet-stream'))]
                response = requests.request("GET", url, data=payload, files=files, stream=True)
            else:
                # 使用默认的参考音频
                files = [('prompt_wav', ('prompt_wav', open('asset/cross_lingual_prompt.wav', 'rb'), 'application/octet-stream'))]
                response = requests.request("GET", url, data=payload, files=files, stream=True)
        else:  # instruct2模式
            if not prompt_wav:
                prompt_wav = 'asset/zero_shot_prompt.wav'
            
            files = [('prompt_wav', ('prompt_wav', open(prompt_wav, 'rb'), 'application/octet-stream'))]
            payload = {
                'tts_text': text,
                'instruct_text': instruct_text,
            }
            response = requests.request("GET", url, data=payload, files=files, stream=True)
        
        # 处理响应
        tts_audio = b''
        for r in response.iter_content(chunk_size=16000):
            tts_audio += r
        
        # 转换音频数据
        tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
        
        # 保存音频文件
        torchaudio.save(output_wav, tts_speech, 22050)
        
        return output_wav
    except Exception as e:
        print(f"Error in TTS inference: {e}")
        return None

def openai_tts(text, voice="wise"):
    """
    调用OpenAI兼容的TTS服务
    """
    try:
        # 生成唯一的文件名
        unique_id = str(uuid.uuid4())[:8]
        output_wav = f"tts_output_{unique_id}.wav"
        
        url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1/audio/speech"
        payload = {
            'input': text,
            'voice': voice
        }
        response = requests.request("POST", url, json=payload, stream=True)
        
        # 处理响应
        tts_audio = b''
        for r in response.iter_content(chunk_size=16000):
            tts_audio += r
        
        # 转换音频数据
        tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
        
        # 保存音频文件
        torchaudio.save(output_wav, tts_speech, 22050)
        
        return output_wav
    except Exception as e:
        print(f"Error in OpenAI TTS: {e}")
        return None

def generate_speech(text, mode, spk_id, prompt_text, prompt_wav, voice, instruct_text):
    """
    Gradio接口函数
    """
    if not text.strip():
        return None, "请输入要转换为语音的文本"
    
    # 检查TTS服务是否可用
    try:
        requests.get(f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/")
    except requests.exceptions.ConnectionError:
        return None, f"无法连接到TTS服务 (http://{DEFAULT_HOST}:{DEFAULT_PORT})，请确保服务正在运行"
    
    if mode == "openai":
        wav_file = openai_tts(text, voice)
    else:
        wav_file = tts_inference(text, mode, spk_id, prompt_text, prompt_wav, instruct_text)
    
    if wav_file and os.path.exists(wav_file):
        return wav_file, f"语音合成成功！模式: {mode}"
    else:
        return None, "语音合成失败，请检查服务和参数设置"

# 创建Gradio界面
with gr.Blocks(title="TTS语音合成演示") as demo:
    gr.Markdown("# 🎵 TTS语音合成演示")
    gr.Markdown("将文本转换为自然流畅的语音")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="输入文本",
                placeholder="请输入要转换为语音的文本...",
                lines=3,
                value="你好，我是爱新觉罗玄烨，你是有什么事情要禀报吗？"
            )
            
            mode = gr.Radio(
                choices=["sft", "zero_shot", "cross_lingual", "instruct2", "openai"], # instruct2 for cosyvoice2
                value="openai",
                label="合成模式"
            )
            
            with gr.Group():
                gr.Markdown("### 模式参数设置")
                spk_id = gr.Textbox(label="说话人ID", value="中文女")
                voice = gr.Textbox(label="声音类型(OpenAI模式)", value="wise")
                
                instruct_text = gr.Textbox(
                    label="Instruct文本(Instruct模式)", 
                    value="用四川话说这句话"
                )
                
                prompt_text = gr.Textbox(
                    label="参考文本(Zero-shot/Cross-lingual模式)", 
                    value="希望你以后能够做的比我还好呦。"
                )
                
                prompt_wav = gr.Audio(
                    label="参考音频(Zero-shot/Cross-lingual模式)", 
                    type="filepath"
                )
            
            generate_btn = gr.Button("生成语音", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="合成语音")
            status_output = gr.Textbox(label="状态信息", interactive=False)
    
    # 示例
    gr.Examples(
        examples=[
            ["你好，我是超级无敌破坏神，你找我有事吗？", "sft", "中文女", "", None, "wise"],
            ["很高兴认识你，我是爱新觉罗玄烨。", "zero_shot", "中文女", "希望你以后能够做的比我还好呦。", "asset/zero_shot_prompt.wav", "wise"],
            ["Hello, I am your daddy, I am a super powerful destroyer.", "cross_lingual", "", "", "asset/cross_lingual_prompt.wav", "wise"]
        ],
        inputs=[text_input, mode, spk_id, prompt_text, prompt_wav, voice, instruct_text],
        outputs=[audio_output, status_output],
        fn=generate_speech,
        cache_examples=False
    )
    
    # 事件处理
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, mode, spk_id, prompt_text, prompt_wav, voice, instruct_text],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
