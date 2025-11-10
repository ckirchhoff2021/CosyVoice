import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """TTS请求模型"""
    model: str = Field(default="cosyvoice", description="TTS模型名称")
    input: str = Field(..., description="要转换为语音的文本内容")
    voice: str = Field(default="wise", description="语音音色")


default_prompts = {
    "wise": {
        "prompt_text": "我其实呢不是很喜欢用豆包，而且我不太喜欢用电动车。嗯，我只喜欢油车，我是一个非常聪明的人。",
        "prompt_wav": "asset/wise.wav"
    },
    "default": {
        "prompt_text": "希望你以后能够做的比我还好呦。",
        "prompt_wav": "asset/zero_shot_prompt.wav"
    }
}

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/v1/audio/speech/")
@app.post("/v1/audio/speech/")
async def openai_speech(request: TTSRequest):
    input_text = request.input
    voice = request.voice
    if voice not in default_prompts:
        voice = "default"
    prompt_text = default_prompts[voice]["prompt_text"]
    prompt_wav = default_prompts[voice]["prompt_wav"]
    
    prompt_speech_16k = load_wav(prompt_wav, 16000)
    model_output = cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)


# python server.py --model_dir /home/chenxiang.101/workspace/checkpoints/CosyVoice2-0.5B/
