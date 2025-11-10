import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


def local_test():
    cosyvoice = CosyVoice2('/home/chenxiang.101/workspace/checkpoints/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

    # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
    # zero_shot usage
    # prompt_speech_16k = load_wav('/home/chenxiang.101/workspace/DeepThinking/audio/resources/voices/spk_1757671327.wav', 16000)
    # for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    #     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
    # for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
    #     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # instruct usage
    # for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    #     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        
    # prompt_speech_16k = load_wav('/home/chenxiang.101/workspace/DeepThinking/audio/resources/voices/spk_1757671327.wav', 16000)
    # speech_text = "这个呀就是我们精心制作准备的纪念品。大家可以看到这个色泽和这个材质啊，哎呀多么的光彩照人。"

    # prompt_speech_16k = load_wav('/home/chenxiang.101/workspace/DeepThinking/audio/resources/voices/yifan.wav', 16000)
    # speech_text = "啊？说什么！测试测试 "

    prompt_speech_16k = load_wav("/home/chenxiang.101/workspace/Self.wav", 16000)
    speech_text = "我其实呢不是很喜欢用豆包，而且我不太喜欢用电动车。嗯，我只喜欢油车，我是一个非常聪明的人。"

    for i, j in enumerate(cosyvoice.inference_zero_shot('大家好，我叫陈祥，很高兴认识大家，我来自浙江杭州，我喜欢喝冻顶乌龙，不过最近好久没喝了，另外我还喜欢喝咖啡，咖啡能让我的超级大脑保持清醒。', speech_text, prompt_speech_16k, stream=False)):
        torchaudio.save('zero_shot_{}_x.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        
        
def server_test():
    url = "https://sd46bj4fk47gkoier6psg.apigateway-cn-beijing.volceapi.com/mlp/s-20251106195228-m6hm8/v1/audio/speech/"
    import requests
    import numpy as np
    import torch
    
    data = {
        'input': "你可真是个大聪明，你给我讲个故事吧！",
        'voice': "wise"
    }
    response = requests.request("POST", url, json=data, stream=True)
    tts_audio = b''
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    torchaudio.save('outputs/server_test.wav', tts_speech, 22050)
        
        
if __name__ == '__main__':
    # local_test()
    server_test()
