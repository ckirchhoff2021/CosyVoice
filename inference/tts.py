
import os
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import time
import torch


class SpeechGenerator:
    def __init__(self):
        self.model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B'
        self.cosyvoice = AutoModel(model_dir=self.model_dir)

    def timestamp(self):
        return int(time.time() * 1000)

    def voice_clone(
        self, 
        tts_text, 
        prompt_wav,
        save_file=''
    ):
        '''
        example:
        tts_text = '八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮'
        prompt_wav = './asset/zero_shot_prompt.wav'
        save_file = 'outputs/tt_clone_{}.wav'.format(self.timestamp())
        '''
        prompt_text='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
        if len(save_file) == 0:
            save_file = 'outputs/tt_clone_{}.wav'.format(self.timestamp())
        
        try:
            audio_chunks = []
            for _, j in enumerate(
                self.cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False)
            ):
                audio_chunks.append(j['tts_speech'])
            # 拼接所有音频片段
            full_audio = torch.cat(audio_chunks, dim=1)
            torchaudio.save(save_file, full_audio, self.cosyvoice.sample_rate)
            print("Success, save file {}".format(save_file))
            
        except Exception as e:
            print("Task failed ", e)
    
    def cross_lingual_gen(
        self, 
        tts_text,
        prompt_wav,
        save_file='',
    ):
        '''
        example:
        tts_text = 'You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]'
        prompt_wav = './asset/zero_shot_prompt.wav'
        save_file = 'outputs/tts_cross_{}.wav'.format(self.timestamp())
        '''
        if len(save_file) == 0:
            save_file = 'outputs/tts_cross_{}.wav'.format(self.timestamp())
        
        try:
            audio_chunks = []
            for _, j in enumerate(
                self.cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=False)
            ):
                audio_chunks.append(j['tts_speech'])
            # 拼接所有音频片段
            full_audio = torch.cat(audio_chunks, dim=1)
            torchaudio.save(save_file, full_audio, self.cosyvoice.sample_rate)
            print("Success, save file {}".format(save_file))
            
        except Exception as e:
            print("Task failed ", e)
            
    
    def instruct_gen(
        self, 
        tts_text,
        instruct_prompt,
        prompt_wav,
        save_file='',
    ):
        '''
        example:
        tts_text = '好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。'
        instruct_prompt = 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>'
        prompt_wav = './asset/zero_shot_prompt.wav'
        save_file = 'outputs/tts_instruct_{}.wav'.format(self.timestamp())
        
        tts_text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
        instruct_prompt = 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>'
        prompt_wav = './asset/zero_shot_prompt.wav'
        save_file = 'outputs/tts_instruct_{}.wav'.format(self.timestamp())
        
        '''
        if len(save_file) == 0:
            save_file = 'outputs/tts_instruct_{}.wav'.format(self.timestamp())
        
        try:
            audio_chunks = []
            for _, j in enumerate(
                self.cosyvoice.inference_instruct2(tts_text, instruct_prompt, prompt_wav, stream=False)
            ):
                audio_chunks.append(j['tts_speech'])
            # 拼接所有音频片段
            full_audio = torch.cat(audio_chunks, dim=1)
            torchaudio.save(save_file, full_audio, self.cosyvoice.sample_rate)
            print("Success, save file {}".format(save_file))
            
        except Exception as e:
            print("Task failed ", e)


def case():
    generator = SpeechGenerator()
    
    '''
    generator.voice_clone('八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮', './asset/zero_shot_prompt.wav')
    generator.cross_lingual_gen(
        'You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]', 
        './asset/zero_shot_prompt.wav'
    )
    generator.instruct_gen(
        '好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。',
        'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
        './asset/zero_shot_prompt.wav'
    )
    '''
    
    text = open('inference/refs.txt', 'r').read()
    generator.instruct_gen(
        text,
        'You are a helpful assistant. 请用有磁性的嗓音生成。<|endofprompt|>',
        './asset/zero_shot_prompt.wav'
    )
    


if __name__ == '__main__':
    case()
