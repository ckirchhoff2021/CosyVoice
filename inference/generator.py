import sys
import os
import warnings
import argparse

# 在最最开始就使用 os.dup2 重定向文件描述符
# 这是最底层的方式，可以捕获 C++ 库的输出
original_stdout_fd = os.dup(1)
original_stderr_fd = os.dup(2)
devnull_fd = os.open(os.devnull, os.O_WRONLY)

# 先重定向到 /dev/null
os.dup2(devnull_fd, 1)
os.dup2(devnull_fd, 2)

# 设置环境变量
os.environ['ORT_LOGGING_LEVEL'] = '3'  # 3 = ERROR 级别
os.environ['ORT_LOG_LEVEL'] = '3'  # 另一种设置方式
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# 抑制 Python 警告
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import logging
import torchaudio
import time


# 设置 logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ['transformers', 'deepspeed', 'onnxruntime', 'torch', 'lightning', 'pkg_resources', 'festival', 'cosyvoice']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# 用于恢复和禁用输出的函数
def enable_print():
    os.dup2(original_stdout_fd, 1)
    os.dup2(original_stderr_fd, 2)

def disable_print():
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)


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
            for _, j in enumerate(
                self.cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False)
            ):
                torchaudio.save(save_file, j['tts_speech'], self.cosyvoice.sample_rate)
            enable_print()
            print("Success, save file {}".format(save_file))
            disable_print()
            
        except Exception as e:
            enable_print()
            print("Task failed ", e)
            disable_print()
    
    
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
            for _, j in enumerate(
                self.cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=False)
            ):
                torchaudio.save(save_file, j['tts_speech'], self.cosyvoice.sample_rate)
            enable_print()
            print("Success, save file {}".format(save_file))
            disable_print()
            
        except Exception as e:
            enable_print()
            print("Task failed ", e)
            disable_print()
            
    
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
            for _, j in enumerate(
                self.cosyvoice.inference_instruct2(tts_text, instruct_prompt, prompt_wav, stream=False)
            ):
                torchaudio.save(save_file, j['tts_speech'], self.cosyvoice.sample_rate)
            enable_print()
            print("Success, save file {}".format(save_file))
            disable_print()
            
        except Exception as e:
            enable_print()
            print("Task failed ", e)
            disable_print()


def case():
    # 初始化时保持静默
    disable_print()
    generator = SpeechGenerator()
    
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
    
    # 最后恢复输出
    enable_print()


if __name__ == '__main__':
    # case()
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='voice_clone', choices=['voice_clone', 'cross_lingual_gen', 'instruct_gen'])
    parser.add_argument('--tts_text', type=str, default='八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮')
    parser.add_argument('--prompt_wav', type=str, default='./asset/zero_shot_prompt.wav')
    parser.add_argument('--instruct_prompt', type=str, default='')
    
    args = parser.parse_args()
    disable_print()
    
    generator = SpeechGenerator()
    if args.task_type == 'voice_clone':
        generator.voice_clone(args.tts_text, args.prompt_wav)
    elif args.task_type == 'cross_lingual_gen':
        generator.cross_lingual_gen(args.tts_text, args.prompt_wav)
    elif args.task_type == 'instruct_gen':
        generator.instruct_gen(args.tts_text, args.instruct_prompt, args.prompt_wav)
    else:
        print("Unknown task type")
    
    enable_print() 
