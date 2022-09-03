from .mlp import MLP
from .gin import GIN
from .spline import SplineCNN
from .rel import RelCNN
from .dgmc import DGMC
from .gan_t import GAN
from .GOTSim import GOTSim
from .BERT import case_bert

__all__ = [
    'MLP',
    'GIN',
    'SplineCNN',
    'RelCNN',
    'DGMC',
    'GAN',
    'GOTSim',
    'case_bert',
]#这里是为了将常规程序包的对象绑定相应的名称，该文件在导入时隐式执行
