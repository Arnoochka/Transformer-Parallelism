from torch.distributed import ProcessGroup
from torch.nn import Module
from torch.nn import Module
from .ParallelModuleGenerator import TensorParallelModuleGenerator
from ..TransformerLayers import TransformerType
from .ParallelAttentionGenerator import ParallelCrossAttentionGenerator, ParallelSelfAttentionGenerator
from .ParallelLinearLayersGenerator import ColumnParallelLinearGenerator
from .layers import ParallelTransformerEncoderLayer, ParallelTransformerDecoderLayer
    
class ParallelTransformerDecoderGenerator(TensorParallelModuleGenerator):
    config = None
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderLayer:
        ParallelSelfAttentionGenerator.config = cls.config
        ParallelCrossAttentionGenerator.config = cls.config
        
        use_encoder = cls.config.transformer_type == TransformerType.EncoderDecoder
        self_attn = ParallelSelfAttentionGenerator(module.self_attn, tp_group)
        ColumnParallelLinearGenerator.use_all_gather = True
        ffn = ColumnParallelLinearGenerator(module.ffn, tp_group)
        if use_encoder:
            cross_attn = ParallelCrossAttentionGenerator(module.cross_attn)
            return ParallelTransformerDecoderLayer(cls.config,
                                                   self_attn,
                                                   cross_attn,
                                                   ffn,
                                                   tp_group)
        else:
            return ParallelTransformerDecoderLayer(cls.config,
                                       self_attn,
                                       None,
                                       ffn,
                                       tp_group)
        
        
class ParallelTransformerEncoderGenerator(TensorParallelModuleGenerator):
    config = None
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderLayer:
        ParallelSelfAttentionGenerator.config = cls.config
        attn = ParallelSelfAttentionGenerator(module.self_attn, tp_group)
        ColumnParallelLinearGenerator.use_all_gather = True
        ffn = ColumnParallelLinearGenerator(module.ffn, tp_group)
        
        return ParallelTransformerEncoderLayer(cls.config,
                                               attn,
                                               ffn,
                                               tp_group)
        
        