from torch.nn import Module, ModuleList
import torch
from torch.distributed import ProcessGroup
from .ParallelTransformerLayersGenerator import (
    ParallelTransformerEncoderLayerGenerator,
    ParallelTransformerDecoderLayerGenerator
)
from .ParallelModuleGenerator import TensorParallelModuleGenerator
from .ParallelLinearLayersGenerator import (
    ColumnParallelLinearGenerator,
    RowParallelLinearGenerator
)
from .ParallelTransformers import (
    ParallelTransformerEncoderModel,
    ParallelTransformerDecoderModel,
    ParallelTransformerEncoderDecoderModel
)
    
class ParallelTransformerEncoderModelGenerator(TensorParallelModuleGenerator):
    config = None
    encoder_layer_gen = ParallelTransformerEncoderLayerGenerator
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderModel:
        cls.encoder_layer_gen.config = cls.config
        layers = ModuleList(
            [cls.encoder_layer_gen(layer, tp_group) for layer in module.layers]
        )
        ColumnParallelLinearGenerator.use_all_gather = True
        linear = ColumnParallelLinearGenerator(module.linear, tp_group)
        device = torch.cuda.current_device()
        embedding = module.embedding.to(device)
        pos_encoding = module.pos_encoding.to(device)
        return ParallelTransformerEncoderModel(cls.config,
                                               layers,
                                               linear,
                                               embedding,
                                               pos_encoding,
                                               tp_group)
    
class ParallelTransformerDecoderModelGenerator(TensorParallelModuleGenerator):
    config = None
    decoder_layer_gen = ParallelTransformerDecoderLayerGenerator
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderModel:
        cls.decoder_layer_gen.config = cls.config
        layers = ModuleList(
            [cls.decoder_layer_gen(layer, tp_group) for layer in module.layers]
        )
        ColumnParallelLinearGenerator.use_all_gather = True
        linear = ColumnParallelLinearGenerator(module.linear, tp_group)
        device = torch.cuda.current_device()
        embedding = module.embedding.to(device)
        pos_encoding = module.pos_encoding.to(device)
        return ParallelTransformerDecoderModel(cls.config,
                                               layers,
                                               linear,
                                               embedding,
                                               pos_encoding,
                                               tp_group)
    
class ParallelTransformerEncoderDecoderModelGenerator(TensorParallelModuleGenerator):
    config = None
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerEncoderDecoderModel:
        ParallelTransformerEncoderModelGenerator.config = cls.config
        ParallelTransformerDecoderModelGenerator.config = cls.config
        
        module.layers = module.encoder_layers
        encoder = ParallelTransformerEncoderModelGenerator(module, tp_group)
        module.layers = module.decoder_layers
        decoder = ParallelTransformerDecoderModelGenerator(module, tp_group)
        ColumnParallelLinearGenerator.use_all_gather = True
        linear = ColumnParallelLinearGenerator(module.linear, tp_group)
        device = torch.cuda.current_device()
        embedding = module.embedding.to(device)
        pos_encoding = module.pos_encoding.to(device)
        return ParallelTransformerEncoderDecoderModel(cls.config,
                                                      encoder,
                                                      decoder, 
                                                      linear,
                                                      embedding,
                                                      pos_encoding,
                                                      tp_group)