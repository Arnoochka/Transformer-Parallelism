from torch.nn import Module, ModuleList
from torch.distributed import ProcessGroup
from . import (ParallelTransformerEncoderGenerator,
               ParallelTransformerDecoderGenerator,
               ColumnParallelLinearGenerator,
               TensorParallelModuleGenerator)
from .ParallelTransformers import (
    ParallelTransformerEncoderModel,
    ParallelTransformerDecoderModel,
    ParallelTransformerEncoderDecoderModel
)
    
class ParallelTransformerEncoderModelGenerator(TensorParallelModuleGenerator):
    config = None
    encoder_layer_gen = ParallelTransformerEncoderGenerator
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderModel:
        cls.encoder_layer_gen.config = cls.config
        layers = ModuleList(
            [cls.encoder_layer_gen(layer) for layer in module.layers]
        )
        ColumnParallelLinearGenerator.use_all_gather = True
        linear = ColumnParallelLinearGenerator(module, tp_group)
        return ParallelTransformerEncoderModel(cls.config, layers, linear, tp_group)
    
class ParallelTransformerDecoderModelGenerator(TensorParallelModuleGenerator):
    config = None
    decoder_layer_gen = ParallelTransformerDecoderGenerator
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderModel:
        cls.decoder_layer_gen.config = cls.config
        layers = ModuleList(
            [cls.decoder_layer_gen(layer) for layer in module.layers]
        )
        ColumnParallelLinearGenerator.use_all_gather = True
        linear = ColumnParallelLinearGenerator(module, tp_group)
        return ParallelTransformerEncoderModel(cls.config, layers, linear, tp_group)
    
class ParallelTransformerEncoderDecoderModelGenerator(TensorParallelModuleGenerator):
    config = None
    def __new__(cls, module: Module, tp_group: ProcessGroup, config) -> ParallelTransformerEncoderDecoderModel:
        ParallelTransformerEncoderModelGenerator.config = cls.config
        ParallelTransformerDecoderModelGenerator.config = cls.config
        
        module.layers = module.encoder_layers
        encoder = ParallelTransformerEncoderModelGenerator(module, tp_group)
        module.layers = module.decoder_layers
        decoder = ParallelTransformerDecoderModelGenerator(module, tp_group)
        ColumnParallelLinearGenerator.use_all_gather = True
        linear = ColumnParallelLinearGenerator(module)
        
        return ParallelTransformerEncoderDecoderModel(config,
                                                      encoder,
                                                      decoder, 
                                                      linear,
                                                      tp_group)