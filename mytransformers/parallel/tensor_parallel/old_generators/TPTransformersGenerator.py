from torch.nn import Module, ModuleList
import torch
from torch.distributed import ProcessGroup
from .TPTransformerLayersGenerator import (
    TPTransformerEncoderLayerGenerator,
    TPTransformerDecoderLayerGenerator
)
from .TPModuleGenerator import TPModuleGenerator
from .TPLinearLayersGenerator import TPColumnLinearGenerator
from mytransformers.parallel.tensor_parallel.tp_layers import (
    TPTransformerEncoderModel,
    TPTransformerDecoderModel,
    TPTransformerEncoderDecoderModel
)
    
class TPTransformerEncoderModelGenerator(TPModuleGenerator):
    config = None
    encoder_layer_gen = TPTransformerEncoderLayerGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPTransformerDecoderModel:
        cls.encoder_layer_gen.config = cls.config
        layers = ModuleList(
            [cls.encoder_layer_gen(layer, tp_group) for layer in module.encoder_layers]
        )
        TPColumnLinearGenerator.use_all_gather = True
        linear = TPColumnLinearGenerator(module.linear, tp_group)
        device = torch.cuda.current_device()
        embedding = module.embedding.to(device)
        pos_encoding = module.pos_encoding.to(device)
        return TPTransformerEncoderModel(cls.config,
                                               layers,
                                               linear,
                                               embedding,
                                               pos_encoding,
                                               tp_group)
    
class TPTransformerDecoderModelGenerator(TPModuleGenerator):
    config = None
    decoder_layer_gen = TPTransformerDecoderLayerGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPTransformerDecoderModel:
        cls.decoder_layer_gen.config = cls.config
        layers = ModuleList(
            [cls.decoder_layer_gen(layer, tp_group) for layer in module.decoder_layers]
        )
        TPColumnLinearGenerator.use_all_gather = True
        linear = TPColumnLinearGenerator(module.linear, tp_group)
        device = torch.cuda.current_device()
        embedding = module.embedding.to(device)
        pos_encoding = module.pos_encoding.to(device)
        return TPTransformerDecoderModel(cls.config,
                                               layers,
                                               linear,
                                               embedding,
                                               pos_encoding,
                                               tp_group)
    
class TPTransformerEncoderDecoderModelGenerator(TPModuleGenerator):
    config = None
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPTransformerEncoderDecoderModel:
        TPTransformerEncoderModelGenerator.config = cls.config
        TPTransformerDecoderModelGenerator.config = cls.config
        
        encoder = TPTransformerEncoderModelGenerator(module, tp_group)
        decoder = TPTransformerDecoderModelGenerator(module, tp_group)
        TPColumnLinearGenerator.use_all_gather = True
        linear = TPColumnLinearGenerator(module.linear, tp_group)
        device = torch.cuda.current_device()
        embedding = module.embedding.to(device)
        pos_encoding = module.pos_encoding.to(device)
        return TPTransformerEncoderDecoderModel(cls.config,
                                                      encoder,
                                                      decoder, 
                                                      linear,
                                                      embedding,
                                                      pos_encoding,
                                                      tp_group)