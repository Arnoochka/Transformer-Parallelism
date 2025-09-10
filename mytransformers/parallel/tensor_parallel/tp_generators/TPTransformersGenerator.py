from torch.nn import Module, ModuleList
import torch
from torch.distributed import ProcessGroup
from .TPTransformerLayersGenerator import (
    TPTransformerEncoderLayerGenerator,
    TPTransformerDecoderLayerGenerator
)
from .TPModuleGenerator import TPModuleGenerator
from .TPLinearLayersGenerator import TPColumnLinearGenerator
    
class TPTransformerEncoderModelGenerator(TPModuleGenerator):
    encoder_layer_gen = TPTransformerEncoderLayerGenerator
    linear_gen = TPColumnLinearGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        module.encoder_layers = ModuleList(
            [cls.encoder_layer_gen(layer, tp_group)
             for layer in module.encoder_layers])
        
        cls.linear_gen.use_all_gather = True
        module.linear = cls.linear_gen(module.linear, tp_group)
        device = torch.cuda.current_device()
        module.embedding = module.embedding.to(device)
        module.pos_encoding = module.pos_encoding.to(device)
        return module
    
class TPTransformerDecoderModelGenerator(TPModuleGenerator):
    decoder_layer_gen = TPTransformerDecoderLayerGenerator
    linear_gen = TPColumnLinearGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        module.decoder_layers = ModuleList(
            [cls.decoder_layer_gen(layer, tp_group)
             for layer in module.decoder_layers])
        
        cls.linear_gen.use_all_gather = True
        module.linear = cls.linear_gen(module.linear, tp_group)
        device = torch.cuda.current_device()
        module.embedding = module.embedding.to(device)
        module.pos_encoding = module.pos_encoding.to(device)
        return module
    
class TPTransformerEncoderDecoderModelGenerator(TPModuleGenerator):
    encoder_gen = TPTransformerEncoderModelGenerator
    decoder_gen = TPTransformerDecoderModelGenerator
    linear_gen = TPColumnLinearGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        module.encoder_layers = cls.encoder_gen(module, tp_group).encoder_layers
        module.decoder_layers = cls.decoder_gen(module, tp_group).decoder_layers
        cls.linear_gen.use_all_gather = True
        module.linear = TPColumnLinearGenerator(module.linear, tp_group)
        device = torch.cuda.current_device()
        module.embedding = module.embedding.to(device)
        module.pos_encoding = module.pos_encoding.to(device)
        return module