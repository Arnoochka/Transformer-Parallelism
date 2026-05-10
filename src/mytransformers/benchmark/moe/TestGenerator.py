from .TestModel import TestModel
import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
from typing import List, Tuple, Callable, Any, Dict
from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    FakeModule, FakeTensorModule, InnerStrategyModule, LeaderStrategyModule, FinalStrategyDictModule)
from torch.distributed import ProcessGroup
from mytransformers.parallel.moe_parallel.pipeline import BaseScheduler
from mytransformers.parallel.moe_parallel.generators import MoePipelineGenerator, MoeLayerConfig, MoeSparseBlockDPModuleGenerator
from mytransformers.parallel.moe_parallel.layers import MoeExperts, MoeDPExperts, MoeComputeLayer, MoeFakeLayer

class TestPipeGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: TestModel,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                inner_comm_groups: List[ProcessGroup],
                final_comm_group: ProcessGroup,
                embed_size: int,
                vocab_size: int,
                device: torch.device) -> nn.Module:
        
        num_layers = len(module.layers)
        
        orig_modules = cls.get_orig_modules(module)
        fake_modules = cls.get_fake_modules(num_layers, device)
        modules = [(name, orig_module, fake_module)
                   for (name, orig_module), fake_module in zip(orig_modules, fake_modules)]
        num_modules_per_stage = len(modules) // num_stages
        inner_boundary_points = [num_modules_per_stage * k - 1 for k in range(num_stages)][1:]
        
        inner_strategies = cls.get_inner_strategies(len(inner_boundary_points))
        
        stage: nn.ModuleDict = PipelineGenerator.get_stage(modules,
                                                           inner_boundary_points,
                                                           groups_info,
                                                           inner_comm_groups,
                                                           inner_strategies)
        
        module = cls.replace_modules(module, stage)
        modules = [module for name, module in stage.items()]
        fake_args = TestPipeGenerator.build_fake_args(num_layers, embed_size, vocab_size)
        pipeline = PipelineGenerator(module,
                                     modules,
                                     FinalStrategyDictModule(send_rank=num_stages-1),
                                     final_comm_group,
                                     fake_args)
        return pipeline.to(device)
        
    @staticmethod
    def get_fake_modules(num_layers: int, device: torch.device) -> List[FakeModule]:
        return [FakeTensorModule(device)] +\
            [FakeTensorModule(device) for _ in range(num_layers)] +\
                [FakeTensorModule(device)]
    
    @staticmethod
    def get_orig_modules(module: TestModel) -> List[Tuple[str, nn.Module]]:
        return [("embed_tokens", module.embed_tokens)] +\
            [(f"layers-{idx}", layer) for idx, layer in enumerate(module.layers)] +\
                [('lm_head', module.lm_head)]
    
    @staticmethod
    def replace_modules(module: TestModel, stage: nn.ModuleDict) -> nn.Module:
        setattr(module, 'embed_tokens', stage['embed_tokens'])
        setattr(module, 'lm_head', stage['lm_head'])
        layers = nn.ModuleList()
        for name, pipe_module in stage.items():
            if "layers" in name.split("-"):
                layers.append(pipe_module)
        module.layers = layers
        return module        
         
    @staticmethod
    def get_inner_strategies(num_points: int) -> List[InnerStrategyModule]:
        return [LeaderStrategyModule() for _ in range(num_points)]
    
    @staticmethod
    def build_fake_args(num_layers: int, embed_size: int, vocab_size: int) -> Callable:
        def _get_fake_args(mbatch_data: Dict) -> List[Any]:
            b, s = mbatch_data['input_ids'].size()
            return [((b, s, embed_size),)] +\
                [((b, s, embed_size),) for _ in range(num_layers)] +\
                    [((b, s, vocab_size),)] 
        
        return _get_fake_args
    
    
class TestMoePipeGenerator(TestPipeGenerator):
    def __new__(cls,
                module: TestModel,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                final_comm_group: ProcessGroup,
                embed_size: int,
                vocab_size: int,
                num_experts: int,
                k: int,
                scheduler: BaseScheduler,
                moe_group: ProcessGroup,
                device: torch.device) -> nn.Module:
        
        num_layers = len(module.layers)
        orig_modules = cls.get_orig_modules(module)
        num_modules_per_stage = len(orig_modules) // num_stages
        inner_boundary_points = [num_modules_per_stage * k - 1 for k in range(num_stages)][1:]
        moe_configs = cls.get_moe_layer_configs(orig_modules,
                                                num_experts,
                                                inner_boundary_points,
                                                moe_group)
        
        fake_modules = cls.get_fake_modules(num_layers, device)
        
        modules = [(name, orig_module, fake_module, is_moe)
                   for (name, orig_module, is_moe), fake_module in zip(orig_modules, fake_modules)]
        
        
        stage: nn.ModuleDict = MoePipelineGenerator.get_stage(modules=modules,
                                                              moe_configs=moe_configs,
                                                              k=k,
                                                              groups_info=groups_info,
                                                              inner_boundary_points=inner_boundary_points,
                                                              scheduler=scheduler,
                                                              device=device)
        module = cls.replace_modules(module, stage)
        modules = [module for module in stage.values()]
        fake_args = cls.build_fake_args(num_layers, embed_size, vocab_size)
        pipeline = MoePipelineGenerator(module,
                                        modules,
                                        inner_boundary_points,
                                        FinalStrategyDictModule(send_rank=num_stages-1),
                                        final_comm_group,
                                        fake_args,
                                        scheduler)
        return pipeline.to(device)
    
    @staticmethod
    def get_orig_modules(module: TestModel) -> List[Tuple[str, nn.Module, bool]]:
        return [("embed_tokens", module.embed_tokens, False)] +\
            [(f"layers-{idx}", layer, True) for idx, layer in enumerate(module.layers)] +\
                [('lm_head', module.lm_head, False)]
                
    @staticmethod
    def replace_modules(module: TestModel, stage: nn.ModuleDict) -> nn.Module:
        setattr(module, 'embed_tokens', stage['embed_tokens'])
        setattr(module, 'lm_head', stage['lm_head'])
        layers = nn.ModuleList()
        for name, moe_module in stage.items():
            if "layers" in name.split("-"):
                if type(moe_module) is MoeComputeLayer:
                    moe_module.module.moe = moe_module.moe
                layers.append(moe_module)
        module.layers = layers
        return module 
    
    @staticmethod
    def build_fake_args(num_layers: int, embed_size: int, vocab_size: int) -> Callable:
        def _get_fake_args(mbatch_data: Dict) -> List[Any]:
            b, s = mbatch_data['input_ids'].size()
            return [((b, s, embed_size),)] +\
                [((b,s,embed_size), (b, s, embed_size)) for _ in range(num_layers)] +\
                    [((b, s, vocab_size),)] 
        
        return _get_fake_args
                
                
    @staticmethod
    def get_moe_layer_configs(orig_modules: List[Tuple[str, nn.Module, bool]],
                              num_experts: int,
                              inner_boundary_points: List[int],
                              moe_group: ProcessGroup) -> Dict[str, MoeLayerConfig]:
        
        def _make_gate(gate_module: nn.Module, route_fn: Callable) -> Callable:
            return lambda hs: route_fn(gate_module(hs))
        
        main_rank = 0
        moe_layer_configs = {}
        for idx, (name, module, is_moe) in enumerate(orig_modules):
            if is_moe:
                expert_idxs = torch.arange(0, num_experts)
                expert_idxs = list(torch.split(expert_idxs, expert_idxs.size(0) // dist.get_world_size()))
                _gate = module.moe.gate.to(device=torch.cuda.current_device())
                _route = module.moe.route_tokens_to_experts
                gate = _make_gate(_gate, _route)
                next_main_rank = main_rank + 1 if idx in inner_boundary_points else main_rank
                moe_layer_configs[name] = MoeLayerConfig(module.moe, gate, main_rank, next_main_rank, expert_idxs, moe_group)
            
            if idx in inner_boundary_points:
                main_rank += 1
                    
        return moe_layer_configs
    
    
class TestMoeDPGenerator(TestPipeGenerator):
    def __new__(cls,
                module: TestModel,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                final_comm_group: ProcessGroup,
                embed_size: int,
                vocab_size: int,
                num_experts: int,
                k: int,
                scheduler: BaseScheduler,
                moe_group: ProcessGroup,
                device: torch.device) -> nn.Module:
        
        orig_modules = cls.get_orig_modules(module)
        moe_configs = cls.get_moe_layer_configs(orig_modules,
                                                num_experts,
                                                moe_group)
        for idx, (name, module, is_moe) in enumerate(moe_configs):
            if is_moe:
                moe_config: MoeLayerConfig = moe_configs[name]
                moe = MoeSparseBlockDPModuleGenerator(module=moe_config.module,
                                                      gate=moe_config.gate,
                                                      k=k,
                                                      main_rank=moe_config.main_rank,
                                                      next_main_rank=moe_config.next_main_rank,
                                                      replace_experts_layer=MoeDPExperts,
                                                      expert_idxs=moe_config.expert_idxs,
                                                      moe_group=moe_config.group,
                                                      scheduler=scheduler,
                                                      device=device)
                module.layers[idx].moe = moe
                
        def _forward(mbatches: List, **forward_kwargs):
            original_forward = module.forward
            mbatch = mbatches[0]
            mbatch.data.update(forward_kwargs)
            return original_forward(**mbatch)
        
        module.forward = _forward
            
        return module.to(device)
    
    @staticmethod
    def get_orig_modules(module: TestModel) -> List[Tuple[str, nn.Module, bool]]:
        return [("embed_tokens", module.embed_tokens, False)] +\
            [(f"layers-{idx}", layer, True) for idx, layer in enumerate(module.layers)] +\
                [('lm_head', module.lm_head, False)]
                
                
    @staticmethod
    def get_moe_layer_configs(orig_modules: List[nn.Module],
                              num_experts: int,
                              moe_group: ProcessGroup) -> Dict[str, MoeLayerConfig]:
        
        def _make_gate(gate_module: nn.Module, route_fn: Callable) -> Callable:
            return lambda hs: route_fn(gate_module(hs))
        
        moe_layer_configs = {}
        for name, module, is_moe in orig_modules:
            if is_moe:
                expert_idxs = torch.arange(0, num_experts)
                expert_idxs = list(torch.split(expert_idxs, expert_idxs.size(0) // dist.get_world_size()))
                _gate = module.moe.gate.to(device=torch.cuda.current_device())
                _route = module.moe.route_tokens_to_experts
                gate = _make_gate(_gate, _route)
                moe_layer_configs[name] = MoeLayerConfig(module.moe, gate, 0, 0, expert_idxs, moe_group)
                    
        return moe_layer_configs
                
    