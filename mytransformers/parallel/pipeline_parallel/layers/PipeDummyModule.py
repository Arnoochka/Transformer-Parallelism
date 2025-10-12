import torch
from torch import Tensor
from mytransformers.parallel.pipeline_parallel.fake_output_generators import FakeGenerator
from .PipeModule import PipeModule, PipeRole

class PipeDummyModule(PipeModule):
    def __init__(self, fake_generator: FakeGenerator):
        super().__init__(PipeRole.dummy)
        self.fake_generator = fake_generator
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Tensor:
        return self.fake_generator()