from models.generator import Generator
from thop import profile
import torch
import torch.nn as nn
model = Generator(80)

input = torch.randn(1, 80, 80)
out = model(input)
print(out.shape)
with torch.cuda.device(0):
   flops, params = profile(model, inputs=(input, ))
   print(flops/1e9,params/1e6)
