from .flow import vFlow
from ..indexer import topK, GeMV
from ..cache import Mean
import torch
from ..context_base import ContextBase
from .registry import register
from typing import Dict

@register("block_sparse_attention")
class BlockSparseAttention(vFlow):
    
    def __init__(self):
        super().__init__()
        
        self.gemv = GeMV()
        self.output_func = topK()
        self.reduction = Mean(dim=1)
    
    def forward_indexer(self, q, o, cache, ctx):
        
        q_mean = q.mean(dim=1, keepdim=True)
        score = self.gemv(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)
            
    def forward_cache(self, cache: Dict[str, torch.Tensor], loc:torch.Tensor, ctx: ContextBase):
        
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
     
     
    def create_cache(self, page_size: int, head_dim: int):
         
         return {
             "centroids": (1, head_dim)
         }