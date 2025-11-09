import torch
from typing import Dict

from .flow import vFlow
from ..indexer import topK, GeMV, Softmax, Max, GeMM
from ..cache import Mean
from ..abs import ContextBase
from .registry import register

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


@register("gqa_block_sparse_attention")
class GQABlockSparseAttention(vFlow):
    
    def __init__(self):
        super().__init__()
        
        self.gemm = GeMM()
        self.softmax = Softmax(dim=0, scale=0.09)
        self.max_op = Max(dim=2)
        self.output_func = topK()
        self.reduction = Mean(dim=1)
    
    def forward_indexer(self, q, o, cache, ctx):
        
        score = self.gemm(q, cache["centroids"], ctx=ctx)
        self.softmax(score, ctx=ctx)
        aggr_score = self.max_op(score, ctx=ctx)
        self.output_func(aggr_score, o, ctx=ctx)
            
    def forward_cache(self, cache: Dict[str, torch.Tensor], loc:torch.Tensor, ctx: ContextBase):
        
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
     
     
    def create_cache(self, page_size: int, head_dim: int):
         
         return {
             "centroids": (1, head_dim)
         }
         