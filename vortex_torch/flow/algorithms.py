import torch
from typing import Dict

from .flow import vFlow
from ..indexer import topK, GeMV, Softmax, Max, Sum, GeMM, Maximum, Multiply
from ..cache import Mean as CMean, Max as CMax, Min as CMin
from ..abs import ContextBase
from .registry import register

@register("block_sparse_attention")
class BlockSparseAttention(vFlow):
    
    def __init__(self):
        super().__init__()
        #indexer ops
        self.gemv = GeMV()
        self.output_func = topK()

        #cache ops
        self.reduction = CMean(dim=1)
    
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
        #indexer ops
        self.gemm = GeMM()
        self.softmax = Softmax(dim=0, scale=0.09)
        self.max_op = Max(dim=2)
        self.output_func = topK()

        #cache ops
        self.reduction = CMean(dim=1)
    
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



@register("gqa_quest_sparse_attention")
class GQAQuestSparseAttention(vFlow):
    
    def __init__(self):
        super().__init__()
        
        #indexer ops
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.maximum_op = Maximum()
        self.sum = Sum(dim=2)
        self.max_op = Max(dim=1)
        self.output_func = topK()

        #cache ops
        self.reduction_max = CMax(dim=1)
        self.reduction_min = CMin(dim=1)
    
    def forward_indexer(self, q, o, cache, ctx):
        
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s = self.maximum_op(s_max, s_min, ctx=ctx)
        score = self.sum(s, ctx=ctx)
        aggr_score = self.max_op(score, ctx=ctx)
        self.output_func(aggr_score, o, ctx=ctx)
            
    def forward_cache(self, cache: Dict[str, torch.Tensor], loc:torch.Tensor, ctx: ContextBase):
        
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

     
     
    def create_cache(self, page_size: int, head_dim: int):
         
         return {
             "max": (1, head_dim),
             "min": (1, head_dim)
         }