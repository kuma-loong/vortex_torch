from abc import ABC, abstractmethod
from ..context_base import ContextBase
import torch
from typing import Dict, Tuple

class vFlow(ABC):
    def __init__(self):
        super().__init__()
        
    
    @abstractmethod
    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        pass
    
    
    @abstractmethod
    def forward_cache(self, cache: Dict[str, torch.Tensor], loc:torch.Tensor, ctx: ContextBase):
        pass
    
    
    @abstractmethod
    def create_cache(self, page_size: int, head_dim: int)->Dict[str, Tuple[Tuple[int, int]]]:
        pass
    
    
    def get_cache_meta_info(self, page_size: int, head_dim: int)->Dict[str, Tuple[Tuple[int, int]]]:
        
        cache_meta_info = self.create_cache(page_size, head_dim)
        assert "k" not in cache_meta_info.keys()
        assert "v" not in cache_meta_info.keys()
        cache_meta_info["k"] = (page_size, head_dim)
        cache_meta_info["v"] = (page_size, head_dim)
        
        return cache_meta_info
    
    
    def get_token_ratio(self, page_size:int, head_dim: int):
        
        token_ratio = 0.0
        for (_, cache_shape) in self.get_cache_meta_info(page_size, head_dim).items():
            token_ratio += (cache_shape[0] * cache_shape[1]) / (page_size * head_dim)
        
        return token_ratio