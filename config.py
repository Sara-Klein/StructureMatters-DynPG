from dataclasses import dataclass
import typing
from typing import Iterable
import numpy as np
import math

@dataclass
class _AlgosConfig():
    algorithm: Iterable[Iterable[str]] = ("pg","dpg",) # dpg or pp
    env_name: Iterable[str] = ("Sutton-extended",)
    action_list: Iterable[Iterable[int]] = ((0, 300, 3, 0, 5, 5, 0),)
    reward_list: Iterable[Iterable[Iterable[float]]] = (((0,0),(-0.75,10),(0,0),(0,0),(1.25,1.25),(1.25,1.25)),)
    discount_factor: Iterable[float] = (0.99,)
    
    eps: Iterable[float] = (0.01,) # accuracy

    pg_lr: Iterable[float] = (0.3,) # learningrate for PG
    dpg_lr: Iterable[Iterable[float]] = ((1.5, 1., 0.7, 0.5, 0.4, 0.3),) #learningrates for DynPG
    adaptive: Iterable[bool] = (True,) # then stepsizes are choosen acording to the theory in the DynPG paper
    dpg_Ns: Iterable[Iterable[float]] = ((50, 100, 150, 200, 250, 300),) # number of training steps for DynPG
    
    computational_power: Iterable[int] = (1200,)#

    
    gradient_clipping :Iterable[bool] = (False,)
    reward_normalization :Iterable[bool] = (False,)
    copy_weights :Iterable[bool] = (False,)

    rounds: Iterable[int] = (1000,)

TestConfig = _AlgosConfig()