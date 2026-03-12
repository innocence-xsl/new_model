"""
==============================================================================
文件名称: base_component.py
所属部门: Components (加工车间)
主要功能: 光学器件的抽象基类 (机器蓝图)
辅食解读: 
    这是所有加工机器的“通用模版”。
    它规定了：以后不管在这个厂里造什么新机器（光栅、透镜、新晶体），
    都必须带一个叫 `propagate`（加工）的按钮。
    流水线只认这个按钮，按下去，脉冲就发生变化。
==============================================================================
"""

from core.pulse import Pulse

class BaseComponent:
    """
    所有光学器件的抽象基类 (蓝图)
    """
    def __init__(self, name="Unnamed Component"):
        self.name = name

    def propagate(self, pulse: Pulse) -> Pulse:
        """
        核心物理加工方法。
        脉冲传入该器件后，发生物理变化（损耗、色散、放大等）。
        必须在子类中被重写！
        """
        raise NotImplementedError(f"[{self.name}] 必须在子类中实现 propagate 方法！")