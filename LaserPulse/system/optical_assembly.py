"""
==============================================================================
文件名称: optical_assembly.py
所属部门: System (系统总控室)
主要功能: 光学总成与物理仿真循环 (流水线传送带)
辅食解读: 
    这是工厂的“传送带控制台”。
    它负责把车间里的反射镜、晶体按顺序排好（组成一个谐振腔），
    然后命令脉冲在里面一圈一圈地跑（Round Trips）。
    跑一圈记录一次能量，直到完成我们指定的圈数。
==============================================================================
"""

from components.base_component import BaseComponent
from core.pulse import Pulse
import numpy as np

class RegenerativeAmplifierAssembly:
    """
    再生放大器总装配体
    负责管理腔内零件的排列顺序，并控制脉冲在其中的往返循环。
    """
    def __init__(self, name: str, cavity_components: list[BaseComponent]):
        self.name = name
        # 腔内元件列表（脉冲将严格按照这个列表的顺序依次穿过）
        self.components = cavity_components
        
        # 数据记录本：记录每一次往返后的物理状态
        self.history = {
            'round_trip': [],
            'pulse_energy': [],
        }

    def simulate(self, seed_pulse: Pulse, num_round_trips: int) -> Pulse:
        """
        启动数字仿真：让种子光在腔内循环指定的圈数
        """
        print(f"\n🚀 开始运行 [{self.name} 脚本]，设定往返次数: {num_round_trips}")
        
        # 记录第 0 圈（初始状态）的能量
        current_energy = seed_pulse.get_energy()
        self.history['round_trip'].append(0)
        self.history['pulse_energy'].append(current_energy)
        print(f"初始化 - 注入种子光能量: {current_energy:.2e} J")

        # 核心循环：脉冲在腔内一圈圈地跑
        for rt in range(1, num_round_trips + 1):
            # 遍历腔内的每一个零件
            for component in self.components:
                # 脉冲穿过当前零件，发生物理状态改变
                seed_pulse = component.propagate(seed_pulse)
            
            # 一圈跑完，记录当前圈数和能量
            current_energy = seed_pulse.get_energy()
            self.history['round_trip'].append(rt)
            self.history['pulse_energy'].append(current_energy)
            
            # 打印少量日志，方便观察（每 10 圈打印一次）
            if rt % 10 == 0 or rt == num_round_trips:
                print(f"   -> 已跑完 {rt:03d} 圈 | 当前脉冲能量为: {current_energy:.2e} J")
                
        print(f"🏁 仿真结束！最终输出能量: {current_energy:.2e} J\n")
        return seed_pulse