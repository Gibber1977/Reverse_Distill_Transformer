#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行RDT模型并将所有输出保存到output.log文件
"""

import os
import sys
import contextlib
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# 设置matplotlib后端为Agg，这样即使没有GUI也能生成图像
matplotlib.use('Agg')

# 创建一个类来捕获标准输出和标准错误
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 写入日志头部信息
        self.log.write(f"===== RDT模型运行日志 - {self.timestamp} =====\n\n")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 保存图表的原始方法
original_savefig = plt.savefig
original_show = plt.show

# 重写plt.savefig方法，使其同时保存到日志文件中
def custom_savefig(fname, *args, **kwargs):
    # 调用原始的savefig方法
    original_savefig(fname, *args, **kwargs)
    
    # 在日志中记录图表保存信息
    sys.stdout.write(f"\n[图表已保存] {fname}\n")

# 重写plt.show方法，确保图表被保存
def custom_show(*args, **kwargs):
    # 获取当前图表
    fig = plt.gcf()
    
    # 生成唯一的图表文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fig_filename = f"figure_{timestamp}.png"
    
    # 保存图表
    original_savefig(fig_filename)
    
    # 在日志中记录图表信息
    sys.stdout.write(f"\n[图表已生成] {fig_filename}\n")
    
    # 调用原始的show方法
    original_show(*args, **kwargs)

def main():
    # 设置输出重定向到日志文件
    log_file = "output.log"
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    
    # 重写matplotlib的savefig和show方法
    plt.savefig = custom_savefig
    plt.show = custom_show
    
    try:
        # 记录开始运行的信息
        print("开始运行RDT模型...\n")
        
        # 导入并运行主程序
        from main import main as run_rdt
        run_rdt()
        
        print("\n运行完成，所有输出已保存到", log_file)
        
    except Exception as e:
        print(f"\n运行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复标准输出和标准错误
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        # 恢复原始的matplotlib方法
        plt.savefig = original_savefig
        plt.show = original_show
        
        print(f"日志已保存到 {log_file}")

if __name__ == "__main__":
    main()