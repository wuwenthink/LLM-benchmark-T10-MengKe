# -*- coding: utf-8 -*-
"""评测期间的「静默闸门」占位实现。

评测跑满并发时, 同一个进程里若有其它后台轮询(监控/统计/上报), 会与被测端点抢
带宽与连接数。本模块提供一个统一的开合钩子: 评测开始 set_active(True)、结束
set_active(False); 单机运行没有任何后台轮询, 因此这里是空实现。

需要时可在你自己的部署里替换本文件, 例如暂停 Prometheus 抓取、停掉日志上报等。
"""


def set_active(active: bool = True, **kwargs):
    """评测开始/结束时被调用; 返回值不参与评测逻辑。"""
    return None
