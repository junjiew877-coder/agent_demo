from datetime import datetime

def get_current_time() -> str:
    """从本机系统读取当前本地日期与时间（不依赖模型知识）。"""
    now = datetime.now()
    weekdays = "一二三四五六日"
    return (
        f"当前系统本地时间: {now.strftime('%Y-%m-%d %H:%M:%S')} "
        f"（星期{weekdays[now.weekday()]}）"
    )