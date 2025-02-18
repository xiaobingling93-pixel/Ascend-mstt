import os
import pickle
import sys
import logging
from collections import defaultdict

# 配置 logging
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
    handlers=[logging.StreamHandler()]  # 输出到控制台
)

# 定义允许的类白名单
SAFE_CLASSES = {
    # 内置安全类型
    "builtins": {"str", "int", "float", "list", "dict", "tuple"},
}

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 检查模块和类是否在白名单中
        if module in SAFE_CLASSES and name in SAFE_CLASSES[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden class: {module}.{name}")

def load_recorder_data(path, world_size):
    """加载所有 rank 的 recorder 数据"""
    recorder_dict = {}
    for rank in range(world_size):
        file_path = os.path.join(path, str(rank))
        try:
            with open(file_path, "rb") as f:
                res = SafeUnpickler(f).load()
                recorder_dict[str(rank)] = res
        except Exception as e:
            logging.error(f"Failed to load data from {file_path}: {e}")
    return recorder_dict

def extract_hccl_info(recorder_dict):
    """从 recorder 数据中提取 HCCL 相关信息"""
    hccl_dict = {}
    for rank, recorder in recorder_dict.items():
        last_entry = recorder["entries"][-1]
        second_last_entry = recorder["entries"][-2]
        hccl_dict[rank] = {
            'state': last_entry['state'],
            'record_id': last_entry['record_id'],
            'pg_id': last_entry['pg_id'],
            'time_discovered_completed_ns': last_entry['time_discovered_completed_ns'],
            'last_name': second_last_entry["frames"][0]['name'],
            'name': last_entry["frames"][0]['name']
        }
    return hccl_dict

def analyze_pg_groups(hccl_dict):
    """分析 HCCL 数据，按 pg_id 分组并检查问题"""
    pg_groups = defaultdict(list)
    for key, op in hccl_dict.items():
        pg_groups[op['pg_id']].append(op)

    for pg_id, group in pg_groups.items():
        scheduled_ops = [op for op in group if op['state'] == 'scheduled']
        completed_ops = [op for op in group if op['state'] == 'completed']

        # 情况 1: 所有卡都是 scheduled，且 record_id 和 name 相同
        if len(scheduled_ops) == len(group):
            record_id = scheduled_ops[0]['record_id']
            name = scheduled_ops[0]['name']
            all_same = all(op['record_id'] == record_id and op['name'] == name for op in scheduled_ops)
            if all_same:
                logging.info(f"通信域 {pg_id} 的通信算子 {name} 执行过慢，导致通信超时")

        # 情况 2: 存在 completed 算子且 record_id 比其他 scheduled 算子少 1
        if completed_ops and scheduled_ops:
            completed_op = completed_ops[0]
            scheduled_record_id = scheduled_ops[0]['record_id']
            if completed_op['record_id'] == scheduled_record_id - 1:
                logging.info(f"通信域 {pg_id} 第 {completed_op['pg_id']} 卡计算超时，导致其他卡等待触发飞行记录器和 HCCL 超时")

        # 情况 3: 所有算子均为 completed，且取最新的 time_discovered_completed_ns
        if not scheduled_ops and completed_ops:
            latest_op = max(completed_ops, key=lambda x: x['time_discovered_completed_ns'] or 0)
            logging.info(f"通信域 {pg_id} 的通信算子 {latest_op['name']} 的非通信任务耗时过长")

def main():
    if len(sys.argv) < 2:
        logging.error("Usage: python analysis_flight.py <path>")
        sys.exit(1)

    path = sys.argv[1]
    world_size = 8

    # 加载数据
    recorder_dict = load_recorder_data(path, world_size)
    if not recorder_dict:
        logging.error("No valid recorder data found.")
        return

    # 提取 HCCL 信息
    hccl_dict = extract_hccl_info(recorder_dict)

    # 分析 HCCL 数据
    analyze_pg_groups(hccl_dict)

if __name__ == "__main__":
    main()