import re
import yaml
from msprobe.core.common.file_utils import FileOpen
from msprobe.core.common.const import Const
from msprobe.visualization.utils import GraphConst


class MappingConfig:
    MAX_STRING_LEN = 10000

    def __init__(self, yaml_file):
        with FileOpen(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        try:
            self.config = {key: self.validate(key, value) for data in config for key, value in data.items()}
        except Exception as e:
            raise RuntimeError("Line of yaml contains content that is not '- key: value'.") from e
        self.classify_config = self._classify_and_sort_keys()

    @staticmethod
    def validate(key, value):
        if not isinstance(key, str):
            raise ValueError(f"{key} must be a string.")
        if not isinstance(value, str):
            raise ValueError(f"{value} must be a string.")
        return value

    @staticmethod
    def convert_to_regex(s):
        """
        字符串转换为正则表达式, {}替换为d+以匹配一个或多个数字, 开始和结束添加.*以匹配任意前缀和后缀
        Args:
            s: 字符串
        Returns: 正则表达式
        """
        escaped_pattern = re.escape(s)
        pattern = re.sub(r'\\\{\\\}', r'\\d+', escaped_pattern)
        pattern = f'.*{pattern}.*'
        return pattern

    @staticmethod
    def _replace_parts(origin_string, mapping_key, mapping_value):
        if GraphConst.BRACE in mapping_key:
            parts = mapping_key.split(GraphConst.BRACE)
            m_parts = mapping_value.split(GraphConst.BRACE)
            return origin_string.replace(parts[0], m_parts[0]).replace(parts[1], m_parts[1])
        else:
            return origin_string.replace(mapping_key, mapping_value)

    def get_mapping_string(self, origin_string: str):
        if len(origin_string) > MappingConfig.MAX_STRING_LEN:
            return origin_string
        for category, items in self.classify_config.items():
            if category in origin_string:
                for key, value in items:
                    if re.match(MappingConfig.convert_to_regex(key), origin_string):
                        return MappingConfig._replace_parts(origin_string, key, value)
        return origin_string

    def _classify_and_sort_keys(self):
        categorized_dict = {}
        for key, value in self.config.items():
            parts = key.split(Const.SEP)
            # 获取第一个部分作为新的分类key
            category_key = parts[0]

            if category_key not in categorized_dict:
                categorized_dict[category_key] = []

            # 将原始的key-value对添加到对应的分类中
            categorized_dict[category_key].append((key, value))

        # 对每个分类中的项按key中的.数量进行排序, .数量越多排越靠前, 优先匹配
        for category in categorized_dict:
            categorized_dict[category].sort(key=lambda x: -x[0].count(Const.SEP))

        return categorized_dict
