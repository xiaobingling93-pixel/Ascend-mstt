import json
import ast

def parse_value(value):
    try:
        parsed_value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed_value = value
    return parsed_value

def convert_json(json_obj):
    if isinstance(json_obj, dict):
        return {k: convert_json(v) for k, v in json_obj.items()}
    elif isinstance(json_obj, list):
        return [convert_json(elem) for elem in json_obj]
    elif isinstance(json_obj, str):
        return parse_value(json_obj)
    else:
        return json_obj
