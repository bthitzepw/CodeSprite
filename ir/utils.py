from typing import Dict, Any
from ir.layers import Layer


def collect_params(obj, prefix: str = "") -> Dict[str, Any]:
    """递归收集 IR 模型的所有参数"""
    state = {}
    _collect_params_recursive(obj, prefix, state)
    return state


def _collect_params_recursive(obj, prefix: str, state: Dict[str, Any]):
    if isinstance(obj, Layer):
        for key, param in obj.params.items():
            full_key = f"{prefix}{key}" if prefix else key
            state[full_key] = param
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(obj, attr_name)
                if isinstance(attr, Layer):
                    sub_prefix = f"{prefix}{attr.name}." if prefix else f"{attr.name}."
                    _collect_params_recursive(attr, sub_prefix, state)
            except AttributeError:
                pass
        if hasattr(obj, 'blocks'):
            for i, block in enumerate(obj.blocks):
                sub_prefix = f"{prefix}block_{i}." if prefix else f"block_{i}."
                _collect_params_recursive(block, sub_prefix, state)


def load_state_dict(obj, state_dict: Dict[str, Any], prefix: str = "",
                    transform_fn=None):
    """从 flat key 字典加载参数到 IR 模型

    Args:
        transform_fn: 可选的参数转换函数，接收 (key, value) 返回转换后的值
    """
    _load_state_dict_recursive(obj, state_dict, prefix, transform_fn)


def _load_state_dict_recursive(obj, state_dict: Dict[str, Any], prefix: str,
                               transform_fn):
    if isinstance(obj, Layer):
        for key in list(obj.params.keys()):
            full_key = f"{prefix}{key}" if prefix else key
            if full_key in state_dict:
                value = state_dict[full_key]
                if transform_fn is not None:
                    value = transform_fn(full_key, value)
                obj.params[key] = value
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(obj, attr_name)
                if isinstance(attr, Layer):
                    sub_prefix = f"{prefix}{attr.name}." if prefix else f"{attr.name}."
                    _load_state_dict_recursive(attr, state_dict, sub_prefix, transform_fn)
            except AttributeError:
                pass
        if hasattr(obj, 'blocks'):
            for i, block in enumerate(obj.blocks):
                sub_prefix = f"{prefix}block_{i}." if prefix else f"block_{i}."
                _load_state_dict_recursive(block, state_dict, sub_prefix, transform_fn)
