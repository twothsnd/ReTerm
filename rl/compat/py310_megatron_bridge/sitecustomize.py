"""Python 3.10 compatibility for Megatron-Bridge's Python 3.12 typing imports."""

import typing

import typing_extensions

for _name in (
    "LiteralString",
    "Never",
    "NotRequired",
    "Required",
    "Self",
    "TypeAlias",
    "Unpack",
    "override",
):
    if not hasattr(typing, _name) and hasattr(typing_extensions, _name):
        setattr(typing, _name, getattr(typing_extensions, _name))
