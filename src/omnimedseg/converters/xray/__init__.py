import importlib
import os
import inspect

CONVERTERS = {}

converter_dir = os.path.dirname(__file__)
for fname in os.listdir(converter_dir):
    if fname.endswith(".py") and fname != "__init__.py":
        module = importlib.import_module(f".{fname[:-3]}", package=__name__)

        # Look for a function starting with "convert_"
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("convert_"):
                dataset_name = name.replace("convert_", "").upper()
                CONVERTERS[dataset_name] = func
