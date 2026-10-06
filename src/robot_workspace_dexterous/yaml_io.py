"""Safe YAML parsing with the optional LibYAML accelerator."""
import yaml


def load_yaml(stream):
    return yaml.load(stream, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
