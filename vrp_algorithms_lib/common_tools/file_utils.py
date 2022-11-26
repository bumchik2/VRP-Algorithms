import json


def read_json(json_filename):
    with open(json_filename, 'r') as f:
        return json.loads(f.read())


def save_json(json_object, json_filename, indent=4):
    with open(json_filename, 'w') as f:
        json.dump(json_object, f, indent=indent)
