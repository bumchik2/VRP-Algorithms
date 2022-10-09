import json


def read_json(json_filename):
    with open(json_filename, 'r') as f:
        return json.loads(f.read())
