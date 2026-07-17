import os


def get_dataset_path(name):
    for path in [f'data/valid']:
        if os.path.exists(path):
            return path