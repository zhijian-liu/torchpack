import subprocess

__all__ = ['is_inside_work_tree', 'get_commit_hash', 'get_remote_url']


def is_inside_work_tree():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            stderr=subprocess.DEVNULL).decode('utf-8').strip() == 'true'
    except subprocess.CalledProcessError:
        return False


def get_commit_hash(revision='HEAD'):
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', revision],
            stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None


def get_remote_url(name='origin'):
    try:
        return subprocess.check_output(
            ['git', 'config', '--get', f'remote.{name}.url'],
            stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None
