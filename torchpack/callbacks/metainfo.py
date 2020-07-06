import os.path as osp

from ..environ import get_run_dir
from ..utils import fs, git, io
from ..utils.config import configs
from .callback import Callback

__all__ = ['MetaInfoSaver']


class MetaInfoSaver(Callback):
    master_only = True

    def __init__(self, save_dir=None):
        if save_dir is None:
            save_dir = osp.join(get_run_dir(), 'metainfo')
        self.save_dir = fs.normpath(save_dir)

    def _before_train(self):
        if configs:
            io.save(osp.join(self.save_dir, 'configs.yaml'), configs.dict())

        if git.is_inside_work_tree():
            metainfo = dict()
            remote = git.get_remote_url()
            if remote:
                metainfo['remote'] = remote
            metainfo['commit'] = git.get_commit_hash()
            io.save(osp.join(self.save_dir, 'git.json'), metainfo, indent=4)