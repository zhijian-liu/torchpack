import os
from typing import Optional

from torchpack.callbacks.callback import Callback
from torchpack.environ import get_run_dir
from torchpack.utils import fs, git, io
from torchpack.utils.config import configs

__all__ = ['MetaInfoSaver']


class MetaInfoSaver(Callback):
    master_only: bool = True

    def __init__(self, save_dir: Optional[str] = None) -> None:
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'metainfo')
        self.save_dir = fs.normpath(save_dir)

    def _before_train(self) -> None:
        if configs:
            io.save(os.path.join(self.save_dir, 'configs.yaml'),
                    configs.dict())

        if git.is_inside_work_tree():
            metainfo = dict()
            remote = git.get_remote_url()
            if remote:
                metainfo['remote'] = remote
            metainfo['commit'] = git.get_commit_hash()
            io.save(os.path.join(self.save_dir, 'git.json'),
                    metainfo,
                    indent=4)
