import re

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger

__all__ = ['ScalarPrinter']


class ScalarPrinter(Callback):
    """
    Print scalar data into terminal.
    """

    def __init__(self, regexes=None, blacklist=None):
        """
        Args:
            regex (list[str] or None): A list of regex. Only names
                matching some regex will be allowed for printing.
                Defaults to match all names.
            blacklist (list[str] or None): A list of regex. Names matching
                any regex will not be printed. Defaults to match no names.
        """
        self.regexes = [re.compile(regex) for regex in regexes]

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        texts = []
        for k in sorted(self.trainer.monitors.keys()):
            if not any(regex.match(k) for regex in self.regexes):
                continue
            _, v = self.trainer.monitors[k]
            texts.append('[{}] = {:.5g}'.format(k, v))
        if texts:
            logger.info('\n+ '.join([''] + texts))
