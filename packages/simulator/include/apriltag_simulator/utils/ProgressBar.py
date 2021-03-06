import sys
import math


class ProgressBar:

    def __init__(self, scale=1.0, buf=sys.stdout, header='Progress'):
        self._finished = False
        self._buffer = buf
        self._header = header
        self._last_value = -1
        self._scale = max(0.0, min(1.0, scale))
        self._max = int(math.ceil(100 * self._scale))

    def update(self, percentage):
        percentage_int = int(max(0, min(100, percentage)))
        if percentage_int == self._last_value:
            return
        percentage = int(math.ceil(percentage * self._scale))
        if self._finished:
            return
        # compile progress bar
        pbar = "%s: [" % self._header if self._scale > 0.5 else "["
        # progress
        pbar += "=" * percentage
        if percentage < self._max:
            pbar += ">"
        pbar += " " * (self._max - percentage - 1)
        # this ends the progress bar
        pbar += "] {:d}%".format(percentage_int)
        # print
        self._buffer.write(pbar)
        self._buffer.flush()
        # return to start of line
        self._buffer.write("\b" * len(pbar))
        # end progress bar
        if percentage >= self._max:
            self._buffer.write("\n")
            self._buffer.flush()
            self._finished = True
        self._last_value = percentage_int

    def done(self):
        self.update(100)


class CountingProgressBar(ProgressBar):

    def __init__(self, total, scale=1.0, buf=sys.stdout, header='Progress'):
        super(CountingProgressBar, self).__init__(scale, buf, header)
        self._count = 0
        self._total = total

    def tick(self):
        self.update(self._count + 1)

    def update(self, count):
        self._count = count
        perc = max(0, min(100, int((count / self._total) * 100)))
        super(CountingProgressBar, self).update(perc)
