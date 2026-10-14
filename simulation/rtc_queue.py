"""Command-count delay forecasting and action queue advancement for RTC."""
from collections import deque


class DelayEstimator:
    def __init__(self, initial=12, window=5, margin=1, maximum=None):
        if initial < 0 or window < 1 or margin < 0 or (maximum is not None and maximum < 1):
            raise ValueError("Invalid delay estimator")
        self.initial, self.margin = initial, margin
        self.maximum = maximum
        self.history = deque(maxlen=window)

    def estimate(self, available):
        if available < 1:
            raise ValueError("No unconsumed previous actions")
        forecast = max(self.history) if self.history else self.initial
        prediction = max(1, forecast + self.margin)
        if self.maximum is not None:
            prediction = min(prediction, self.maximum)
        return min(available, prediction)

    def observe(self, consumed):
        if consumed < 0:
            raise ValueError("Consumed actions cannot be negative")
        self.history.append(consumed)


def consume_while_sampling(future, plan, cursor, execute, done):
    """Keep the old plan moving; return exact consumption and queue exhaustion.

    The caller swaps atomically on its simulation thread and skips exactly the
    consumed prefix in the new plan. No simulation state enters inference here.
    """
    start = cursor
    length = len(plan["actions"])
    while not future.done() and cursor < length and not done():
        execute(plan, cursor)
        cursor += 1
    exhausted = cursor == length and not future.done() and not done()
    return cursor-start, exhausted
