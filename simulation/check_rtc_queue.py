"""Independent queue tests with delayed completion and a finite old plan."""
from .rtc_queue import DelayEstimator, consume_while_sampling


def main():
    estimator = DelayEstimator(initial=12, window=3, margin=1)
    assert estimator.estimate(40) == 13
    for delay in (4, 9, 6):
        estimator.observe(delay)
    assert estimator.estimate(40) == 10
    estimator.observe(3)
    estimator.observe(2)
    assert estimator.estimate(40) == 7
    assert estimator.estimate(2) == 2

    bounded = DelayEstimator(initial=12, window=3, margin=1, maximum=12)
    assert bounded.estimate(25) == 12
    bounded.observe(18)
    assert bounded.estimate(25) == 12
    assert bounded.estimate(4) == 4

    class Completion:
        def __init__(self, commands):
            self.remaining = commands
        def done(self):
            return self.remaining <= 0

    for needed, start, expected, exhausted in [(4, 10, 4, False), (8, 47, 3, True), (0, 10, 0, False)]:
        future = Completion(needed)
        executed = []
        def execute(plan, index):
            executed.append(plan["actions"][index])
            future.remaining -= 1
        consumed, actual_exhausted = consume_while_sampling(
            future, {"actions": list(range(50))}, start, execute, lambda: False)
        assert consumed == expected and actual_exhausted == exhausted
        assert executed == list(range(start, start+expected))
        # A new plan is indexed by commands consumed since its observation.
        new = list(range(100, 150))
        assert new[consumed] == 100+expected
    print("RTC delay forecast, continuous consumption, and prefix alignment: passed")


if __name__ == "__main__":
    main()
