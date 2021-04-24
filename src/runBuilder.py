class RunBuilder():
    @staticmethod
    def get_runs(params):
        from collections import namedtuple
        from itertools import product

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
