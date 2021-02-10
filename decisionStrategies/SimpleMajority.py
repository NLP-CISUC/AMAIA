# to import file from main directory
from decisionStrategies.DecisionStrategy import DecisionStrategy


class SimpleMajority(DecisionStrategy):
    # Given a set of answers, delivers the most frequent answer to the user

    def __init__(self):
        super().__init__()
