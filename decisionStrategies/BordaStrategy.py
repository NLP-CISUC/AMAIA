'''
import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.append(current_dir)
'''
from decisionStrategies.DecisionStrategy import DecisionStrategy

'''
Borda Count decision method. Method logic in https://en.wikipedia.org/wiki/Borda_count
'''
class BordaStrategy(DecisionStrategy):

    def __init__(self):
        super().__init__()

    def sort_answers(self, votes):
        scores = {}
        for l in votes:
            for idx, elem in enumerate(reversed(l)):
                if not elem in scores:
                    scores[elem] = 0
                scores[elem] += idx

        return sorted(scores.keys(), key=lambda elem: scores[elem], reverse=True)