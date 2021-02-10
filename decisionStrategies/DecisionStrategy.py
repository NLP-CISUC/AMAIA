

class DecisionStrategy:

    def __init__(self):
        None

    def sort_answers(self, votes):
        return self.majority_sort(votes)

    def getNAnswers(self, answers, n):
        winner = self.sort_answers(answers)
        return winner[0:n] if len(winner) >= n else winner

    def majority_sort(votes):
        scores = {}
        for elem in votes:
            if not elem in scores:
                scores[elem] = 1
            else:
                scores[elem] = scores[elem] + 1

        return sorted(scores.keys(), key=lambda elem: scores[elem], reverse=True)


