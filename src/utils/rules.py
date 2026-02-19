from typing import List


class RuleBase:
    def check(self, obj) -> bool:
        raise NotImplementedError()

# controlla se qualcosa è contenuto in qualcos'altro
class ContainRule(RuleBase):
    def __init__(self, target, reverse=False) -> None:
        """
         Check if target is in obj.
        `reverse`: if True, check if obj is in target
        """
        self.target = target
        self.reverse = reverse

    def check(self, obj) -> bool:
        if self.reverse:
            return obj in self.target
        else:
            return self.target in obj

# negazione logica
class NotRule(RuleBase):
    def __init__(self, rule: RuleBase) -> None:
        self.rule = rule

    def check(self, obj) -> bool:
        return not self.rule.check(obj)

# accetta lista di regole, tutte devono essere vere
class AndRule(RuleBase):
    def __init__(self, rules: List[RuleBase]) -> None:
        self.rules = rules

    def check(self, obj) -> bool:
        return all(rule.check(obj) for rule in self.rules)

# almeno una regola deve essere vera
class OrRule(RuleBase):
    def __init__(self, rules: List[RuleBase]) -> None:
        self.rules = rules

    def check(self, obj) -> bool:
        return any(rule.check(obj) for rule in self.rules)