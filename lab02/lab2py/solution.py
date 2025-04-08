import sys
from collections import deque


class Clause:
    id_counter = 1

    def __init__(self, literals, parents=None):
        self.literals = literals
        self.parents = parents
        self.id = Clause.id_counter
        Clause.id_counter += 1

    def __hash__(self):
        return hash(self.literals)

    def __eq__(self, other):
        return isinstance(other, Clause) and self.literals == other.literals

    def __str__(self):
        retVal = str()
        retVal += str(self.id)
        retVal += ". "
        for index, val in enumerate(self.literals):
            if index == len(self.literals) - 1:
                retVal += str(val)
                retVal += " "
            else:
                retVal += str(val)
                retVal += " v "
        if self.parents is not None:
            retVal += "("
            retVal += str(self.parents[0].id)
            retVal += ", "
            retVal += str(self.parents[1].id)
            retVal += ")"
        return retVal


def negation(literal):
    literal = "~" + literal
    while literal[:2] == "~~":
        literal = literal[2:]
    return literal


def tautology(clause):
    return any(negation(literal) in clause for literal in clause)


def resolute(first, second):
    additions = set()
    for lit1 in first:
        for lit2 in second:
            if lit1 == negation(lit2):
                temp = set()
                for val in first:
                    if val != lit1:
                        temp.add(val)
                for val in second:
                    if val != lit2:
                        temp.add(val)
                if not tautology(temp):
                    newClause = frozenset(temp)
                    additions.add(newClause)
    return additions


def resolutionPrint(conclusion, statement, ending):
    if ending is not None:
        tree = set()
        q = deque()
        q.append(ending)
        while q:
            top = q.popleft()
            if top in tree:
                continue
            tree.add(top)
            if top.parents is not None:
                q.append(top.parents[0])
                q.append(top.parents[1])
        for node in sorted(tree, key=lambda obj: obj.id):
            print(node)
        print("===============")
    print("[CONCLUSION]:", end=" ")
    for i in range(len(statement) - 1):
        print(statement[i], end=" v ")
    print(statement[len(statement) - 1], end=" ")
    if conclusion:
        print("is true")
    else:
        print("is unknown")
    print("===============")


def runResolution(startClauses, newClauses, statement):
    previous = set()
    for clause in startClauses:
        previous.add(clause.literals)
        print(clause)
    for clause in newClauses:
        previous.add(clause.literals)
        print(clause)
    print("===============")
    q = deque(newClauses)
    clauses = newClauses.union(startClauses)
    while q:
        # SoS strategy
        current = q.popleft()
        iteration = set()
        for clause in clauses:
            if clause == current:
                continue
            additions = resolute(current.literals, clause.literals)
            for addition in additions:
                if not addition:
                    nil = Clause(frozenset(["NIl"]), (current, clause))
                    return resolutionPrint(True, statement, nil)
                if any(addition.issuperset(clause) for clause in previous):
                    continue
                currentClause = Clause(addition, (current, clause))
                iteration.add(currentClause)

        for clause in iteration:
            q.append(clause)
            clauses.add(clause)
            previous.add(clause.literals)

    return resolutionPrint(False, statement, None)


def letHimCook(userInput, startClauses):
    with open(userInput, "r") as f:
        for line in f:
            line = line.strip().lower()
            query = line[len(line) - 1]
            line = line[:-2].split(" v ")
            if query == "?":
                newClauses = set()
                for literal in line:
                    currentClause = Clause(frozenset([negation(literal)]))
                    newClauses.add(currentClause)
                runResolution(startClauses, newClauses, line)
            elif query == "+":
                if not tautology(frozenset(line)):
                    currentClause = Clause(frozenset(line))
                    startClauses.add(currentClause)
            else:
                dummy = Clause(frozenset(line))
                startClauses.discard(dummy)

    return 0


def main(args):
    if args[0] == "resolution":
        startClauses = set()
        lastLine = None
        with open(args[1], "r") as f:
            for line in f:
                if line[0] == "#":
                    continue
                if lastLine is not None:
                    if not tautology(frozenset(lastLine)):
                        currentClause = Clause(frozenset(lastLine))
                        startClauses.add(currentClause)

                line = line.strip().lower()
                lastLine = line.split(" v ")

        newClauses = set()
        for literal in lastLine:
            currentClause = Clause(frozenset([negation(literal)]))
            newClauses.add(currentClause)
        runResolution(startClauses, newClauses, lastLine)

    else:
        startClauses = set()
        with open(args[1], "r") as f:
            for line in f:
                if line[0] == "#":
                    continue
                line = line.strip().lower()
                line = line.split(" v ")
                if not tautology(frozenset(line)):
                    currentClause = Clause(frozenset(line))
                    startClauses.add(currentClause)
        letHimCook(args[2], startClauses)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
