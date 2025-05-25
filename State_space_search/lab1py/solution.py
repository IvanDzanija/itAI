import argparse
import heapq
from collections import deque

# Parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alg", choices=["bfs", "ucs", "astar"], help="algorithm")
parser.add_argument("--ss", required=True, help="Graph desc")
parser.add_argument("--h", help="Heuristic desc")
parser.add_argument(
    "--check-optimistic",
    action="store_true",
)
parser.add_argument(
    "--check-consistent",
    action="store_true",
)
args = parser.parse_args()

# Parse the graph description
start = None
end = set()
graph = dict()
with open(args.ss) as f:
    index = 0
    for line in f:
        if line[0] == "#":
            continue
        index += 1
        if index == 1:
            start = line.strip()
        elif index == 2:
            for state in line.split():
                end.add(state)
        else:
            currentSplit = line.split(":")
            currentState = currentSplit[0].strip()
            graph[currentState] = set()
            for neighbor in currentSplit[1].split():
                neighbor, weight = neighbor.split(",")
                graph[currentState].add((neighbor, float(weight)))

# Parse the heuristic description
heuristic = {}
if args.h:
    with open(args.h) as f:
        for line in f:
            currentSplit = line.split(":")
            heuristic[currentSplit[0].strip()] = float(currentSplit[1].strip())


def printSucc(node, pathing, cost):
    print("[FOUND_SOLUTION]: yes")
    print("[STATES_VISITED]:", len(pathing))
    path = deque()
    while node != "nullNode":
        path.appendleft(node)
        node = pathing[node]
    print("[PATH_LENGTH]:", len(path))
    print("[TOTAL_COST]:", cost)
    print("[PATH]:", end=" ")
    while path:
        node = path.popleft()
        if path:
            print(node, end=" => ")
        else:
            print(node)
    return


def printFail():
    print("[FOUND_SOLUTION]: no")
    return


def bfs():
    q = deque()
    previous = {}
    q.append((start, "nullNode", 0))
    while q:
        node, prev, cost = q.popleft()
        if node in previous.keys():  # NEED THIS CHECK
            continue
        previous[node] = prev
        if node in end:
            return printSucc(node, previous, cost)
        for neighbor, weight in sorted(graph[node]):
            if neighbor not in previous.keys():
                q.append((neighbor, node, cost + weight))
    return printFail()


# Dijkstra's algorithm with cuttoff
def ucs():
    pq = []
    heapq.heappush(pq, (0, start, "nullNode"))
    previous = {}
    while pq:
        cost, node, prev = heapq.heappop(pq)
        if node in previous.keys():
            continue
        previous[node] = prev
        if node in end:
            return printSucc(node, previous, cost)
        for neighbor, weight in graph[node]:
            if neighbor not in previous.keys():
                heapq.heappush(pq, (cost + weight, neighbor, node))
    return printFail()


# Dijkstra's algorithm with cuttoff and heuristic
def astar():
    pq = []
    heapq.heappush(pq, (0 + heuristic[start], start, "nullNode"))
    previous = {}
    while pq:
        cost, node, prev = heapq.heappop(pq)
        cost -= heuristic[node]
        if node in previous.keys():
            continue
        previous[node] = prev
        if node in end:
            return printSucc(node, previous, cost)
        for neighbor, weight in graph[node]:
            if neighbor not in previous.keys():
                heapq.heappush(
                    pq, (cost + weight + heuristic[neighbor], neighbor, node)
                )
    return printFail()


# Heuristic h has to not overestimate the true cost at all states to be optimistic
# Formal: (h(s) <= h*(s) for all s in States) => Heuristic is optimistic
# h* is the true cost to reach the goal state from s
# h is the heuristic cost to reach the goal state from s
def printOptimistic(checked):
    print("# HEURISTIC-OPTIMISTIC", args.h)
    optimistic = True
    for node in sorted(graph.keys()):
        print("[CONDITION]", end=": ")
        if node in checked.keys() and checked[node] >= heuristic[node]:
            print("[OK] h(" + node + ") <= h*:", heuristic[node], "<=", checked[node])
        else:
            print("[ERR] h(" + node + ") <= h*:", heuristic[node], "<=", checked[node])
            optimistic = False
    print(
        "[CONCLUSION]:",
        ("Heuristic is optimistic." if optimistic else "Heuristic is not optimistic."),
    )
    return


# def checkOptimistic():
#     # Optimization idea: reverse the graph and run Dijkstra's algorithm from the end states
#     # Problem with this approach: only if there is a small number of end states this is efficient
#     # With one end states the complexity O((E+V) * logV)
#     # One counter example: we set every state as an end state and then our algorithm is O(V * (E + V) * logV)
#     checked = {key: float("inf") for key in graph.keys()}
#     reverseGraph = {}
#     for node in graph.keys():
#         for neighbor, weight in graph[node]:
#             if neighbor not in reverseGraph.keys():
#                 reverseGraph[neighbor] = set()
#             reverseGraph[neighbor].add((node, weight))
#         if node not in reverseGraph.keys():
#             reverseGraph[node] = set()

#     for node in end:
#         checked[node] = float(0.0)
#         pq = []
#         heapq.heappush(pq, (0, node))
#         seen = set()
#         while pq:
#             cost, node = heapq.heappop(pq)
#             if node in seen:
#                 continue
#             seen.add(node)
#             checked[node] = min(checked[node], cost)
#             for neighbor, weight in reverseGraph[node]:
#                 heapq.heappush(pq, (cost + weight, neighbor))
#     printOptimistic(checked)
#     return


def checkOptimisticFast():
    # Multisource Dijkstra's algorithm on reversed graph
    # Now the complexity is  O((E+V) * logV)
    reverseGraph = {}
    for node in graph.keys():
        for neighbor, weight in graph[node]:
            if neighbor not in reverseGraph.keys():
                reverseGraph[neighbor] = set()
            reverseGraph[neighbor].add((node, weight))
        if node not in reverseGraph.keys():
            reverseGraph[node] = set()

    pq = []
    checked = {}
    for node in end:
        heapq.heappush(pq, (float(0), node))
    while pq:
        cost, node = heapq.heappop(pq)
        if node in checked.keys():
            continue
        checked[node] = cost
        for neighbor, weight in reverseGraph[node]:
            heapq.heappush(pq, (cost + weight, neighbor))
    printOptimistic(checked)

    return


# Heuristic h has to be monotonic to be consistent
# Formal: (h(s) <= h(s') + c for all s -> s' in Transitions) => Heuristic is consistent
# c is the cost to move from s to s'
# h(s) is the heuristic cost to reach the goal state from s
# h(s') is the heuristic cost to reach the goal state from s'
def checkConsistent():
    print("# HEURISTIC-CONSISTENT", args.h)
    consistent = True
    for node in graph.keys():
        for neighbor, weight in graph[node]:
            print("[CONDITION]", end=": ")
            if heuristic[node] <= heuristic[neighbor] + weight:
                print(
                    "[OK] h(" + node + ") <= h(" + neighbor + ") + c:",
                    heuristic[node],
                    "<=",
                    heuristic[neighbor],
                    "+",
                    weight,
                )
            else:
                print(
                    "[ERR] h(" + node + ") <= h(" + neighbor + ") + c:",
                    heuristic[node],
                    "<=",
                    heuristic[neighbor],
                    "+",
                    weight,
                )
                consistent = False

    print(
        "[CONCLUSION]:",
        ("Heuristic is consistent." if consistent else "Heuristic is not consistent."),
    )
    return


if args.alg == "bfs":
    bfs()
elif args.alg == "ucs":
    ucs()
elif args.alg == "astar":
    astar()
if args.check_optimistic:
    # checkOptimistic()
    checkOptimisticFast()
if args.check_consistent:
    checkConsistent()
