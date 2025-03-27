from itertools import permutations


def count_inversions(state):
    """Counts the number of inversions in the puzzle state."""
    flat_state = [tile for tile in state if tile != 0]  # Ignore the blank space (0)
    inversions = sum(
        1
        for i in range(len(flat_state))
        for j in range(i + 1, len(flat_state))
        if flat_state[i] > flat_state[j]
    )
    return inversions


def is_solvable(state):
    """A state is solvable if it has an even number of inversions."""
    return count_inversions(state) % 2 == 0


# Define the tiles in 8-puzzle (numbers 1-8 and empty space represented as 0)
tiles = list(range(9))  # [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Generate all possible permutations and filter only solvable ones
solvable_states = [state for state in permutations(tiles) if is_solvable(state)]

# Save to a file
with open("reachable_8_puzzle_states.txt", "w") as f:
    for state in solvable_states:
        st = list()
        for x, i in enumerate(state):
            if i == 0:
                st.append("x")
            else:
                st.append(str(i))
            if x == 2 or x == 5:
                st.append("_")

        f.write("".join(st) + " ")
