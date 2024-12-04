from typing import List

def is_sub_location(loc1: List[int], loc2: List[int]) -> bool:
    """Check if loc1 is a sub-location of loc2."""
    return loc1[0] >= loc2[0] and loc1[1] <= loc2[1]


def get_loc_of_subtext(text: str, sutext: str) -> List[int] | None:
    """Get the location of a subtext in a text."""
    pos = text.find(sutext)
    if pos == -1:
        return None
    return pos, pos + len(sutext)
