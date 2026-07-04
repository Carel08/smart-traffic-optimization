"""
Basic smoke tests for the project setup.
"""


def test_import_config():
    from src.config import DEFAULT_NUM_INTERSECTIONS, MAX_NUM_INTERSECTIONS

    assert DEFAULT_NUM_INTERSECTIONS > 0
    assert MAX_NUM_INTERSECTIONS >= DEFAULT_NUM_INTERSECTIONS