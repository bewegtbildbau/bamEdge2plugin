"""
BamEdge2 Nuke plugin — menu.py
Adds BamEdge2 to the Nodes toolbar.
"""

import nuke

toolbar = nuke.toolbar("Nodes")
m = toolbar.addMenu("BamEdge2")
m.addCommand(
    "BamEdge2",
    'nuke.createNode("BamEdge2")',
)
