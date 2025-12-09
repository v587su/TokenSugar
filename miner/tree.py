
import ast


def get_node_depth_and_width(node):
    def _helper(node, depth):
        max_depth = depth
        max_width = 0
        
        for attr in ['body', 'orelse', 'finalbody']:
            if hasattr(node, attr) and getattr(node, attr):
                children = getattr(node, attr)
                max_width = max(max_width, len(children))  # Update max width
                
                # Recurse into children
                for child in children:
                    child_depth, child_width = _helper(child, depth + 1)
                    max_depth = max(max_depth, child_depth)  # Update max depth
                    max_width = max(max_width, child_width)  # Update max width

        return max_depth, max_width
    
    # Start recursion with initial depth 1
    depth, width = _helper(node, 1)
    return depth, width

def _is_large_node(node):
    depth, width = get_node_depth_and_width(node)
    return depth > 2 or width > 1

class Tree:
    """This is a customized ast tree to support mining idioms
    supprted function:
    1. get all adjacent statements in its children
    2. get all expressions in its children
    3. abstract the variables
    Example:
    tree = MinerTree(src)
    print(tree.get_all_adjacent_statements())
    """
    def __init__(self, tree):
        self.tree = ast.parse(tree) if isinstance(tree, str) else tree


    def get_all_statements(self):
        """
        Get all statements in the tree
        """
        stmts = []
        compound_stmts = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.stmt):
                depth, width = get_node_depth_and_width(node)
                if width <= 1 and depth <= 2:
                    stmts.append(node)
                if width > 0:
                    compound_stmts.append(node)
        return stmts, compound_stmts
    
    def get_all_expressions(self):
        """
        Get all expressions in its children
        """
        expressions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.expr):
                expressions.append(node)
        return expressions

    def get_all_adjacent_statements(self, n=2):
        """
        Get all adjacent statements (the adjacent statements belong to the same parent node) in the tree
        """
        adjacent_groups = []

        def find_adjacent_statements(node):
            for attr in ['body', 'orelse', 'finalbody']:
                if not (hasattr(node, attr) and getattr(node, attr)):
                    continue
                children = getattr(node, attr)
                for i in range(len(children) - n + 1):
                    if any([_is_large_node(children[j]) for j in range(i, i + n)]):
                        continue
                    if all(isinstance(children[j], ast.stmt) for j in range(i, i + n)):
                        adjacent_groups.append(children[i:i + n])
                for child in children:
                    find_adjacent_statements(child)

        find_adjacent_statements(self.tree)
        return adjacent_groups
    
   