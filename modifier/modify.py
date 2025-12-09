import logging
import collections
import warnings
from itertools import chain
import re
from textwrap import dedent
from rope.refactor import restructure, similarfinder, sourceutils, patchedast
from rope.refactor.similarfinder import _RopeVariable, _ASTMatcher, Match
from rope.base import codeanalyze, ast

reg = re.compile(r"SUGARWILDCARD_(\d+)")

def make_pattern(code):
    return reg.sub(lambda match: "${" + match.group(0) + "}", code)

def make_head_pattern(code):
    # find the highest wildcard in the code
    wildcard_list = reg.findall(code)
    wildcard_ids = [int(w) for w in wildcard_list]
    max_id = max(wildcard_ids) if wildcard_ids else 0
    code = code + f'\n    SUGARWILDCARD_{max_id + 1}'
    return reg.sub(lambda match: "${" + match.group(0) + "}", code)


class ASTDict(dict):
    # Override the `in` operator to use ast_equal for comparison
    def __contains__(self, key):
        for existing_key in self.keys():
            if ast_equal(existing_key, key):
                return True
        return False
    
    # Override the item access `self.matched_asts[node]` to use ast_equal for key lookup
    def __getitem__(self, key):
        for existing_key in self.keys():
            if ast_equal(existing_key, key):
                return super().__getitem__(existing_key)  # Use the found key to get the value
        raise KeyError(f"Key {key} not found in ASTDict")

# Function for AST node comparison
def ast_equal(node_a, node_b):
    return ast.dump(node_a, annotate_fields=False, include_attributes=False) == ast.dump(node_b, annotate_fields=False, include_attributes=False)
   

class _MultiChangeComputer:
    def __init__(self, code, lines, matches, goals):
        self.source = code
        self.goals = goals
        self.matches = matches
        self.lines = lines
        self.matched_asts = ASTDict()
        self._nearest_roots = {}


    def get_changed(self):
        collector = codeanalyze.ChangeCollector(self.source)
        i = 0
        for match, goal in zip(self.matches, self.goals):
            # if hasattr(match, 'is_head'):
            start, end = match.get_region()
            replacement = self._get_matched_text(match, goal)
            collector.add_change(start, end, replacement)
            i += 1
        return collector.get_changed()

    def _is_expression(self, match):
        return isinstance(
            match, similarfinder.ExpressionMatch
        )

    def _get_matched_text(self, match, goal):
        mapping = {}
        for name in goal.get_names():
            node = match.get_ast(name)
         
            force = self._is_expression(match) and match.ast == node
            if hasattr(node, 'region'):
                mapping[name] = self._get_node_text(node, goal, force)
            elif isinstance(node, ast.AST):
                mapping[name] = ast.unparse(node)
            else:
                mapping[name] = node
        unindented = goal.substitute(mapping)
        return self._auto_indent(match.get_region()[0], unindented)

    def _get_node_text(self, node, goal, force=False):
        start, end = patchedast.node_region(node)
        main_text = self.source[start:end]
        collector = codeanalyze.ChangeCollector(main_text)
   
        for node in self._get_nearest_roots(node):
            sub_start, sub_end = patchedast.node_region(node)
            collector.add_change(
                sub_start - start, sub_end - start, self._get_node_text(node, goal)
            )
        result = collector.get_changed()
        if result is None:
            return main_text
        return result

    def _auto_indent(self, offset, text):
        lineno = self.lines.get_line_number(offset)
        indents = sourceutils.get_indents(self.lines, lineno)
        result = []
        for index, line in enumerate(text.splitlines(True)):
            if index != 0 and line.strip():
                result.append(" " * indents)
            result.append(line)
        return "".join(result)

    def _get_nearest_roots(self, node):
        if node not in self._nearest_roots:
            result = []
            for child in ast.iter_child_nodes(node):
                if child in self.matched_asts:
                    result.append(child)
                else:
                    result.extend(self._get_nearest_roots(child))
            self._nearest_roots[node] = result
        return self._nearest_roots[node]


class _MyPatchingASTWalker(patchedast._PatchingASTWalker):
    def _arg(self, node):
        if node.annotation:
            self._handle(node, [node.arg, ":", node.annotation])
        else:
            self._handle(node, [node.arg])
    

class SimilarFinder(similarfinder.RawSimilarFinder):

    def __init__(self, source, node=None, does_match=None, use_head_matcher=False):
        super().__init__(source, node, does_match)
        self.use_head_matcher = use_head_matcher
    
    def _init_using_ast(self, node, source):
        self.source = source
        self._matched_asts = {}
        if not hasattr(node, "region"):
            walker = _MyPatchingASTWalker(source, children=False)
            walker(node)
        self.ast = node
    
    def _get_matched_asts(self, code):
        if code not in self._matched_asts:
            wanted = self._create_pattern(code)
            if self.use_head_matcher:
                matcher = _HeadASTMatcher(self.ast, wanted, self.does_match)
            else:
                matcher = _ASTMatcher(self.ast, wanted, self.does_match)
            matches = matcher.find_matches()
            self._matched_asts[code] = matches
        return self._matched_asts[code]

class _HeadASTMatcher(_ASTMatcher):
    def __init__(self, body, pattern, does_match):
        super().__init__(body, pattern, does_match)
    
    def _match_nodes(self, expected, node, mapping):
        logging.debug(f"mapping: {mapping}")
        logging.debug(f"type(expected): {type(expected)}")
        if isinstance(expected, ast.Name):
            logging.debug(f"expected.id: {expected.id}")
            logging.debug(f"self.ropevar.is_var(expected.id): {self.ropevar.is_var(expected.id)}")
            if self.ropevar.is_var(expected.id):
                return self._match_wildcard(expected, node, mapping)
        if not isinstance(expected, ast.AST):
            return expected == node
        if expected.__class__ != node.__class__:
            return False

        children1 = self._get_children(expected)
        children2 = self._get_children(node)
        logging.debug(f"children1: {children1}")
        logging.debug(f"children2: {children2}")
        if len(children1) != len(children2):
            return False
        for child1, child2 in zip(children1, children2):
            if isinstance(child1, ast.AST):
                if not self._match_nodes(child1, child2, mapping):
                    return False
            elif isinstance(child1, (list, tuple)):
                if not isinstance(child2, (list, tuple)) or len(child1) != len(child2):
                    return False
                for c1, c2 in zip(child1, child2):
                    if not self._match_nodes(c1, c2, mapping):
                        return False
            else:
                if child1 and isinstance(child1, str) and self.ropevar.is_var(child1):
                    mapping[self.ropevar.get_base(child1)] = child2
                elif type(child1) is not type(child2) or child1 != child2:
                    return False
        return True
    
    def _get_children(self, node):
        """Return not `ast.expr_context` children of `node`"""
        rtn = []
        for field, child in ast.iter_fields(node):
            if isinstance(child, ast.expr_context) or field in ['body', 'orelse', 'finalbody']:
                continue

            rtn.append(child)
        return rtn 

    def _check_statements(self, node):
        for field, child in ast.iter_fields(node):
            if isinstance(child, (list, tuple)):
                self.__check_stmt_list(child)

    def __check_stmt_list(self, nodes):
        for index in range(len(nodes)):
            if len(nodes) - index >= len(self.pattern):
                current_stmts = nodes[index : index + len(self.pattern)]
                mapping = {}
                try:
                    if self._match_stmts(current_stmts, mapping):
                        self.matches.append(HeadStatementMatch(current_stmts, mapping))
                except Exception as e:
                    print(f'Error matching statements: {e}')
                    print(f'Current statements: {current_stmts}')
                    print(f'Pattern: {self.pattern}')
                    continue


class HeadStatementMatch(Match):
    def __init__(self, ast_list, mapping):
        super().__init__(mapping)
        self.ast_list = ast_list
        self.is_head = True
        start = ast_list[0].region[0]
        first_stmt = ast_list[0].body[0]
        end = first_stmt.region[0] - first_stmt.col_offset
        self.region = (start, end-1) 


    def get_region(self):
        return self.region[0], self.region[1]
