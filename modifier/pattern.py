import sys
import re
import time
import logging
from .modify import similarfinder, make_pattern, make_head_pattern
from tqdm import tqdm
from miner.reward import _get_wildcard_names, WrittenVariableCollector
from modifier.modify import SimilarFinder, _MultiChangeComputer
from rope.base import codeanalyze, ast
from rope.refactor import restructure, patchedast
from rope.refactor.similarfinder import CodeTemplate

def _to_rope_vars(vars):
    new_vars = []
    for var in vars:
        if re.match(r'^SUGARWILDCARD_\d+$', var):
            new_vars.append('${' + var + '}')
        else:
            continue
    return new_vars

rec_wildcard = r'^SUGARWILDCARD_\d+$'

def _make_goal(code, pattern_type, pattern_id):
    tree = ast.parse(code if pattern_type != 'stmt_head' else code + 'pass')
    names = _get_wildcard_names(tree)
    if pattern_type in ['stmt', 'stmt_head']:
        token = f'<{pattern_id}_stmt_token>'
        if len(names) == 0:
            return token
        collector = WrittenVariableCollector()
        collector.visit(tree)
        written_vars = set(collector.written)
        read_vars = set(names) - written_vars
        return ';'.join(sorted(_to_rope_vars(written_vars))) + token + ';'.join(sorted(_to_rope_vars(read_vars)))
        # return token + ';'.join(sorted(_to_rope_vars(names)))
    else:
        token = f'<{pattern_id}_expr_token>'
        if len(names) == 0:
            return token
        else:
            names.sort(key=lambda x: int(x.split('_')[1]))
            return token + ';'.join(sorted(_to_rope_vars(names))) + '<\expr_token>'# I can also try unique names here


class Pattern:
    def __init__(self, code, pattern_type, reward, freq, predefined_id=None):
        self.id = predefined_id if predefined_id is not None else id(code)
        self.code = code
        self.pattern = make_pattern(code) if pattern_type != 'stmt_head' else make_head_pattern(code)
        self.goal = _make_goal(code, pattern_type, self.id)
        self.pattern_type = pattern_type
        self.reward = reward
        self.freq = freq
        self.finder = similarfinder.RawSimilarFinder(self.code) if pattern_type != 'stmt_head' else None
        self.special_tokens = set([f'<{self.id}_stmt_token>']) if 'stmt' in pattern_type else set([f'<{self.id}_expr_token>', '<\expr_token>'])
        self.parent_patterns = set()
        self.children_patterns = set()
    

    def get_match(self, pattern):
        matches = list(self.finder.get_matches(pattern))
        return matches
    
    def to_dict(self):
        return {
            'id': self.id,
            'code': self.code,
            'pattern': self.pattern,
            'goal': self.goal,
            'type': self.pattern_type,
            'reward': self.reward,
            'freq': self.freq,
            'parent_patterns': list(self.parent_patterns),
            'children_patterns': list(self.children_patterns),
        }

    @classmethod
    def from_dict(cls, dict):
        instance = cls(dict['code'], dict['type'], dict['reward'], dict['freq'], predefined_id=dict['id'])
        return instance

    def __repr__(self):
        return f"Pattern(id={self.id}, pattern={self.pattern}, goal={self.goal}, type={self.pattern_type}, reward={self.reward})"

# conflict & subpattern
def is_subpattern(pattern1, pattern2):
    try:
        return len(pattern1.get_match(pattern2.pattern)) > 0
    except AttributeError as e:
        return False
        



def binary_search(intervals, index):
    low, high = 0, index - 1
    while low <= high:
        mid = (low + high) // 2
        if intervals[mid]['end'] <= intervals[index]['start']:
            if mid + 1 < len(intervals) and intervals[mid + 1]['end'] <= intervals[index]['start']:
                low = mid + 1
            else:
                return mid
        else:
            high = mid - 1
    return -1


def apply_patterns(examples, complete_patterns, partial_patterns):

    new_content = []
    old_content = []
    stats = []
    for code in examples['content']:
        old_content.append(code)
        try:
            finder = SimilarFinder(code)
        except Exception as e:
            print(f'Error creating finder: {e}')
            new_content.append(code)
            stats.append(None)
            continue
        raw_matches = []
        
        # Gather all raw matches
        for p in complete_patterns:
            try:    
                match_p = list(finder.get_matches(p.pattern))
            except Exception as e:
                continue
            if len(match_p) > 0:
                raw_matches.extend([{'pattern_id': p.id, 'match': match, 'reward': p.reward, 'start': match.get_region()[0], 'end': match.get_region()[1], 'pattern': p.pattern, 'goal': p.goal} for match in match_p])

        finder = SimilarFinder(code, use_head_matcher=True)
        for p in partial_patterns:
            try:    
                match_p = list(finder.get_matches(p.pattern))
            except Exception as e:
                print(f'Error matching partial pattern: {e}')
                continue
            for match in match_p:
                raw_matches.append({'pattern_id': p.id, 'match': match, 'reward': p.reward, 'start': match.get_region()[0], 'end': match.get_region()[1], 'pattern': p.pattern, 'goal': p.goal})

        # Sort intervals by their end time
        raw_matches.sort(key=lambda x: x['end'])

        # Apply dynamic programming to find the maximum reward and track the matches
        n = len(raw_matches)
        if n == 0:
            new_content.append(code)
            stats.append(None)
            continue
        
        dp = [0] * n
        selected_matches = [[] for _ in range(n)]
        
        dp[0] = raw_matches[0]['reward']
        selected_matches[0] = [raw_matches[0]]
        
        for i in range(1, n):
            # Include current interval
            incl_reward = raw_matches[i]['reward']
            incl_matches = [raw_matches[i]]
            l = binary_search(raw_matches, i)
            if l != -1:
                incl_reward += dp[l]
                incl_matches += selected_matches[l]
            
            # Store the maximum of including or excluding the current interval
            if incl_reward > dp[i - 1]:
                dp[i] = incl_reward
                selected_matches[i] = incl_matches
            else:
                dp[i] = dp[i - 1]
                selected_matches[i] = selected_matches[i - 1]
        
        matches = selected_matches[-1]
        mid_time = time.time()
        stat = ','.join([str(m['pattern_id']) for m in matches])
        templates = [CodeTemplate(m['goal']) for m in matches]
        rope_matches = [m['match'] for m in matches]
        # fix the bugs in _ChangeComputer
        lines = codeanalyze.SourceLinesAdapter(code)
        try:
            computer = _MultiChangeComputer(code, lines, rope_matches, templates)
            result = computer.get_changed()
            code = result if result is not None else code
        except Exception as e:
            new_content.append(code)
            stats.append(None)
            continue
        end_time = time.time()
        logging.debug(f"Applied pattern {len(matches)} matches in {end_time - mid_time:.6f} seconds")  # Log pattern application
        new_content.append(code)
        stats.append(stat)
    return {'content': new_content, 'stats': stats, 'old_content': old_content}


