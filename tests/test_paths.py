"""Tests for paths: location defaults, the override order (environment, paths.local.json, default), the host-table
fallback, and named-session lookup. Everything runs against temporary folders; no real data is needed."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import paths  # noqa: E402


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A fake checkout <tmp>/proj/repo with no overrides in force and an unknown host name."""
    repo = tmp_path / 'proj' / 'repo'
    repo.mkdir(parents=True)
    monkeypatch.setattr(paths, 'REPO_ROOT', str(repo))
    monkeypatch.setattr(paths, 'LOCAL_CONFIG', str(repo / 'paths.local.json'))
    monkeypatch.setattr(paths.socket, 'gethostname', lambda: 'some-unknown-machine')
    for var in paths.ENV.values():
        monkeypatch.delenv(var, raising=False)
    return repo


def test_project_root_is_the_folder_around_the_repo_only_when_it_holds_a_data_link(layout):
    assert paths.project() == str(layout)                       # standalone clone: everything stays inside the repo
    assert paths.output('a.png') == str(layout / 'output' / 'a.png')
    assert paths.tmp() == str(layout / 'tmp')
    (layout.parent / 'suite2p_results').mkdir()
    assert paths.project() == str(layout.parent)                # a project folder around the repo is picked up
    assert paths.suite2p('Cadbury') == str(layout.parent / 'suite2p_results' / 'Cadbury')
    assert paths.tmp('x', 'y.csv') == str(layout.parent / 'tmp' / 'x' / 'y.csv')


def test_environment_beats_local_config_beats_default(layout, monkeypatch, tmp_path):
    (layout.parent / 'stimuli').mkdir()
    assert paths.locate('output_dir')[0] == str(layout.parent / 'output')
    with open(paths.LOCAL_CONFIG, 'w') as f:
        json.dump({'output_dir': 'figures', 'suite2p_root': str(tmp_path / 'data')}, f)
    assert paths.output() == str(layout.parent / 'figures')     # relative override: relative to the project folder
    assert paths.suite2p() == str(tmp_path / 'data')
    assert paths.locate('suite2p_root')[1] == 'paths.local.json'
    monkeypatch.setenv('MARMANALYSIS_OUTPUT_DIR', str(tmp_path / 'elsewhere'))
    assert paths.output('f.png') == str(tmp_path / 'elsewhere' / 'f.png')
    assert 'MARMANALYSIS_OUTPUT_DIR' in paths.locate('output_dir')[1]
    monkeypatch.setenv('MARMANALYSIS_PROJECT_ROOT', str(tmp_path / 'other_project'))
    assert paths.project() == str(tmp_path / 'other_project')
    assert paths.tmp() == str(tmp_path / 'other_project' / 'tmp')
    assert paths.stimuli() == str(tmp_path / 'other_project' / 'stimuli')   # the default follows the project root


def test_project_root_override_relative_to_the_repo(layout, monkeypatch):
    monkeypatch.setenv('MARMANALYSIS_PROJECT_ROOT', os.path.join('..', '..'))
    assert paths.project() == str(layout.parent.parent)


def test_bad_local_config_is_an_error_not_ignored(layout):
    with open(paths.LOCAL_CONFIG, 'w') as f:
        f.write('{not json')
    with pytest.raises(ValueError, match='not valid JSON'):
        paths.output()
    with open(paths.LOCAL_CONFIG, 'w') as f:
        json.dump({'suite2p': '/data'}, f)                      # a misspelled key must not pass silently
    os.utime(paths.LOCAL_CONFIG, (1, 1))                        # force a re-read whatever the clock resolution
    with pytest.raises(ValueError, match='unknown keys'):
        paths.suite2p()
    with pytest.raises(KeyError):
        paths.locate('nonsense')


def test_data_roots_fall_back_to_the_host_table(layout, monkeypatch):
    assert paths.host_paths('Galactica.local') == paths.HOSTS['galactica']
    assert paths.host_paths('nowhere') == (None, None)
    path, source = paths.locate('suite2p_root')                 # unknown host, no link: the in-project name, flagged
    assert path == str(layout / 'suite2p_results') and 'NOT CONFIGURED' in source
    monkeypatch.setattr(paths.socket, 'gethostname', lambda: 'OBSIDIAN')
    assert paths.suite2p() == paths.HOSTS['obsidian'][0] and paths.stimuli() == paths.HOSTS['obsidian'][1]
    (layout / 'suite2p_results').mkdir()                        # a folder in the project wins over the host table
    assert paths.suite2p() == str(layout / 'suite2p_results')
    assert paths.stimuli() == paths.HOSTS['obsidian'][1]


def test_named_sessions(layout, monkeypatch):
    data = layout.parent / 'suite2p_results'
    monkeypatch.setattr(paths, 'SESSIONS', {'AnimalA20200101d101010tUTC': ('AnimalA', '20200101d', '101010tUTC_exact'),
                                            'AnimalB20200202d': ('AnimalB', '20200202d', '*Images*'),
                                            'AnimalB20200303d111111tUTC': ('AnimalB', '20200303d', '111111tUTC_exact')})
    data.mkdir()
    # an exact folder name is answered without looking at the disk, so module-level constants never fail at import
    exact = str(data / 'AnimalA' / '20200101d' / '101010tUTC_exact')
    assert paths.session('AnimalA20200101d101010tUTC') == exact
    # a session is named by its slug or a UNIQUE prefix of it; an animal with two sessions does not name either
    assert paths.session('AnimalA20200101d') == exact and paths.session('AnimalA') == exact
    assert paths.session_name('AnimalB202003') == 'AnimalB20200303d111111tUTC'
    assert paths.session_name('AnimalB20200202d') == 'AnimalB20200202d'   # an exact slug wins over prefix matching
    for ambiguous_or_unknown in ('AnimalB', 'Animal', '', 'pd'):
        with pytest.raises(KeyError):
            paths.session_name(ambiguous_or_unknown)
    with pytest.raises(FileNotFoundError):
        paths.session('AnimalB20200202d')
    with pytest.raises(KeyError):
        paths.session('nobody')
    day = data / 'AnimalB' / '20200202d'
    (day / '090000tUTC_stimImagesX' / 'suite2p_a').mkdir(parents=True)
    (day / '080000tUTC_stimImagesY').mkdir()                    # no suite2p output: not a usable session
    (day / '100000tUTC_stimImagesZ' / 'suite2p_b').mkdir(parents=True)
    (day / '110000tUTC_stimDots' / 'suite2p_c').mkdir(parents=True)
    assert paths.find_sessions('AnimalB', '20200202d', '*Images*') == [str(day / '090000tUTC_stimImagesX'),
                                                                        str(day / '100000tUTC_stimImagesZ')]
    assert len(paths.find_sessions('AnimalB', '20200202d')) == 3
    assert paths.session('AnimalB20200202d') == str(day / '090000tUTC_stimImagesX')


def test_real_registry_is_consistent():
    assert not hasattr(paths, 'SESSION_ALIASES')                # sessions have no nicknames, only slugs
    for slug, (animal, date, name) in paths.SESSIONS.items():
        assert slug.startswith(animal + date) and paths.session_name(slug) == slug
    assert set(paths.IN_PROJECT) | {'project_root'} == set(paths.ENV)
    for key, path, source, exists in paths.describe():
        assert os.path.isabs(path) and source


# Stand-alone utility scripts that still carry their own host blocks. (The analysis_for_*.py monoliths live in
# old_analysis_scripts/, which is skipped as a whole.)
LEGACY = {'torch_test.py', 'scrape_stickpng.py', 'process_stickpng_assets.py', 'generate_stimulus_space.py'}
SKIP_DIRS = {'tests', 'tdann', 'external', 'old_analysis_scripts', '__pycache__', 'output', 'tmp'}


def test_no_other_module_spells_out_a_location():
    """paths.py is the only module that may name a data or output folder, or any absolute path. A string such as
    'output', 'tmp/x.csv', 'suite2p_results/...' or '/Users/...' anywhere else ties the code to one working directory
    or one machine again. Docstrings are exempt (usage examples take paths as arguments)."""
    import ast
    import re
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    relative = re.compile(r'^(suite2p_results|stimuli|output|tmp)(/.*)?$')
    absolute = re.compile(r'(^|[\s\'"=(])(/Users/|/home/|/Volumes/|/marmostor/|[A-Za-z]:\\\\?[A-Za-z])')
    offenders = []
    for folder, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
        for name in sorted(files):
            if not name.endswith('.py') or name in LEGACY or (folder == repo and name == 'paths.py'):
                continue
            path = os.path.join(folder, name)
            tree = ast.parse(open(path).read(), path)
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                        docstrings.add(id(first.value))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    if relative.match(node.value) or absolute.search(node.value):
                        offenders.append('%s:%d: %r' % (os.path.relpath(path, repo), node.lineno, node.value[:70]))
    assert not offenders, 'locations spelled out outside paths.py:\n  ' + '\n  '.join(offenders)
