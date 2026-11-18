#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where things live: the one module that knows filesystem locations.

No other module should contain an absolute path, a host-name table, or a path that only works from one working
directory. Ask this module instead: moving the project, or cloning the repo onto another machine, then needs no code
change, at most one setting.

Locations (each function joins optional further parts onto its location):

    project(*parts)   the working folder around the repo: data links, outputs, scratch, third-party folders.
                      Default: the folder that contains this repo if it holds a data link (suite2p_results/ or
                      stimuli/), else the repo folder itself.
    suite2p(*parts)   <animal>/<date>/<session>/ folders with suite2p output (``base_path`` in the legacy scripts).
                      Default: <project>/suite2p_results if present, else this machine's entry in HOSTS.
    stimuli(*parts)   the stimulus sets, i.e. the local mirror of MarmoScope/Stimulus/Sets (``stim_path``).
                      Default: <project>/stimuli if present, else this machine's entry in HOSTS.
    output(*parts)    figures and tables that are kept.                 Default: <project>/output
    tmp(*parts)       scratch and intermediate files.                   Default: <project>/tmp

    session(key)      folder of a named session: a slug from SESSIONS or a unique prefix of one, e.g.
                      'Cadbury20221016d152643tUTC' or 'Cadbury20221016d'.
    find_sessions(animal, date, pattern)   session folders that match and hold suite2p output.

Overriding a location (first hit wins):
    1. an environment variable: MARMANALYSIS_PROJECT_ROOT, MARMANALYSIS_SUITE2P_ROOT, MARMANALYSIS_STIMULI_ROOT,
       MARMANALYSIS_OUTPUT_DIR, MARMANALYSIS_TMP_DIR
    2. ``paths.local.json`` next to this file (not tracked by git), for example
           {"suite2p_root": "/Volumes/data/suite2p_results", "stimuli_root": "/Volumes/data/Stimulus/Sets"}
       with the keys project_root, suite2p_root, stimuli_root, output_dir, tmp_dir
    3. the default above
A relative override is taken relative to the project folder (for project_root itself: relative to the repo folder).

Paths come back absolute and NOT symlink-resolved, so the in-project data links stay visible in messages and records.
No folder has to exist for this module to import or answer: a missing location shows up where it is used.

Run:  python marmanalysis/paths.py      prints every location, where its value came from, and whether it exists
"""
import glob
import json
import os
import socket

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG = os.path.join(REPO_ROOT, 'paths.local.json')
# Location key -> the environment variable that overrides it. The keys are also the paths.local.json keys.
ENV = {'project_root': 'MARMANALYSIS_PROJECT_ROOT', 'suite2p_root': 'MARMANALYSIS_SUITE2P_ROOT',
       'stimuli_root': 'MARMANALYSIS_STIMULI_ROOT', 'output_dir': 'MARMANALYSIS_OUTPUT_DIR',
       'tmp_dir': 'MARMANALYSIS_TMP_DIR'}
# Name of each location inside the project folder.
IN_PROJECT = {'suite2p_root': 'suite2p_results', 'stimuli_root': 'stimuli', 'output_dir': 'output', 'tmp_dir': 'tmp'}
DATA_KEYS = ('suite2p_root', 'stimuli_root')
# Known machines, keyed by a substring of the lower-cased host name: (suite2p_root, stimuli_root). The single copy of
# the host block that the analysis_for_*.py scripts carry. Used only when the project folder has no data link.
HOSTS = {
    'galactica': (r'/Users/davidh/Data/Freiwald/suite2p_results', r'/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets'),
    'obsidian': (r'F:\Data', r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets'),
    'dobbin': (r'D:\Data', r'C:\Users\DavidH\Sync\Freiwald\MarmoScope\Stimulus\Sets'),
}

# Named sessions: slug (sessionio.session_slug, AnimalDateTime) -> (animal, date, session folder name or glob pattern).
# The one place a session's location is spelled out. Per-analysis settings stay with the analysis that uses them.
# A session is ALWAYS named by its slug. There are deliberately no nicknames: an area name such as "PD" or an
# animal name fits several sessions, so it cannot name one of them (DH 2026-11-17).
SESSIONS = {
    'Cadbury20221016d152643tUTC': ('Cadbury', '20221016d',                          # area PD, 730 um field
                                   '152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_'
                                   'stimImagesSongFOBonly'),
    'Curly20231103d162355tUTC': ('Curly', '20231103d', '162355tUTC*ImagesFOBmin*'),   # FOBmin, 3 mm field
    'Curly20240130d172813tUTC': ('Curly', '20240130d', '172813tUTC*'),                # PD/MT border, 3 mm field
}

_config_cache = {}


def host_paths(hostname=None):
    """(suite2p_root, stimuli_root) of a known machine from HOSTS, else (None, None)."""
    h = (hostname or socket.gethostname()).lower()
    for name, roots in HOSTS.items():
        if name in h:
            return roots
    return (None, None)


def local_config():
    """Per-machine overrides from paths.local.json, {} when there is no such file. A file that is not a JSON object
    with known keys raises ValueError: a silently ignored override would send data lookups or output elsewhere."""
    try:
        stamp = (LOCAL_CONFIG, os.path.getmtime(LOCAL_CONFIG))
    except OSError:
        return {}
    if _config_cache.get('stamp') != stamp:
        with open(LOCAL_CONFIG) as f:
            try:
                cfg = json.load(f)
            except ValueError as e:
                raise ValueError('%s is not valid JSON: %s' % (LOCAL_CONFIG, e))
        if not isinstance(cfg, dict):
            raise ValueError('%s must hold a JSON object' % LOCAL_CONFIG)
        unknown = sorted(set(cfg) - set(ENV))
        if unknown:
            raise ValueError('%s has unknown keys %s (known: %s)' % (LOCAL_CONFIG, unknown, ', '.join(ENV)))
        _config_cache.update(stamp=stamp, cfg=cfg)
    return _config_cache['cfg']


def _absolute(path, base):
    path = os.path.expanduser(os.path.expandvars(str(path)))
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(base, path))


def locate(key):
    """(path, where the value came from) for a location key: project_root, suite2p_root, stimuli_root, output_dir or
    tmp_dir."""
    if key not in ENV:
        raise KeyError('unknown location %r (known: %s)' % (key, ', '.join(ENV)))
    value, source = os.environ.get(ENV[key]), 'environment variable %s' % ENV[key]
    if not value:
        value, source = local_config().get(key), os.path.basename(LOCAL_CONFIG)
    if key == 'project_root':
        if value:
            return _absolute(value, REPO_ROOT), source
        around = os.path.dirname(REPO_ROOT)
        if any(os.path.exists(os.path.join(around, IN_PROJECT[k])) for k in DATA_KEYS):
            return around, 'default: the folder around the repo (it holds a data link)'
        return REPO_ROOT, 'default: the repo folder (the folder around it holds no data link)'
    root = locate('project_root')[0]
    if value:
        return _absolute(value, root), source
    inside = os.path.join(root, IN_PROJECT[key])
    if key in DATA_KEYS and not os.path.exists(inside):
        host = host_paths()[DATA_KEYS.index(key)]
        if host:
            return host, 'host table entry for this machine (no %s in the project folder)' % IN_PROJECT[key]
        return inside, 'default, NOT CONFIGURED: set %s or %s in %s' % (ENV[key], key, os.path.basename(LOCAL_CONFIG))
    return inside, 'default: inside the project folder'


def project(*parts):
    """The project folder (the working folder around the repo), joined with ``parts``."""
    return os.path.join(locate('project_root')[0], *parts)


def suite2p(*parts):
    """The suite2p results root (<animal>/<date>/<session>/...), joined with ``parts``."""
    return os.path.join(locate('suite2p_root')[0], *parts)


def stimuli(*parts):
    """The stimulus sets root (local mirror of MarmoScope/Stimulus/Sets), joined with ``parts``."""
    return os.path.join(locate('stimuli_root')[0], *parts)


def output(*parts):
    """The folder for figures and tables that are kept, joined with ``parts``. Not created here."""
    return os.path.join(locate('output_dir')[0], *parts)


def tmp(*parts):
    """The scratch folder, joined with ``parts``. Not created here."""
    return os.path.join(locate('tmp_dir')[0], *parts)


def find_sessions(animal, date, pattern='*'):
    """Session folders <suite2p root>/<animal>/<date>/<pattern> that hold at least one suite2p* folder, sorted."""
    hits = sorted(glob.glob(suite2p(animal, date, pattern)))
    return [h for h in hits if os.path.isdir(h) and glob.glob(os.path.join(h, 'suite2p*'))]


def session_name(key):
    """The registered slug that ``key`` names: the slug itself, or a unique prefix of one such as
    'Cadbury20221016d'. KeyError when no session, or more than one, matches."""
    if key in SESSIONS:
        return key
    hits = [s for s in SESSIONS if s.startswith(key)] if key else []
    if len(hits) != 1:
        raise KeyError('%r names %s (registered: %s)'
                       % (key, 'more than one session: ' + ', '.join(hits) if hits else 'no registered session',
                          ', '.join(SESSIONS)))
    return hits[0]


def session(key):
    """Folder of a named session: a slug from SESSIONS or a unique prefix of one (see session_name). A session
    registered by its exact folder name is answered without touching the disk (safe at import); one registered by a
    pattern resolves to its first match that holds suite2p output, and raises FileNotFoundError when there is none."""
    slug = session_name(key)
    animal, date, name = SESSIONS[slug]
    if not glob.has_magic(name):
        return suite2p(animal, date, name)
    hits = find_sessions(animal, date, name)
    if not hits:
        raise FileNotFoundError('no session folder with suite2p output matches %s' % suite2p(animal, date, name))
    return hits[0]


def describe():
    """One row per location: (key, path, where the value came from, exists)."""
    rows = []
    for key in ENV:
        path, source = locate(key)
        rows.append((key, path, source, os.path.exists(path)))
    return rows


def main():
    print('repo          %s' % REPO_ROOT)
    for key, path, source, exists in describe():
        print('%-13s %s\n%14s%s | %s' % (key, path, '', 'exists' if exists else 'MISSING', source))
    print('named sessions')
    for slug in SESSIONS:
        try:
            path = session(slug)
            state = 'exists' if os.path.isdir(path) else 'MISSING'
        except FileNotFoundError as e:
            path, state = str(e), 'MISSING'
        print('   %-28s %-8s %s' % (slug, state, path))


if __name__ == '__main__':
    main()
