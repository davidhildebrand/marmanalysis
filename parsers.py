#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
from warnings import warn


def parse_log_eyecal(eyecal_log, data=None):
    """
    Parse the log file output of Santiago Otero Coronel's eye-tracking calibration.
    """

    lines = eyecal_log.splitlines()
    out = {'zero': None,
           'crse': {'data': {}},
           'circ': {'data': {}},
           'grdf': {'data': {}},
           'grdt': {'data': {}}}

    pattern_ai_idx = r'.*AI_data\.shape\s*=\s*(\(([0-9]+),\s*([0-9]+)\)|see\s*next\s*entry).*'
    pattern_pos = r'.*\.pos\s*=\s*\[([\-0-9\.\s]+)\].*'
    pattern_cvals = r'.*calib_values_candidate\s*=\s*\[([\-0-9\.\s]+)\].*'

    lmode = ''
    for idx_line, line in enumerate(lines):
        # First pass through eye-calibration log file
        if 'EXP \toculomatic zeroing start' in line:
            # '234.8201 \tEXP \tcoarse eye-tracking calibration start, AI_data.shape = (134078, 6)'
            lmode = 'zero'
            # '234.7010 \tEXP \toculomatic zeroing end, AI_data.shape = (133928, 6)'
            if 'zeroing end' in line:
                lmode = ''
        elif 'EXP \tcoarse eye-tracking calibration start' in line:
            # '234.8201 \tEXP \tcoarse eye-tracking calibration start, AI_data.shape = (134078, 6)'
            lmode = 'crse'
            # '316.3800 \tEXP \tcoarse eye-tracking calibration end, coarse_stims_pos = [[ 5 -5]'
            if 'calibration end' in line:
                lmode = ''
        elif 'EXP \tcircular trajectory calibration start' in line:
            # '316.3821 \tEXP \tcircular trajectory calibration start, AI_data.shape = (180857, 6)'
            lmode = 'circ'
            # '388.5193 \tEXP \tcircular trajectory calibration end, AI_data.shape = (222356, 6)'
            if 'calibration end' in line:
                lmode = ''
        elif 'EXP \tgrid faces calibration start' in line:
            # '388.5270 \tEXP \tgrid faces calibration start, AI_data.shape = (222368, 6)'
            lmode = 'grdf'
            # '579.5484 \tEXP \tgrid faces calibration end, AI_data.shape = (332223, 6)'
            if 'calibration end' in line:
                lmode = ''
        elif 'EXP \tgrid target eye-tracking calibration start' in line:
            # '579.5631 \tEXP \tgrid target eye-tracking calibration start, AI_data.shape = (332239, 6)'
            lmode = 'grdt'
            # '697.7133 \tEXP \tgrid target eye-tracking calibration end, AI_data.shape = (399857, 6)'
            if 'calibration end' in line:
                lmode = ''
        if lmode == '':
            continue

        if re.match(pattern_ai_idx, line) is not None:
            g_ai_idx = re.match(pattern_ai_idx, line).groups()
            if g_ai_idx[0].replace(' ', '') != 'seenextentry':
                tmp_ai_idx = int(g_ai_idx[1])
            else:
                # Use 'sep' character as a placeholder for subsequent AI range value.
                tmp_ai_idx = chr(31)
        else:
            tmp_ai_idx = None
        if re.match(pattern_pos, line) is not None:
            g_pos = re.match(pattern_pos, line).groups()
            tmp_pos = np.fromstring(g_pos[0].strip(), dtype=float, sep=' ')
        else:
            tmp_pos = None
        if re.match(pattern_cvals, line) is not None:
            tmp_cvals = np.fromstring(re.match(pattern_cvals, line).groups()[0].strip(), dtype=float, sep=' ')
        else:
            tmp_cvals = None

        match lmode:
            case 'zero':
                pattern_zero = r'.*oculomatic\s*zeroing,?\s*(presenting|hiding)\s*face,?.*'
                # '6.3156 \tEXP \toculomatic zeroing, presenting face, AI_data.shape = (2529, 6)'
                # '234.7007 \tEXP \toculomatic zeroing, hiding face, AI_data.shape = (133928, 6)'
                if re.match(pattern_zero, line) is not None:
                    g = re.match(pattern_zero, line).groups()
                    if g[0] == 'presenting':
                        out['zero'] = {'AIrng': [None, None]}
                        out['zero']['AIrng'][0] = tmp_ai_idx
                    elif g[0] == 'hiding':
                        out['zero']['AIrng'][1] = tmp_ai_idx

            case 'crse':
                pattern_coarse = r'.*coarse\s*(eye-tracking\s*calibration|trial)\s*(start|end|\d+)?,?\s*' + \
                                 r'(showing\s*face|hiding\s*face|candidate)?,?.*'
                # '234.8201 \tEXP \tcoarse eye-tracking calibration start, AI_data.shape = (134078, 6)'
                # '234.8261 \tEXP \tcoarse trial 0, showing face,face.pos = [ 5. -5.], AI_data.shape = (134078, 6)'
                # '243.3274 \tEXP \tcoarse trial 0, hiding face, AI_data.shape = (138942, 6)'
                # '243.8488 \tEXP \tcoarse trial 0, showing face,face.pos = [ 5. -5.], AI_data.shape = (139240, 6)',
                # '247.9988 \tEXP \tcoarse trial 0, hiding face, coarse_oculomatic_calib_values_candidate = [ 0.825335  -2.4251535] for face.pos = [ 5. -5.], AI_data.shape = (141609, 6)'
                # '304.0263 \tEXP \tcoarse trial 20, showing face,face.pos = [-5.  5.], AI_data.shape = (173761, 6)'
                # '305.1594 \tEXP \tcoarse trial 20, hiding face, coarse_oculomatic_calib_values_candidate = ' + \
                #   '[-3.3869004   0.05144549] for face.pos = [-5.  5.], AI_data.shape = (174405, 6)'
                # '316.3800 \tEXP \tcoarse eye-tracking calibration end, coarse_stims_pos = [[ 5 -5]'
                if re.match(pattern_coarse, line) is not None:
                    g = re.match(pattern_coarse, line).groups()

                    if g[0].replace(' ', '') == 'eye-trackingcalibration':
                        if g[1] == 'start':
                            if 'n_trials' not in out['crse']:
                                out['crse']['n_trials'] = -1
                                out['crse']['AIrng'] = [None, None]
                            out['crse']['AIrng'][0] = tmp_ai_idx
                        elif g[1] == 'end':
                            if tmp_ai_idx is None:
                                tmp_ai_idx = max(max([out['crse']['data'][i]['AIrng']
                                                      for i, d in enumerate(out['crse']['data'])]))
                            out['crse']['AIrng'][1] = tmp_ai_idx
                    elif g[0] == 'trial':
                        trl = int(g[1])
                        if out['crse']['n_trials'] - 1 < trl:
                            out['crse']['n_trials'] = trl + 1
                            out['crse']['data'][trl] = {'AIrng': [None, None], 'pos': None, 'cvals': None}
                        if g[2].replace(' ', '') == 'showingface':
                            out['crse']['data'][trl]['AIrng'][0] = tmp_ai_idx
                            out['crse']['data'][trl]['pos'] = tmp_pos
                        elif g[2].replace(' ', '') == 'hidingface':
                            out['crse']['data'][trl]['AIrng'][1] = tmp_ai_idx
                            out['crse']['data'][trl]['cvals'] = tmp_cvals
                        elif g[2] == 'candidate':
                            continue

            case 'circ':
                pattern_circ = r'.*circular\s*trajectory\s*(calibration|trial)\s*(start|end|\d+),?\s*' + \
                               r'(start|turn|end)?,?\s*(faceID\s*=\s*)?(\d+)?\s*(start|end)?,?.*'
                # '316.3927 \tEXP \tcircular trajectory trial 0 start, faceID = 9, AI_data.shape = (180857, 6)'
                # '325.4014 \tEXP \tcircular trajectory trial 0, turn 3 start, AI_data.shape = (186039, 6)'
                # '328.4045 \tEXP \tcircular trajectory trial 0, turn 3 end, AI_data.shape = (187767, 6)'
                # '328.4045 \tEXP \tcircular trajectory trial 0 end, AI_data.shape = (187767, 6)'
                if re.match(pattern_circ, line) is not None:
                    g = re.match(pattern_circ, line).groups()
                    if g[0] == 'calibration':
                        if g[1] == 'start':
                            if 'n_trials' not in out['circ']:
                                out['circ']['n_trials'] = -1
                                out['circ']['AIrng'] = [None, None]
                            out['circ']['AIrng'][0] = tmp_ai_idx
                        elif g[1] == 'end':
                            out['circ']['AIrng'][1] = tmp_ai_idx
                    elif g[0] == 'trial':
                        trl = int(g[1])
                        if out['circ']['n_trials'] - 1 < trl:
                            out['circ']['n_trials'] = trl + 1
                            out['circ']['data'][trl] = {'n_turns': -1, 'AIrng': [None, None], 'stim': None}
                        if g[2] == 'start':
                            out['circ']['data'][trl]['AIrng'][0] = tmp_ai_idx
                            out['circ']['data'][trl]['stim'] = g[3].replace(' ', '').replace('=', '').strip() + g[4]
                        elif g[2] == 'turn':
                            trn = int(g[4])
                            if out['circ']['data'][trl]['n_turns'] - 1 < trn:
                                if g[5] == 'start':
                                    out['circ']['data'][trl][trn] = {'AIrng': [None, None]}
                                    out['circ']['data'][trl][trn]['AIrng'][0] = tmp_ai_idx
                                elif g[5] == 'end':
                                    out['circ']['data'][trl][trn]['AIrng'][1] = tmp_ai_idx
                                    out['circ']['data'][trl]['n_turns'] = trn + 1
                        elif g[2] == 'end':
                            out['circ']['data'][trl]['AIrng'][1] = tmp_ai_idx

            case 'grdf':
                pattern_gridf = r'.*grid\s*faces?\s*(calibration|trial)\s*(start|end|\d+),?\s*(face|ISI)?\s*' + \
                                r'(start|end)?,?.*'
                # '388.5401 \tEXP \tgrid face trial 0, ISI start, AI_data.shape = (222368, 6)',
                # '389.0546 \tEXP \tgrid face trial 0, ISI end, AI_data.shape = see next entry',
                # '389.0546 \tEXP \tgrid face trial 0, face start, face.pos = [ 0. -5.], AI_data.shape = (222662, 6)'
                # '392.0783 \tEXP \tgrid face trial 0, face end, AI_data.shape = see next entry'
                # '392.0783 \tEXP \tgrid face trial 1, ISI start, AI_data.shape = (224406, 6)'
                if re.match(pattern_gridf, line) is not None:
                    g = re.match(pattern_gridf, line).groups()

                    # Replace 'sep' character placeholder with AI range value.
                    ks = ['isi', 'face']
                    for k in ks:
                        rngs = [out['grdf']['data'][i][k]['AIrng'] if out['grdf']['data'][i][k] else [None, None]
                                for i, d in enumerate(out['grdf']['data'])]
                        if np.any(np.array(rngs) == chr(31)):
                            septs = np.where(np.array(rngs) == chr(31))[0]
                            for sti in septs:
                                sepi = np.argwhere(np.array(out['grdf']['data'][sti][k]['AIrng']) == chr(31))[0][0]
                                out['grdf']['data'][sti][k]['AIrng'][sepi] = tmp_ai_idx

                    if g[0] == 'calibration':
                        if g[1] == 'start':
                            if 'n_trials' not in out['grdf']:
                                out['grdf']['n_trials'] = -1
                                out['grdf']['AIrng'] = [None, None]
                            out['grdf']['AIrng'][0] = tmp_ai_idx
                        elif g[1] == 'end':
                            out['grdf']['AIrng'][1] = tmp_ai_idx
                    elif g[0] == 'trial':
                        trl = int(g[1])
                        if out['grdf']['n_trials'] - 1 < trl:
                            out['grdf']['n_trials'] = trl + 1
                            out['grdf']['data'][trl] = {'isi': {'AIrng': [None, None]},
                                                        'face': {'AIrng': [None, None], 'pos': None}}
                        if g[2] == 'face':
                            if g[3] == 'start':
                                out['grdf']['data'][trl]['face']['AIrng'][0] = tmp_ai_idx
                                out['grdf']['data'][trl]['face']['pos'] = tmp_pos
                            elif g[3] == 'end':
                                out['grdf']['data'][trl]['face']['AIrng'][1] = tmp_ai_idx
                        elif g[2] == 'ISI':
                            if g[3] == 'start':
                                out['grdf']['data'][trl]['isi']['AIrng'][0] = tmp_ai_idx
                            elif g[3] == 'end':
                                out['grdf']['data'][trl]['isi']['AIrng'][1] = tmp_ai_idx

            case 'grdt':
                pattern_gridt = r'.*grid\s*target\s*(eye-tracking\s*calibration|trial)\s*(start|end|\d+),?\s*' + \
                                r'(ISI|central\s*target|grid\s*target|face\s*reward)?\s*(start|fixation|end)?,?\s*' + \
                                r'(start|interrupted|completed|fixation\s*success|fixation\s*fail)?,?.*'
                # '579.5631 \tEXP \tgrid target eye-tracking calibration start, AI_data.shape = (332239, 6)'
                # '579.5829 \tEXP \tgrid target trial 0, ISI start, AI_data.shape = (332239, 6)',
                # '580.5769 \tEXP \tgrid target trial 0, ISI end, AI_data.shape = see next entry',
                # '580.5769 \tEXP \tgrid target trial 0, central target start, AI_data.shape = (332820, 6)',
                # '580.9177 \tEXP \tgrid target trial 0, central target fixation start, AI_data.shape = (333014, 6)',
                # '580.9247 \tEXP \tgrid target trial 0, central target fixation interrupted, AI_data.shape = (333018, 6)',
                # '583.6077 \tEXP \tgrid target trial 1, central target start, AI_data.shape = (334556, 6)',
                # '585.4010 \tEXP \tgrid target trial 1, central target end, fixation success, AI_data.shape = see next entry',
                # '585.4149 \tEXP \tgrid target trial 1, grid target start, grid_target.pos = [5. 0.], AI_data.shape = (335589, 6)',
                # '585.4177 \tEXP \tgrid target trial 1, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (335596, 6)',
                # '585.5225 \tEXP \tgrid target trial 1, grid target fixation completed, grid_target.pos = [5. 0.], AI_data.shape = (335652, 6)',
                # '585.5332 \tEXP \tgrid target trial 1, grid target end, fixation success, AI_data.shape = see next entry',
                # '585.5332 \tEXP \tgrid target trial 1, face reward start, face.pos = [5. 0.], AI_data.shape = (335657, 6)',
                # '586.0616 \tEXP \tgrid target trial 1, face reward end, AI_data.shape = see next entry',
                if re.match(pattern_gridt, line) is not None:
                    g = re.match(pattern_gridt, line).groups()

                    # Replace 'sep' character placeholder with AI range value.
                    ks = ['isi', 'ctr', 'targ', 'rwrd']
                    for k in ks:
                        rngs = [out['grdt']['data'][i][k]['AIrng'] if out['grdt']['data'][i][k] else [None, None]
                                for i, d in enumerate(out['grdt']['data'])]
                        if np.any(np.array(rngs) == chr(31)):
                            septs = np.where(np.array(rngs) == chr(31))[0]
                            for sti in septs:
                                sepi = np.argwhere(np.array(out['grdt']['data'][sti][k]['AIrng']) == chr(31))[0][0]
                                out['grdt']['data'][sti][k]['AIrng'][sepi] = tmp_ai_idx

                    if g[0].replace(' ', '') == 'eye-trackingcalibration':
                        if g[1] == 'start':
                            if 'n_trials' not in out['grdt']:
                                out['grdt']['n_trials'] = -1
                                out['grdt']['AIrng'] = [None, None]
                            out['grdt']['AIrng'][0] = tmp_ai_idx
                        elif g[1] == 'end':
                            out['grdt']['AIrng'][1] = tmp_ai_idx
                    elif g[0] == 'trial':
                        trl = int(g[1])
                        if out['grdt']['n_trials'] - 1 < trl:
                            out['grdt']['n_trials'] = trl + 1
                            out['grdt']['data'][trl] = {'isi': {'AIrng': [None, None]},
                                                        'ctr': None,
                                                        'targ': None,
                                                        'rwrd': None}
                        if g[2].replace(' ', '') == 'centraltarget':
                            if g[3] == 'start':
                                out['grdt']['data'][trl]['ctr'] = {'AIrng': [None, None], 'success': None}
                            elif g[3] == 'fixation':
                                if g[4] == 'start':
                                    out['grdt']['data'][trl]['ctr']['AIrng'][0] = tmp_ai_idx
                                elif g[4] == 'interrupted':
                                    out['grdt']['data'][trl]['ctr']['AIrng'][1] = tmp_ai_idx
                                elif g[4] == 'completed':
                                    out['grdt']['data'][trl]['ctr']['AIrng'][1] = tmp_ai_idx
                            elif g[3] == 'end':
                                if g[4].replace(' ', '') == 'fixationsuccess':
                                    out['grdt']['data'][trl]['ctr']['success'] = True
                                elif g[4].replace(' ', '') == 'fixationfail':
                                    out['grdt']['data'][trl]['ctr']['success'] = False
                        elif g[2].replace(' ', '') == 'gridtarget':
                            if g[3] == 'start':
                                out['grdt']['data'][trl]['targ'] = {'AIrng': [None, None], 'pos': None,
                                                                    'success': None}
                                out['grdt']['data'][trl]['targ']['pos'] = tmp_pos
                            elif g[3] == 'fixation':
                                if g[4] == 'start':
                                    out['grdt']['data'][trl]['targ']['AIrng'][0] = tmp_ai_idx
                                elif g[4] == 'interrupted':
                                    out['grdt']['data'][trl]['targ']['AIrng'][1] = tmp_ai_idx
                                elif g[4] == 'completed':
                                    out['grdt']['data'][trl]['targ']['AIrng'][1] = tmp_ai_idx
                            elif g[3] == 'end':
                                if g[4].replace(' ', '') == 'fixationsuccess':
                                    out['grdt']['data'][trl]['targ']['success'] = True
                                elif g[4].replace(' ', '') == 'fixationfail':
                                    out['grdt']['data'][trl]['targ']['success'] = False
                        # '588.9046 \tEXP \tgrid target trial 2, face reward start, face.pos = [0. 0.], AI_data.shape = (337584, 6)',
                        # '589.4260 \tEXP \tgrid target trial 2, face reward end, AI_data.shape = see next entry',
                        elif g[2].replace(' ', '') == 'facereward':
                            if g[3] == 'start':
                                out['grdt']['data'][trl]['rwrd'] = {'AIrng': [None, None], 'pos': None}
                                out['grdt']['data'][trl]['rwrd']['AIrng'][0] = tmp_ai_idx
                                out['grdt']['data'][trl]['rwrd']['pos'] = tmp_pos
                            elif g[3] == 'end':
                                out['grdt']['data'][trl]['rwrd']['AIrng'][1] = tmp_ai_idx
                        elif g[2] == 'ISI':
                            if g[3] == 'start':
                                out['grdt']['data'][trl]['isi']['AIrng'][0] = tmp_ai_idx
                            elif g[3] == 'end':
                                out['grdt']['data'][trl]['isi']['AIrng'][1] = tmp_ai_idx
            case _:
                continue
        # del g_ai_idx, g_pos, g_cvals, tmp_ai_idx, tmp_pos, tmp_cvals

    # Add eye-tracking data if available
    if data is not None:
        for k in list(out.keys()):
            match k:
                case 'zero':
                    ai_rng = out[k]['AIrng']
                    if None not in ai_rng:
                        out[k]['AIdata'] = data[ai_rng[0]:ai_rng[1], :2]
                    else:
                        out[k]['AIdata'] = None
                case 'crse':
                    for trl in range(out[k]['n_trials']):
                        ai_rng = out[k]['data'][trl]['AIrng']
                        if None not in ai_rng:
                            out[k]['data'][trl]['AIdata'] = data[ai_rng[0]:ai_rng[1], :2]
                        else:
                            out[k]['data'][trl]['AIdata'] = None
                case 'circ':
                    for trl in range(out[k]['n_trials']):
                        ai_rng = out[k]['data'][trl]['AIrng']
                        if None not in ai_rng:
                            out[k]['data'][trl]['AIdata'] = data[ai_rng[0]:ai_rng[1], :2]
                        else:
                            out[k]['data'][trl]['AIdata'] = None
                        for trn in range(out[k]['data'][trl]['n_turns']):
                            ai_rng = out[k]['data'][trl][trn]['AIrng']
                            if None not in ai_rng:
                                out[k]['data'][trl][trn]['AIdata'] = data[ai_rng[0]:ai_rng[1], :2]
                            else:
                                out[k]['data'][trl][trn]['AIdata'] = None
                case 'grdf':
                    for trl in range(out[k]['n_trials']):
                        for typ in ['isi', 'face']:
                            ai_rng = out[k]['data'][trl][typ]['AIrng']
                            if None not in ai_rng:
                                out[k]['data'][trl][typ]['AIdata'] = data[ai_rng[0]:ai_rng[1], :2]
                            else:
                                out[k]['data'][trl][typ]['AIdata'] = None
                case 'grdt':
                    for trl in range(out[k]['n_trials']):
                        for typ in ['isi', 'ctr', 'targ', 'rwrd']:
                            if out[k]['data'][trl][typ] is not None:
                                ai_rng = out[k]['data'][trl][typ]['AIrng']
                                if None not in ai_rng:
                                    out[k]['data'][trl][typ]['AIdata'] = data[ai_rng[0]:ai_rng[1], :2]
                                else:
                                    out[k]['data'][trl][typ]['AIdata'] = None
    return out


def create_stimulus_record(trials=1) -> pd.DataFrame:
    log = pd.DataFrame({'trial': range(trials),
                        'dur_isi_pre': None,
                        'dur_stim': None,
                        'dur_isi_post': None,

                        't_isi_i': None,
                        't_isi_f': None,
                        'acqfr_isi_i': None,
                        'acqfr_isi_f': None,
                        'dispfr_isi_i': None,
                        'dispfr_isi_f': None,
                        'ai_isi_i': None,
                        'ai_isi_f': None,

                        't_fix_i': None,
                        't_fix_f': None,
                        'acqfr_fix_i': None,
                        'acqfr_fix_f': None,
                        'dispfr_fix_i': None,
                        'dispfr_fix_f': None,
                        'ai_fix_i': None,
                        'ai_fix_f': None,

                        't_stim_i': None,
                        't_stim_f': None,
                        'acqfr_stim_i': None,
                        'acqfr_stim_f': None,
                        'dispfr_stim_i': None,
                        'dispfr_stim_f': None,
                        'ai_stim_i': None,
                        'ai_stim_f': None,

                        'cond': None,
                        'stim_mode': None,
                        'stim_class': None,
                        'stim_subclass': None,

                        'image': None,
                        'image_path': None,

                        'units': None,
                        'pos': None,
                        'size': None,
                        'ori': None,
                        'color': None,
                        'colorSpace': None,
                        'contrast': None,
                        'opacity': None,
                        'texRes': None,

                        'grating_tex': None,
                        'grating_contrast': None,
                        'grating_dir': None,
                        'grating_ori': None,
                        'grating_sf': None,
                        'grating_tf': None,

                        'dots_translation_dir': None,
                        'dots_opticflow_dir': None,
                        'dots_rotation_dir': None,

                        'flash_type': None,

                        'video': None,
                        'video_path': None,
                        'video_fps': None,

                        'freq': None,
                        'lev': None,
                        'ampmod_freq': None,
                        'voc_path': None
                        })
    log.set_index(['trial'])

    return log


def parse_log_stim_image_orig(session_log):
    """
    Parse the session log file output of the original stimulus_image.py script.
    """

    lines = session_log.splitlines()

    # 37.1533         EXP     trial 0/240, stim start, image, cond=7, name=image7:b16.png,
    # path=/FreiwaldSync/MarmoScope/Stimulus/Images/Song_etal_Wang_2020_NatCommun/480288_equalized_RGBA_FOBonly/b16.png,
    # units=deg, pos=[0. 0.], size=[12.   7.2], ori=0.0, color=[1. 1. 1.], colorSpace=rgb, contrast=1.0,
    # opacity=1.0, texRes=512, acqfr=23, AI_data.shape=(1336, 5)
    trialdata = {}
    ims = {}
    impaths = {}
    lf_categories = {}
    tmp_image = None
    tmp_imagepath = None
    tmp_units = None
    tmp_pos = None
    tmp_size = None
    tmp_ori = None
    tmp_category = None
    tmp_catid = None
    tmp_cond = None
    tmp_acqfr = None
    tmp_stimtimestr = ''
    stimtime_mode = False
    tmp_isitimestr = ''
    isitime_mode = False
    for line in lines:
        if 'EXP \tstim_times:' in line:
            stimtime_mode = True
        if 'EXP \tinterstim_times:' in line:
            isitime_mode = True
        if stimtime_mode:
            tmp_stimtimestr = tmp_stimtimestr + line
            if ']' in line:
                stimtime_mode = False
        if isitime_mode:
            tmp_isitimestr = tmp_isitimestr + line
            if ']' in line:
                isitime_mode = False

        if 'stim start' not in line or 'image' not in line:
            continue

        col = line.split('trial')
        if not col:
            continue
        subcol = [sc.strip() for sc in col[1].split(',')]
        tmp_trial = int(subcol[0].split('/')[0].strip())
        if 'cond' in subcol[3]:
            tmp_cond = int(subcol[3].split('=')[1].strip())
        else:
            print('could not get cond from log entry')
        if 'image' in subcol[4]:
            tmp_image = subcol[4].split(':')[1].strip()
            if tmp_image not in ims:
                ims[tmp_image] = tmp_cond
            tmp_category = tmp_image[0]
            if tmp_category not in lf_categories:
                lf_categories[tmp_category] = len(lf_categories)
            tmp_catid = lf_categories[tmp_category]
        else:
            print('could not get image name from log entry')
        if 'path' in subcol[5]:
            tmp_imagepath = subcol[5].split('=')[1].strip()
            if tmp_imagepath not in impaths:
                impaths[tmp_imagepath] = tmp_cond
        else:
            print('could not get image name from log entry')
        if 'units' in subcol[6]:
            tmp_units = subcol[6].split('=')[1].strip()
        else:
            print('could not get units from log entry')
        if 'pos' in subcol[7]:
            tmp_pos = np.fromstring(subcol[7].split('=')[1].strip('[]'), sep=' ')
        else:
            print('could not get pos from log entry')
        if 'size' in subcol[8]:
            tmp_size = np.fromstring(subcol[8].split('=')[1].strip('[]'), sep=' ')
        else:
            print('could not get size from log entry')
        if 'ori' in subcol[9]:
            tmp_ori = float(subcol[9].split('=')[1].strip())
        else:
            print('could not get ori from log entry')
        if 'acqfr' in subcol[15]:
            tmp_acqfr = int(subcol[15].split('=')[1].strip())
        else:
            print('could not get acqfr from log entry')
        trialdata[tmp_trial] = {'cond': tmp_cond,
                                'image': tmp_image,
                                'imagepath': tmp_imagepath,
                                'units': tmp_units,
                                'pos': tmp_pos,
                                'size': tmp_size,
                                'ori': tmp_ori,
                                'category': tmp_category,
                                'catid': tmp_catid,
                                'acqfr': tmp_acqfr}
    return trialdata


def parse_log_stim_image(session_log) -> pd.DataFrame:
    """
    Parse the session log file output of the original stimulus_image.py script into newer DataFrame format.
    """

    lines = session_log.splitlines()

    mode = None
    found_stimfunc = False
    tmp_stimtimestr = ''
    tmp_isitimestr = ''
    for line in lines:
        if 'ImageStim(' in line:
            found_stimfunc = True
        if 'EXP \tstim_times:' in line:
            mode = 'stimtime'
        if 'EXP \tinterstim_times:' in line:
            mode = 'isitime'
        if mode == 'stimtime':
            tmp_stimtimestr = tmp_stimtimestr + line
            if ']' in line:
                mode = None
        if mode == 'isitime':
            tmp_isitimestr = tmp_isitimestr + line
            if ']' in line:
                mode = None

    if tmp_stimtimestr != '':
        s_si = tmp_stimtimestr.find('[')
        s_ei = tmp_stimtimestr.find(']')
        times_stim = np.fromstring(tmp_stimtimestr[s_si + 1:s_ei].strip(' []'), sep=' ')
        dur_stim = np.round(np.mean(times_stim), 2)
    else:
        warn('Could not automatically detect stimulus duration, assuming 1.0 sec.')
        times_stim = []
        dur_stim = 1.0

    if tmp_isitimestr != '':
        s_si = tmp_isitimestr.find('[')
        s_ei = tmp_isitimestr.find(']')
        times_isi = np.fromstring(tmp_isitimestr[s_si + 1:s_ei].strip(' []'), sep=' ')
        dur_isi = np.round(np.min(times_isi), 2)
    else:
        warn('Could not automatically detect interstimulus duration, assuming 1.0 sec.')
        times_isi = []
        dur_isi = 1.0

    if len(times_stim) == len(times_isi):
        n_trials = len(times_stim)
    else:
        n_trials = None
        warn('Number of stimulus and interstimulus times do not match.  Unknown number of trials.')

    log = create_stimulus_record(trials=n_trials)

    for line in lines:
        pattern_isi = r'^\s*([0-9\.]+)\s*EXP\s*trial\s*([0-9]+)\/?([0-9]+)?,?\s*ISI\s*(start|end),?\s*' + \
                      r'acqfr=([0-9]+),?\s*AI_data\.shape=\(([0-9]+),\s*([0-9]+)\)'
        if re.match(pattern_isi, line) is not None:
            g = re.match(pattern_isi, line).groups()
            t = float(g[0])
            trial = int(g[1])
            if g[2] is not None and (n_trials - 1) != int(g[2]):
                warn('Calculated number of trials ({}) does not match number '.format(n_trials) +
                     'referenced in trial {} ({}): {}'.format(trial, g[2], line))
            acqfr = int(g[4])
            ai_shape = (int(g[5]), int(g[6]))

            match g[3]:
                case 'start':
                    log.at[trial, 'dur_isi_pre'] = times_isi[trial]
                    if trial > 0:
                        log.at[trial - 1, 'dur_isi_post'] = times_isi[trial]
                    log.at[trial, 't_isi_i'] = t
                    log.at[trial, 'acqfr_isi_i'] = acqfr
                    # log.at[trial, 'dispfr_isi_i'] = np.nan
                    log.at[trial, 'ai_isi_i'] = ai_shape[0]
                case 'end':
                    log.at[trial, 't_isi_f'] = t
                    log.at[trial, 'acqfr_isi_f'] = acqfr
                    # log.at[trial, 'dispfr_isi_f'] = np.nan
                    log.at[trial, 'ai_isi_f'] = ai_shape[0]
                case _:
                    warn('Unknown ISI event in log file.')

        pattern_fix = r'^\s*([0-9\.]+)\s*EXP\s*trial\s*([0-9]+)\/?([0-9]+)?,\s*fixation\s*(start|end),\s*' + \
                      r'acqfr=([0-9]+),?\s*AI_data\.shape=\(([0-9]+),\s*([0-9]+)\)'
        if re.match(pattern_fix, line) is not None:
            g = re.match(pattern_fix, line).groups()
            t = float(g[0])
            trial = int(g[1])
            if g[2] is not None and (n_trials - 1) != int(g[2]):
                warn('Calculated number of trials ({}) does not match number'.format(n_trials) +
                     'referenced in trial {} ({}): {}'.format(trial, g[2], line))
            acqfr = int(g[4])
            ai_shape = (int(g[5]), int(g[6]))

            match g[3]:
                case 'start':
                    log.at[trial, 't_fix_i'] = t
                    log.at[trial, 'acqfr_fix_i'] = acqfr
                    # log.at[trial, 'dispfr_fix_i'] = np.nan
                    log.at[trial, 'ai_fix_i'] = ai_shape[0]
                case 'end':
                    log.at[trial, 't_fix_f'] = t
                    log.at[trial, 'acqfr_fix_f'] = acqfr
                    # log.at[trial, 'dispfr_fix_f'] = np.nan
                    log.at[trial, 'ai_fix_f'] = ai_shape[0]
                case _:
                    warn('Unknown ISI event in log file.')

        pattern_stim = r'^\s*([0-9\.]+)\s*EXP\s*trial\s*([0-9]+)\/?([0-9]+)?,?\s*stim\s*(start|end),?\s*' + \
                       r'((image),?\s*cond=([0-9]+),?\s*name=[a-zA-Z0-9_]+:([a-zA-Z0-9_]+\.png),?\s*path=([^\s]+),?\s*' + \
                       r'units=([a-zA-Z]+),?\s*pos=\[([\-0-9\.\s]+)\],?\s*size=\[([\-0-9\.\s]+)\],?\s*' + \
                       r'ori=([0-9\.]+),?\s*color=\[([\-0-9\.\s]+)\],?\s*colorSpace=([a-zA-Z]+),?\s*' + \
                       r'contrast=([0-9\.]+),?\s*opacity=([0-9\.]+),?\s*texRes=([0-9]+),?)?\s*' + \
                       r'acqfr=([0-9]+),?\s*AI_data\.shape=\(([0-9]+),\s*([0-9]+)\)'
        if re.match(pattern_stim, line) is not None:
            g = re.match(pattern_stim, line).groups()
            t = float(g[0])
            trial = int(g[1])
            if g[2] is not None and (n_trials - 1) != int(g[2]):
                warn('Calculated number of trials ({}) does not match number'.format(n_trials) +
                     'referenced in trial {} ({}): {}'.format(trial, g[2], line))
            acqfr = int(g[18])
            ai_shape = (int(g[19]), int(g[20]))

            match g[3]:
                case 'start':
                    log.at[trial, 'trial'] = trial
                    log.at[trial, 'dur_stim'] = times_stim[trial]
                    log.at[trial, 't_stim_i'] = t
                    log.at[trial, 'acqfr_stim_i'] = acqfr
                    # log.at[trial, 'dispfr_stim_i'] = np.nan
                    log.at[trial, 'ai_stim_i'] = ai_shape[0]

                    log.at[trial, 'cond'] = int(g[6])
                    log.at[trial, 'stim_mode'] = 'visual'
                    log.at[trial, 'stim_class'] = g[5]
                    log.at[trial, 'stim_subclass'] = None
                    log.at[trial, 'image'] = g[7]
                    log.at[trial, 'image_path'] = g[8]
                    log.at[trial, 'units'] = g[9]
                    log.at[trial, 'pos'] = np.fromstring(g[10], sep=' ')
                    log.at[trial, 'size'] = np.fromstring(g[11], sep=' ')
                    log.at[trial, 'ori'] = float(g[12])
                    log.at[trial, 'color'] = np.fromstring(g[13], sep=' ')
                    log.at[trial, 'colorSpace'] = g[14]
                    log.at[trial, 'contrast'] = float(g[15])
                    log.at[trial, 'opacity'] = float(g[16])
                    log.at[trial, 'texRes'] = int(g[17])
                case 'end':
                    log.at[trial, 't_stim_f'] = t
                    log.at[trial, 'acqfr_stim_f'] = acqfr
                    # log.at[trial, 'dispfr_stim_f'] = np.nan
                    log.at[trial, 'ai_stim_f'] = ai_shape[0]
                case _:
                    warn('Unknown stim event in log file.')

        pattern_conc = r'^\s*([0-9\.]+)\s*EXP\s*conclusion,?\s*(start|end),?\s*' + \
                       r'acqfr=([0-9]+),?\s*AI_data\.shape=\(([0-9]+),\s*([0-9]+)\)'
        if re.match(pattern_conc, line) is not None:
            if 'trial' not in locals() or not found_stimfunc:
                raise Exception('Incorrect log parser chosen. This parser is for image sessions. '
                                'Conclusion reached without finding a trial or no ImageStim found.')
            g = re.match(pattern_conc, line).groups()
            t = float(g[0])
            acqfr = int(g[2])
            ai_shape = (int(g[3]), int(g[4]))

            match g[1]:
                case 'start':
                    time_conc_i = t
                case 'end':
                    if 'time_conc_i' in locals():
                        log.at[trial, 'dur_isi_post'] = t - time_conc_i
                    else:
                        log.at[trial, 'dur_isi_post'] = np.nan
                case _:
                    warn('Unknown conclusion event in log file.')

    return log


def parse_log_stim_dots_orig(session_log):
    """
    Parse the session log file output of the original stimulus_dots.py script.
    """

    lines = session_log.splitlines()

    # 41.9371         EXP     trial 0, stim start, grating, full field, drifting, cond=5, ori=225.0, tex=sin,
    # size=[75.67137421 75.67137421], sf=[1.2 0. ], tf=4, mask=None, contrast=1.0, acqfr=222

    trialdata = {}

    tmp_stimtimestr = ''
    stimtime_mode = False
    tmp_isitimestr = ''
    isitime_mode = False
    for line in lines:
        if 'EXP \tstim_times:' in line:
            stimtime_mode = True
        if 'EXP \tinterstim_times:' in line:
            isitime_mode = True
        if stimtime_mode:
            tmp_stimtimestr = tmp_stimtimestr + line
            if ']' in line:
                stimtime_mode = False
        if isitime_mode:
            tmp_isitimestr = tmp_isitimestr + line
            if ']' in line:
                isitime_mode = False

        if 'stim start' not in line:
            continue

        col = line.split('trial')
        if not col:
            continue
        subcol = [sc.strip() for sc in col[1].split(',')]
        tmp_trial = int(subcol[0].strip())
        tmp_cond = int(subcol[5].split('=')[1].strip())

        tmp_f = float(subcol[13].split('=')[1].strip())
        tmp_acqfr = int(subcol[20].split('=')[1].strip())
        trialdata[tmp_trial] = {'cond': tmp_cond,
                                'f': tmp_f,
                                'acqfr': tmp_acqfr}
    return trialdata
