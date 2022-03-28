# This file is part of modelling_research.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import lsst.daf.butler as dafButler
import numpy as np


def convert_match_tables(
    butler: dafButler.Butler,
    tract: int,
    name_skymap='DC2',
    test: bool = False,
    drop_match_row: bool = True,
):
    dataset_columns = {
        'truth': ('truth_summary', ['id']),
        'object': ('objectTable_tract', ['objectId', 'detect_isPrimary', 'merge_peak_sky']),
    }
    if test:
        dataset_columns['truth'][1].extend(['is_pointsource', 'flux_i'])
        dataset_columns['object'][1].extend(['i_psfFlux'])

    truth, objects = (butler.get(dataset, tract=tract, skymap=name_skymap, parameters={'columns': columns})
                      for (dataset, columns) in dataset_columns.values())

    match_ref, match_target = (
        butler.get(f"match_{cat}_truth_summary_objectTable_tract", tract=tract, skymap=name_skymap)
        for cat in ('ref', 'target')
    )
    data_ref = {
        'index': np.arange(len(match_ref)),
        'match_candidate': match_ref['match_candidate'].values,
        'match_count': match_ref['match_count'].astype('int8').values,
        'match_chisq': match_ref['match_chisq'].values,
        'match_n_chisq_finite': match_ref['match_n_chisq_finite'].astype('int8').values,
        'truth_id': np.array([str(x) for x in truth['id']], dtype='<U16'),
        'match_objectId': match_ref['match_row'].values,
    }
    if not drop_match_row:
        data_ref['match_row'] = match_ref['match_row'].values.copy()

    has_match = match_ref['match_row'].values >= 0
    data_ref['match_objectId'][has_match] = objects.index[match_ref['match_row'][has_match]]
    
    data_target = {
        'index': np.arange(len(match_target)),
        'match_candidate': (objects['detect_isPrimary'] & ~objects['merge_peak_sky']).values,
        'match_truth_id': np.full(len(match_target), '', dtype='<U16'),
        'objectId': objects.index.values,
    }
    if not drop_match_row:
        data_target['match_row'] = match_target['match_row'].values,

    has_match = match_target['match_row'].values >= 0
    data_target['match_truth_id'][has_match] = truth['id'][match_target['match_row'][has_match]]

    if test:
        import matplotlib.pyplot as plt

        matches = data_ref['match_objectId']
        matched = matches >= 0
        stars_matched = truth['is_pointsource'] & matched
        matches = matches[stars_matched]
        mag_true = -2.5*np.log10(truth['flux_i'][stars_matched].values) + 31.4
        mag_meas = -2.5*np.log10(objects['i_psfFlux'].loc[matches].values) + 31.4

        plt.scatter(mag_true, mag_meas - mag_true)
        plt.xlim(15, 25)
        plt.ylim(-1, 1)
        plt.xlabel('mag_i_true')
        plt.xlabel('mag_i_meas')
        plt.show()

    return data_ref, data_target
