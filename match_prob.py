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

__all__ = ['ProbabilisticMatcherConfig', 'ProbabilisticMatcher']

from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree
import time
from typing import Tuple

logger_default = logging.getLogger(__name__)


def _radec_to_xyz(ra, dec):
    """Convert input ra/dec coordinates to spherical unit vectors.

    Parameters
    ----------
    ra, dec: `numpy.ndarray`
        Arrays of right ascension/declination in degrees.

    Returns
    -------
    vectors : `numpy.ndarray`, (N, 3)
        Output unit vectors.
    """
    if ra.size != dec.size:
        raise ValueError('ra and dec must be same size')
    ras = np.radians(ra)
    decs = np.radians(dec)
    vectors = np.empty((ras.size, 3))

    sin_dec = np.sin(np.pi / 2 - decs)
    vectors[:, 0] = sin_dec * np.cos(ras)
    vectors[:, 1] = sin_dec * np.sin(ras)
    vectors[:, 2] = np.cos(np.pi / 2 - decs)

    return vectors


class ProbabilisticMatcherConfig(BaseModel):
    column_order: str = Field(
        default=...,
        description='The reference table column defining the order for matching reference sources',
    )
    column_ref_coord1: str = Field(
        default=...,
        description='The reference table column for the first spatial coordinate (usually x or ra).',
    )
    column_ref_coord2: str = Field(
        default=...,
        description='The reference table column for the second spatial coordinate (usually y or dec).'
                    'Units must match column_ref_coord1.',
    )
    column_target_coord1: str = Field(
        default=...,
        description='The target table column for the first spatial coordinate (usually x or ra).'
                    'Units must match column_ref_coord1.',
    )
    column_target_coord2: str = Field(
        default=...,
        description='The target table column for the second spatial coordinate (usually y or dec).'
                    'Units must match column_ref_coord2.',
    )
    columns_ref_meas: Tuple[str, ...] = Field(
        default=...,
        description='The reference table columns to compute match likelihoods from '
                    '(usually centroids and fluxes/magnitudes)',
    )
    columns_target_meas: Tuple[str, ...] = Field(
        default=...,
        description='Target table columns with measurements corresponding to columns_ref_meas',
    )
    columns_target_err: Tuple[str, ...] = Field(
        default=...,
        description='Reference table columns with standard errors (sigma) corresponding to columns_ref_meas',
    )
    coords_spherical: bool = Field(
        default=True,
        description='Whether column_*_coord[12] are spherical coordinates (ra/dec) or not (pixel x/y)',
    )
    coords_ref_factor: float = Field(
        default=1.0,
        description='Multiplicative factor for reference catalog coordinates.'
                    'If coords_spherical is true, this must be the number of degrees per unit increment of '
                    'column_ref_coord[12]. Otherwise, it must convert the coordinate to the same units'
                    ' as the target coordinates.',
    )
    coords_target_factor: float = Field(
        default=1.0,
        description='Multiplicative factor for target catalog coordinates.'
                    'If coords_spherical is true, this must be the number of degrees per unit increment of '
                    'column_target_coord[12]. Otherwise, it must convert the coordinate to the same units'
                    ' as the reference coordinates.',
    )

    logging_n_rows: int = Field(
        default=None,
        description='Number of matches to make before outputting incremental log message.',
    )

    match_dist_max: float = Field(
        default=...,
        description='Maximum match distance. Units must be arcseconds if coords_spherical, '
                    'or else match those of column_*_coord[12] multiplied by coords_*_factor.',
    )
    match_n_max: int = Field(
        default=10,
        description='Maximum number of spatial matches to consider (in ascending distance order).',
    )
    match_n_finite_min: int = Field(
        default=3,
        description='Minimum number of fields with a finite value to measure match likelihood',
    )

    order_ascending: bool = Field(
        default=False,
        description='Whether to order reference match candidates in ascending order of column_order '
                    '(should be False if the column is a flux and True if it is a magnitude.',
    )


@dataclass
class CatalogExtras:
    """Store frequently-reference (meta)data revelant for matching a catalog.

    Parameters
    ----------
    catalog : `pandas.DataFrame`
        A pandas catalog to store extra information for.
    select : `numpy.array`
        A numpy boolean array of the same length as catalog to be used for
        target selection.
    """
    n: int
    indices: np.array
    select: np.array

    coordinate_factor: float = None

    def __init__(self, catalog: pd.DataFrame, select: np.array = None, coordinate_factor: float = None):
        self.n = len(catalog)
        self.select = select
        self.indices = np.flatnonzero(select) if select is not None else None
        self.coordinate_factor = coordinate_factor


def _mul_column(column: np.array, value: float):
    if value is not None and value != 1:
        column *= value
    return column


class ProbabilisticMatcher:
    """A probabilistic, greedy catalog matcher.

    Parameters
    ----------
    config: `ProbabilisticMatcherConfig`
        A configuration instance.
    """
    config: ProbabilisticMatcherConfig

    def match(
            self,
            catalog_ref: pd.DataFrame,
            catalog_target: pd.DataFrame,
            select_ref: np.array = None,
            select_target: np.array = None,
            logger: logging.Logger = None,
    ):
        """Match catalogs.

        Parameters
        ----------
        catalog_ref : `pandas.DataFrame`
            A reference catalog to match in order of a given column (i.e. greedily).
        catalog_target : `pandas.DataFrame`
            A target catalog for matching sources from `catalog_ref`. Must contain measurements with errors.
        select_ref : `numpy.array`
            A boolean array of the same length as `catalog_ref` selecting the sources that can be matched.
        select_target : `numpy.array`
            A boolean array of the same length as `catalog_target` selecting the sources that can be matched.
        logger : `logging.Logger`
            A Logger for logging.

        Returns
        -------
        catalog_out_ref : `pandas.DataFrame`
            A catalog of identical length to `catalog_ref`, containing match information for rows selected by
            `select_ref` (including the matching row index in `catalog_target`).
        catalog_out_target : `pandas.DataFrame`
            A catalog of identical length to `catalog_target`, containing the indices of matching rows in
            `catalog_ref`.
        exceptions : `dict` [`int`, `Exception`]
            A dictionary keyed by `catalog_target` row number of the first exception caught when matching.
        """
        if logger is None:
            logger = logger_default

        config = self.config
        extras_ref, extras_target = (
            CatalogExtras(catalog, select=select, coordinate_factor=coord_factor)
            for catalog, select, coord_factor in zip(
                (catalog_ref, catalog_target),
                (select_ref, select_target),
                (config.coords_ref_factor, config.coords_target_factor),
            )
        )
        n_ref_match, n_target_match = (len(x) for x in (extras_ref.indices, extras_target.indices))

        (coord1_ref, coord2_ref), (coord1_target, coord2_target) = (
            # Confused as to why this needs to be a list to work properly
            [
                _mul_column(catalog.loc[extras.select, column].values, extras.coordinate_factor)
                for column in columns
            ]
            for catalog, extras, columns in (
                (catalog_ref, extras_ref, (config.column_ref_coord1, config.column_ref_coord2)),
                (catalog_target, extras_target, (config.column_target_coord1, config.column_target_coord2)),
            )
        )
        if config.coords_spherical:
            vec_ref = _radec_to_xyz(coord1_ref, coord2_ref)
            vec_target = _radec_to_xyz(coord1_target, coord2_target)
        else:
            vec_ref = np.vstack((coord1_ref, coord2_ref))
            vec_target = np.vstack((coord1_target, coord2_target))

        logger.info(f'Generating cKDTree with match_n_max={config.match_n_max}')
        tree_obj = cKDTree(vec_target)

        scores, idxs_target = tree_obj.query(
            vec_ref,
            distance_upper_bound=(
                np.radians(config.match_dist_max) if config.coords_spherical
                else config.match_dist_max
            ),
            k=config.match_n_max,
        )
        n_matches = np.sum(idxs_target != n_target_match, axis=1)
        n_matched_max = np.sum(n_matches == config.match_n_max)
        if n_matched_max > 0:
            logger.warning(f'{n_matched_max}/{n_ref_match} ({100. * n_matched_max / n_ref_match:.2f}%)'
                           f' true objects have n_matches=n_match_max({config.match_n_max})')

        target_row_match = np.full(extras_target.n, np.nan, dtype=np.int64)
        ref_candidate_match = np.zeros(extras_ref.n, dtype=bool)
        ref_row_match = np.full(extras_ref.n, np.nan, dtype=int)
        ref_match_count = np.zeros(extras_ref.n, dtype=int)
        ref_match_meas_finite = np.zeros(extras_ref.n, dtype=int)
        ref_chisq = np.full(extras_ref.n, np.nan, dtype=float)

        column_order = catalog_ref.loc[extras_ref.indices, config.column_order].values
        order = np.argsort(column_order if config.order_ascending else -column_order)

        indices = extras_ref.indices[order]
        idxs_target = idxs_target[order]
        n_indices = len(indices)

        data_ref = catalog_ref.loc[indices, config.columns_ref_meas]
        data_target = catalog_target.loc[extras_target.select, config.columns_target_meas]
        errors_target = catalog_target.loc[extras_target.select, config.columns_target_err]

        exceptions = {}
        matched_target = set()

        t_begin = time.process_time()

        logger.info(f'Matching n_indices={n_indices}/{len(catalog_ref)}')
        for index_n, index_row in enumerate(indices):
            ref_candidate_match[index_row] = True
            found = idxs_target[index_n, :]
            found = [x for x in found[found != n_target_match] if x not in matched_target]
            n_found = len(found)
            if n_found > 0:
                chi = (
                    (data_target.iloc[found].values - data_ref.iloc[index_n].values)
                    / errors_target.iloc[found].values
                )
                finite = np.isfinite(chi)
                n_finite = np.sum(finite, axis=1)
                chisq_good = n_finite >= config.match_n_finite_min
                if np.any(chisq_good):
                    try:
                        chisq_sum = np.zeros(n_found, dtype=float)
                        chisq_sum[chisq_good] = np.nansum(chi[chisq_good, :] ** 2, axis=1)
                        idx_chisq_min = np.nanargmin(chisq_sum / n_finite)
                        idx_match = found[idx_chisq_min]
                        ref_match_meas_finite[index_row] = n_finite[idx_chisq_min]
                        ref_match_count[index_row] = len(chisq_good)
                        ref_chisq[index_row] = chisq_sum[idx_chisq_min]
                        row_target = extras_target.indices[idx_match]
                        ref_row_match[index_row] = row_target
                        target_row_match[row_target] = index_row
                        matched_target.add(idx_match)
                    except Exception as error:
                        exceptions[index_row] = error

            if config.logging_n_rows and ((index_n + 1) % config.logging_n_rows == 0):
                logger.info(f'Processed {index_n + 1}/{n_indices} in {time.process_time() - t_begin:.2f}s'
                            f' at sort value={column_order[order[index_n]]:.3f}')

        catalog_out_ref = pd.DataFrame({
            'match_candidate': ref_candidate_match,
            'match_row': ref_row_match,
            'match_count': ref_match_count,
            'match_chisq': ref_chisq,
            'match_n_chisq_finite': ref_match_meas_finite,
        })

        catalog_out_target = pd.DataFrame({
            'match_row': target_row_match,
        })

        return catalog_out_ref, catalog_out_target, exceptions

    def __init__(
            self,
            config: ProbabilisticMatcherConfig,
    ):
        self.config = config
