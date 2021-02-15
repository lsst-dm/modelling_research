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

from lsst.validate.drp.validate import runOneRepo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def getDataIdsHscQuick():
    """Get the ci_hsc data IDs used in validate_drp's runHscQuick script.

    Returns
    -------
    dataIds : `list [`dict`]
        A list of valid data ID dicts.
    """
    # Output from discoverDataIds(base)
    return [
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903334,
         'ccd': 23,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903334,
         'ccd': 22,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903334,
         'ccd': 16,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903334,
         'ccd': 100,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903336,
         'ccd': 24,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903336,
         'ccd': 17,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903338,
         'ccd': 25,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903338,
         'ccd': 18,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903342,
         'ccd': 100,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903342,
         'ccd': 10,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903342,
         'ccd': 4,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903344,
         'ccd': 11,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903344,
         'ccd': 5,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903344,
         'ccd': 0,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903346,
         'ccd': 12,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903346,
         'ccd': 6,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 533,
         'filter': 'HSC-R',
         'visit': 903346,
         'ccd': 1,
         'field': 'STRIPE82L',
         'dateObs': '2013-06-17',
         'taiObs': '2013-06-17',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903986,
         'ccd': 23,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903986,
         'ccd': 22,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903986,
         'ccd': 16,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903986,
         'ccd': 100,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903988,
         'ccd': 24,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903988,
         'ccd': 23,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903988,
         'ccd': 17,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903988,
         'ccd': 16,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903990,
         'ccd': 25,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 903990,
         'ccd': 18,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 904010,
         'ccd': 100,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 904010,
         'ccd': 10,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 904010,
         'ccd': 4,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 904014,
         'ccd': 12,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 904014,
         'ccd': 6,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
        {'pointing': 671,
         'filter': 'HSC-I',
         'visit': 904014,
         'ccd': 1,
         'field': 'STRIPE82L',
         'dateObs': '2013-11-02',
         'taiObs': '2013-11-02',
         'expTime': 30.0},
    ]


def getJobs(repos, dataIds=None):
    """Get the results of validate_drp jobs run on the given repos.

    Parameters
    ----------
    repos : `dict` [`str`, `str`]
        Dict of repo paths keyed by name.
    dataIds : `list [`dict`]
        A list of valid data ID dicts, e.g. as returned by `getDataIdsHscQuick`.

    Returns
    -------
    jobs : `dict` [`str`]
        Dict of validate_drp results keyed by name.
    """
    if dataIds is None:
        dataIds = getDataIdsHscQuick()
    jobs = {key: runOneRepo(repo, dataIds=dataIds) for key, repo in repos.items()}
    return jobs


def plot(jobs):
    """Plot repeatability results for different types of models and sources.

    Parameters
    ----------
    jobs : `dict` [`str`]
        Results of validate_drp runs keyed by name, as returned by `getJobs`.

    Notes
    -----
    The base model is currently base_gaussian. To run this, you'll need to run the validate_drp examples
    with and without meas_modelfit imports, or update validate_drp's runOneRepo method to pass kwargs all
    the way down to reduceSources's nameFluxKey arg (e.g. in DM-22138).
    """
    snrs = [
        (5, 10),
        (10, 20),
        (20, 40),
        (40, 80),
    ]
    types = ['Gal', 'Star']
    metrics_type = {
        key: [f'validate_drp.modelPhotRep{key}{idx + 1}' for idx in range(len(snrs))] for key in types
    }
    metrics_psf = {'Psf': [f'validate_drp.psfPhotRepStar{idx + 1}' for idx in range(len(snrs))]}
    colors = {
        'Star': 'blue',
        'Gal': 'red',
    }
    linestyles = {
        'base': '--',
        'mmf': '-',
        'Psf': ':',
    }
    sns.set_style('darkgrid')
    sns.set(rc={"lines.linewidth": 2.5})
    for band in jobs['base'].keys():
        measurements_repo = {key: job[band].measurements for key, job in jobs.items()}
        plt.figure(figsize=(10, 10))
        plt.xlabel('Min. SNR')
        plt.ylabel(f'{band} PA1 [mmag]')
        for model, measurements in measurements_repo.items():
            metrics_model = {**metrics_type, **metrics_psf} if model == 'base' else metrics_type
            for type_src, metrics in metrics_model.items():
                values = np.array([measurements[metric].datum.quantity.value for metric in metrics])
                label = f'{model} {type_src}' if type_src != 'psf' else type_src
                for idx, (x, y) in enumerate(zip(snrs, values)):
                    plt.plot(x, [y, y], label=label if idx == 0 else None,
                             linestyle=linestyles[model if type_src != 'Psf' else type_src],
                             color=colors['Star' if type_src is 'Psf' else type_src])
        plt.legend()
        plt.title('Photometric repeatability in ci_hsc')
        plt.ylim(bottom=0, top=250)
        plt.xlim(left=snrs[0][0], right=snrs[-1][1])
