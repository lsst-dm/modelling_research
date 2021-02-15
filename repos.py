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

import glob
from lsst.daf.persistence import Butler
import lsst.sphgeom as sphgeom
import numpy as np
import os
import subprocess


def find_copy_single_file(
        path_in, path_out, filename_glob_in, func_filename_out=None, write=True, overwrite=False, link=False, **kwargs
):
    """ Find via glob and copy or link a single file to a new destination.

    Parameters
    ----------
    path_in : `str`
        A valid directory path to search in.
    path_out : `str`
        A valid directory path to output to.
    filename_glob_in : `str`
        The filename glob to search for within `path_in`.
    func_filename_out : `function`
        A function that takes an input filename and kwargs and returns an output filename.
    write : `bool`
        Whether to actually write (copy or symlink) the target file at the destination.
    overwrite : `bool`
        Whether to overwrite a file if it already exists at the destination.
    link : `bool`
        Whether to create a symlink at the destination instead of copying.
    kwargs : `dict`
        Additional arguments to pass to `func_filename_out`.

    Returns
    -------
    filename : `str`
        The full path of the output file.
    """
    if not os.path.isdir(path_in) and os.path.isdir(path_out):
        raise FileNotFoundError(f'path_in={path_in} and/or path_out={path_out} are not directories')
    filename_in = glob.glob(os.path.join(path_in, filename_glob_in))
    if len(filename_in) != 1:
        raise RuntimeError(f'Got unexpected matches={filename_in} for filename_glob={filename_glob_in}')
    else:
        filename_in = filename_in[0]
    filename = filename_in.split(path_in)
    if len(filename) != 2:
        raise RuntimeError(
            f'Got unexpected filename={filename} split on {path_in} for filename_glob={filename_glob_in}')
    else:
        filename = filename[1]
    filename_out = filename
    if func_filename_out is not None:
        filename_out = func_filename_out(filename_out, **kwargs)
    filename_out = f'{path_out}{filename_out}'
    cmd_base, flag = ('ln', '-s') if link else ('cp', '-p')
    if write:
        if (not overwrite) and os.path.exists(filename_out):
            print(f'Skipping {filename_out} as it already exists')
        else:
            dirname = os.path.dirname(filename_out)
            cmds = []
            if not os.path.isdir(dirname):
                cmds.append(["mkdir", "-p", dirname])
            cmds.append([cmd_base, flag, filename_in, filename_out])
            for cmd in cmds:
                try:
                    output = subprocess.check_output(cmd)
                    if output:
                        print(output)
                except subprocess.CalledProcessError as err:
                    print(f"{' '.join(cmd)} returned error: {err}")
                    raise err
    return filename


def format_visit(visit):
    return f'{visit:08d}'


def format_ccd(ccd):
    return f'{ccd:03d}'


def format_path_flat(band, date, raft, sensor, det):
    return os.path.join(band, date, f'flat_{band}-{raft}-{sensor}-{det}_{date}.fits')


def parse_path_flat(path):
    band, date, filename = (path.lstrip(os.sep)).split(os.sep)
    print(f'parsing filename={filename}')
    prefix, raft, sensor, postfix = filename.split('-', maxsplit=3)
    flat, band_file = prefix.split('_')
    if flat != 'flat' or band_file != band:
        raise RuntimeError(
            f'Invalid flat path={path} where {flat}!="flat" and/or directory band={band} != filename band={band_file}'
        )
    det, postfix = postfix.split('_')
    date_flat, extension = postfix.split('.')
    if date_flat != date:
        raise RuntimeError(f'Mismatched path={path} date={date} != filename date={date_flat}')
    if extension != 'fits':
        raise RuntimeError(f'Path={path} extension={extension}!=fits')
    return band, date, raft, sensor, det


def reformat_path_flat(
        path_reformat, band_out=None, date_out=None, raft_out=None, sensor_out=None, det_out=None, prefix=None,
):
    if prefix is None:
        prefix = ''
    band, date, raft, sensor, det = parse_path_flat(path_reformat)
    if band_out is not None:
        band = band_out
    if date_out is not None:
        date = date_out
    if raft_out is not None:
        raft = raft_out
    if sensor_out is not None:
        sensor = sensor_out
    if det_out is not None:
        det = det_out
    return f'{prefix}{format_path_flat(band, date, raft, sensor, det)}'


def validate_filename_raw(filename_raw, ccd_visit):
    visit, raft, filename = filename_raw.split(os.sep)
    if visit != str(ccd_visit['visit']):
        raise RuntimeError(f'{filename_raw} derived visit={visit} doesn\'t match ccd_visit={ccd_visit}')
    visit_full, raft_full, sensor, filename_det = filename.split('-')
    if visit_full.lstrip('0') != visit or visit_full != format_visit(int(visit)):
        raise RuntimeError(f'{filename} derived visit={visit_full} doesn\'t match ccd_visit={ccd_visit}')
    if raft_full != raft:
        raise RuntimeError(f'{filename} derived raft={raft_full} != raft={raft} for filename_raw={filename_raw}')
    blank1, filename_ccd = filename_det.split('det')
    ccd, blank2 = filename_ccd.split('.fits')
    if (blank1 != '') or (blank2 != '') or ccd != format_ccd(ccd_visit['ccd']):
        raise RuntimeError
    return raft, sensor


def copy_repo_subset(
        path_in, path_out, butler_coadd, tract_patches, filters,
        n_top_goodpix=3, n_visits_max=2, filter_flat_override=None,
        write=True, overwrite=False, link=False,
):
    """ Select and copy a subset of an ingested repository to another path for re-ingesting.

    Parameters
    ----------
    path_in : `str`
        The path to the input repository.
    path_out : `str`
        The output path.
    butler_coadd : `lsst.daf.persistence.Butler`
        A Gen2 (for now) Butler to get a skymap and coadds (so as to retrieve input visits).
    tract_patches : `dict` [`int`, iterable [`str`]]
        A dict with an iterable of patch names, keyed by tract number.
    filters : iterable [`str`]
        An iterable of filter names.
    n_top_goodpix : `int`
        The number of ccd-visits per band with the largest goodpix to select.
    n_visits_max : `int`
        The maximum number of additional visits to select from the same CCDs selected by `n_top_goodpix`.
    filter_flat_override : `str`
        A override filter path to output flats to, in case they are only generated in one filter.
    write : `bool`
        Whether to actually attempt to write files, or just print output.
    overwrite : `bool`
        Whether to overwrite existing files.
    link : `bool`
        Whether to create symbolic links instead of copying files.

    """
    butler_in = Butler(path_in)
    htm_pixelization = sphgeom.HtmPixelization(butler_in.get("ref_cat_config", name='cal_ref_cat').indexer['HTM'].depth)
    skymap = butler_coadd.get('deepCoadd_skyMap')

    path_in_calib, path_in_raw, path_in_refcat, path_out_calib, path_out_raw, path_out_refcat = (
        os.path.join(path_base, subdir) for path_base in (path_in, path_out) for subdir in ('CALIB', 'raw', 'ref_cats')
    )
    paths_calib_in, paths_calib_out = (
        {subdir: os.path.join(path_base, subdir) for subdir in ('bfkernels', 'bias', 'dark', 'flat', 'SKY')}
        for path_base in (path_in_calib, path_out_calib)
    )
    path_in_refcat_cal, path_out_refcat_cal = (
        os.path.join(path_base, 'cal_ref_cat') for path_base in (path_in_refcat, path_out_refcat)
    )
    paths_out = [path_out, path_out_calib, path_out_raw, path_out_refcat, path_out_refcat_cal] + list(
        paths_calib_out.values())
    for path_out in paths_out:
        if not os.path.exists(path_out):
            os.mkdir(path_out)
    for filename in ('config.py', 'master_schema.fits'):
        find_copy_single_file(path_in_refcat_cal, path_out_refcat_cal, filename)

    for tract, patches in tract_patches.items():
        skymap_tract = skymap[tract]
        wcs = skymap_tract.getWcs()
        for patch in patches:
            patch_id = tuple(int(x) for x in (patch.split(',')))
            polygon = skymap_tract[patch_id].getOuterSkyPolygon(wcs)
            ranges_refcat = htm_pixelization.envelope(polygon)
            for range_min, range_max in ranges_refcat:
                for idx_refcat in range(range_min, range_max):
                    find_copy_single_file(
                        path_in_refcat_cal, path_out_refcat_cal, f'{idx_refcat}.fits',
                        write=write, overwrite=overwrite, link=link,
                    )

            for band in filters:
                band_flat = filter_flat_override if filter_flat_override is not None else band
                coadd = butler_coadd.get('deepCoadd_calexp', dict(tract=tract, patch=patch, filter=band))
                visits = coadd.getInfo().getCoaddInputs()
                ccds = visits.ccds
                id_ccd, counts = np.unique(ccds['ccd'], return_counts=True)

                # Find CCDs with multiple visits so the calibs can be reused
                repeats = id_ccd[np.where(counts >= n_visits_max)[0]]
                repeats = [ccd for ccd in ccds if ccd['ccd'] in repeats]
                ccds_goodpix = np.argsort([ccd['goodpix'] for ccd in repeats])
                ccds_selected = set()
                for ccd_idx in reversed(ccds_goodpix):
                    ccds_selected.add(repeats[ccd_idx]['ccd'])
                    if len(ccds_selected) == n_top_goodpix:
                        break

                ccds_visits = []
                for ccd_selected in ccds_selected:
                    ccd_visits = [ccd for ccd in repeats if ccd['ccd'] == ccd_selected]
                    for idx in np.argsort([ccd['goodpix'] for ccd in ccd_visits])[-n_visits_max:]:
                        ccds_visits.append(ccd_visits[idx])
                print(f'Copying the following visit, ccd, goodpix for {band} band:')
                print([(ccd['visit'], ccd['ccd'], ccd['goodpix']) for ccd in ccds_visits])
                for ccd_visit in ccds_visits:
                    ccd, visit = format_ccd(ccd_visit['ccd']), f"{ccd_visit['visit']:d}"
                    filename_glob = os.path.join(visit, 'R*', f'*-det{ccd}.fits')
                    # Copy the raw and parse its filename sans leading slash
                    filename_full = find_copy_single_file(
                        path_in_raw, path_out_raw, filename_glob,
                        write=write, overwrite=overwrite, link=link,
                    ).lstrip(os.sep)
                    raft, sensor = validate_filename_raw(filename_full, ccd_visit)
                    print(f"Validated filename={filename_full} for visit, ccd = {visit}, {ccd}")
                    filenames_glob = {
                        'bfkernels': f'bfKernel-{raft}-{sensor}-det{ccd}.pkl',
                        'bias': os.path.join('*', f'bias-{raft}-{sensor}-det{ccd}_*-*-*.fits'),
                        'dark': os.path.join('*', f'dark-{raft}-{sensor}-det{ccd}_*-*-*.fits'),
                        'flat': os.path.join(band, '*-*-*', f'flat_{band}-{raft}-{sensor}-det{ccd}_*.fits'),
                        'SKY': os.path.join('*-*-*', band, f'SKY-*-*-*-{band}-{raft}-{sensor}-det{ccd}_*-*-*.fits'),
                    }
                    for type_calib, filename_glob_in in filenames_glob.items():
                        if type_calib == 'flat' and (band_flat != band):
                            func_filename_out = reformat_path_flat
                            path_kwargs = {'band_out': band_flat, 'prefix': os.sep}
                        else:
                            func_filename_out = None
                            path_kwargs = {}
                        find_copy_single_file(
                            paths_calib_in[type_calib], paths_calib_out[type_calib], filename_glob_in,
                            func_filename_out=func_filename_out, write=write, overwrite=overwrite, link=link,
                            **path_kwargs,
                        )
