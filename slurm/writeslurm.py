import os
from modelling_research.multiprofit_run import get_flags

flags_default = get_flags()


def get_cmds(path=None, toolset='devtoolset-8'):
    cmds = [
        f'source scl_source enable {toolset}',
        'source /software/lsstsw/stack3/loadLSST.bash',
        'setup lsst_distrib',
    ]
    if path is not None:
        cmds.append(f'cd {path}')
    cmds.append('')
    return cmds


def get_slurm_headers(time='96:00:00', queue='normal', n_tasks=1):
    headers = [
        f'-p {queue}',
        f'--time={time}',
        '-N 1',
        f'-n {n_tasks}',
        '',
    ]
    return headers


def get_mpf_fitcosmos(filename, idx_start, idx_end, src,
                      path_catalog='/project/dtaranu/cosmos/hst/COSMOS_25.2_training_sample/',
                      filename_catalog='real_galaxy_catalog_25.2.fits',
                      modelspecfile='${MULTIPROFIT_PATH}/examples/modelspecs-all-psfg2.csv',
                      model_name_hst2hsc='mg8sermpx',
                      bands_hsc='HSC-I', filename_log=None):
    if filename_log is None:
        filename_log = filename
    return 'python ${MULTIPROFIT_PATH}/examples/fitcosmos.py --catalogpath ' + path_catalog + \
           ' --catalogfile ' + filename_catalog + ' --indices {},{} --fit{} 1 --model_name_hst2hsc {} '.format(
                idx_start, idx_end, src, model_name_hst2hsc) + \
           '--modelspecfile {} --file {}_pickle.dat '.format(modelspecfile, filename) + \
           '--bands_hsc {} --redo 0 >> {}.log 2>>{}.err'.format(bands_hsc, filename_log, filename_log)


def get_mpf_task(name_file, name_log=None, **kwargs):
    if name_log is None:
        name_log = name_file
    cmd = f'python ${{MODELLING_RESEARCH_PATH}}/multiprofit_run.py'
    for arg, value in kwargs.items():
        if arg not in flags_default:
            raise RuntimeWarning(f'{arg} not in default flag list')
        cmd = f'{cmd} --{arg} {value}'
    cmd = f'{cmd} >> {name_log}.log 2>> {name_log}.err'
    return cmd


def write_slurm(filename, cmds, sbatches, header='#!/bin/bash'):
    with open(filename, 'w') as f:
        f.write(header + os.linesep)
        f.writelines("#SBATCH " + sbatch + os.linesep for sbatch in sbatches)
        f.writelines(cmd + os.linesep for cmd in cmds)


def write_slurm_fitcosmos(path, src='hsc', filename_src=None, idx_start=0, idx_end=2400, num=50,
                          prefix='cosmos_25.2_fits_', postfix='_psfg2',
                          cpus_multiprog=None, **kwargs):
    if filename_src is None:
        filename_src = src
    if cpus_multiprog:
        tasks = cpus_multiprog
        name_multiprog = prefix + src + '_' + str(idx_start) + '-' + str(idx_end - 1) + postfix + '.bash'
        with open(os.path.join(path, name_multiprog), 'w') as f:
            f.write(get_mpf_fitcosmos(
                prefix + filename_src + '_${1}-${2}' + postfix, '${1}', '${2}',
                src, filename_log=prefix + src + '_${1}-${2}' + postfix, **kwargs) + os.linesep
            )
    else:
        tasks = 1
    cmds = get_cmds()
    sbatches = get_slurm_headers()
    idx = idx_start
    while idx < idx_end:
        idx_end_task = idx + tasks*num - 1
        idx_range = f'_{idx}-{idx_end_task}'
        name = f'{src}{idx_range}'
        sbatches[-1] = '--job-name ' + name
        name_full = f'{prefix}{name}{postfix}'
        file = os.path.join(path, name_full + '.job')
        if cpus_multiprog is not None:
            filename_conf = os.path.join(path, f'{name_full}.conf')
            cmds[-1] = f'srun --output {name_full}.log --ntasks={tasks} --multi-prog {filename_conf}'
            with open(filename_conf, 'w') as f:
                f.writelines(' '.join(
                    [str(task), 'bash', name_multiprog, str(idx + task*num), str(idx + (task+1)*num - 1)]) +
                             os.linesep for task in range(tasks))

        else:
            cmds[-1] = get_mpf_fitcosmos(
                f'{prefix}{filename_src}{idx_range}{postfix}', idx, idx_end_task,
                src, filename_log=name_full, **kwargs)
        write_slurm(file, cmds, sbatches)
        idx = idx_end_task + 1


def write_slurm_fit_mpftask(path, patches=None, multiprog=False, prefix='mpf', **kwargs):
    """Writes slurm files for batch processing of MultiProfitTask.

    Parameters
    ----------
    path : `str`
        File path to write all files.
    patches : `iterable` [`str`]
        An iterable butler of patch strings; default ['4,4'].
    multiprog : bool
        Whether to use multi-prog to process patches simultaneously.
    prefix : `str`
        A prefix for output filenames; default `mpf`.
    kwargs
        Additional keyword arguments to pass to get_mpf_task.

    """
    if patches is None:
        patches = ['4,4']
    if multiprog:
        name_multiprog = f'{prefix}_{patches[0]}-{patches[-1]}'
        with open(os.path.join(path, f'{name_multiprog}.bash'), 'w') as f:
            f.write(get_mpf_task(f'{prefix}_${{1}}', name_patch='${1}', filenameOut=f'{prefix}_${{1}}.fits',
                                 **kwargs))
        cmds_conf = []
    cmds = get_cmds()
    sbatches = get_slurm_headers()
    for idx, patch in enumerate(patches):
        if multiprog:
            cmd_multi = f'{idx} bash {name_multiprog}.bash {patch}{os.linesep}'
            cmds_conf.append(cmd_multi)
        else:
            prefix_patch = f'{prefix}_{patch}'
            cmds[-1] = get_mpf_task(prefix_patch, name_patch=patch, filenameOut=f'{prefix_patch}.fits',
                                    **kwargs)
            filename_job = os.path.join(path, f'{prefix_patch}.job')
            write_slurm(filename_job, cmds, sbatches)
    if multiprog:
        filename_conf = os.path.join(path, f'{name_multiprog}.conf')
        with open(filename_conf, 'w') as f:
            f.writelines(cmds_conf)
        cmds[-1] = f'srun --output {name_multiprog}.log --ntasks={len(patches)} --multi-prog {filename_conf}'
        filename_job = os.path.join(path, f'{name_multiprog}.job')
        write_slurm(filename_job, cmds, sbatches)

