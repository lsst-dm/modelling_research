import os


def writeslurm(filename, cmds, sbatches, header='#!/bin/bash'):
    with open(filename, 'w') as f:
        f.write(header + os.linesep)
        f.writelines("#SBATCH " + sbatch + os.linesep for sbatch in sbatches)
        f.writelines(cmd + os.linesep for cmd in cmds)


def _getmultiprofit(namefile, catalogpath, catalogfile, idx, idxend, src, model_name_hst2hsc, modelspecfile,
                    bands_hsc='HSC-I', namelog=None):
    if namelog is None:
        namelog = namefile
    return 'python ${MULTIPROFIT_PATH}/examples/fitcosmos.py --catalogpath ' + catalogpath + \
           ' --catalogfile ' + catalogfile + ' --indices {},{} --fit{} 1 --model_name_hst2hsc {} '.format(
                idx, idxend, src, model_name_hst2hsc) + \
           '--modelspecfile {} --file {}_pickle.dat '.format(modelspecfile, namefile) + \
           '--bands_hsc {} --redo 0 >> {}.log 2>>{}.err'.format(bands_hsc, namelog, namelog)


def writeslurmcosmos(path, src='hst', filenamesrc=None, start=0, end=2400, num=50,
                     catalogpath='/project/dtaranu/cosmos/hst/COSMOS_25.2_training_sample/',
                     catalogfile='real_galaxy_catalog_25.2.fits', model_name_hst2hsc='mg8sermpx',
                     modelspecfile='${MULTIPROFIT_PATH}/examples/modelspecs-all-psfg2.csv',
                     prefix='cosmos_25.2_fits_', postfix='_psfg2', bands_hsc='HSC-I',
                     time='48:00:00', queue='normal', multiprogcpus=None):
    if filenamesrc is None:
        filenamesrc = src
    cmds = [
        'source scl_source enable devtoolset-6',
        'source /software/lsstsw/stack3/loadLSST.bash',
        'setup lsst_distrib',
        'cd ' + path,
        '',
    ]
    if multiprogcpus:
        tasks = multiprogcpus
        multiprogname = prefix + src + '_' + str(start) + '-' + str(end-1) + postfix + '.bash'
        with open(os.path.join(path, multiprogname), 'w') as f:
            f.write(_getmultiprofit(
                prefix + filenamesrc + '_${1}-${2}' + postfix, catalogpath, catalogfile, '${1}', '${2}',
                src, model_name_hst2hsc, modelspecfile, bands_hsc=bands_hsc,
                namelog=prefix + src + '_${1}-${2}' + postfix) + os.linesep
            )
    else:
        tasks = 1
    sbatches = [
        '-p ' + queue,
        '--time=' + time,
        '-N 1',
        '-n {}'.format(tasks),
        '',
    ]
    idx = start
    while idx < end:
        idxend = idx + tasks*num - 1
        name = src + '_' + str(idx) + '-' + str(idxend)
        sbatches[-1] = '--job-name ' + name
        namefull = prefix + name + postfix
        namefile = prefix + filenamesrc + '_' + str(idx) + '-' + str(idxend) + postfix
        file = os.path.join(path, namefull + '.job')
        if multiprogcpus is not None:
            fileconf = os.path.join(path, namefull + '.conf')
            cmds[-1] = 'srun --output {}.log --ntasks={} --multi-prog {}'.format(namefull, tasks, fileconf)
            with open(fileconf, 'w') as f:
                f.writelines(' '.join(
                    [str(task), 'bash', multiprogname, str(idx + task*num), str(idx + (task+1)*num - 1)]) +
                             os.linesep for task in range(tasks))

        else:
            cmds[-1] = _getmultiprofit(
                namefile, catalogpath, catalogfile, idx, idxend, src, model_name_hst2hsc, modelspecfile,
                bands_hsc=bands_hsc, namelog=namefull)
        writeslurm(file, cmds, sbatches)
        idx = idxend + 1
