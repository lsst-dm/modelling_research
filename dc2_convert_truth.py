# Convert truth tables with string patches to integer
# Run in/from directory containing those tables
import lsst.daf.butler as dafButler

import glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

butler = dafButler.Butler('/repo/dc2', collections=['2.2i/defaults'])
skymap = butler.get('skyMap', skymap='DC2')

drop = True

files = glob.glob('truth_tract*.parquet')
for filename in files:
    print(f'Reading and transforming {filename}')
    tab = pq.read_table(filename)
    tab.drop(['match_objectId', 'match_sep', 'is_good_match', 'is_nearest_neighbor'])

    patches = tab['patch'].to_numpy()

    tract = int(filename[11:15])
    skymap_tract = skymap[tract]
    n_patches = np.prod(skymap_tract.getNumPatches())
    patches_int = np.full(len(patches), n_patches)

    for patch in skymap_tract:
        patch_val = ','.join(f'{x}' for x in patch.getIndex())
        patches_int[patches == patch_val] = skymap_tract.getSequentialPatchIndex(patch)

    n_unchanged = np.sum(patches_int == n_patches)
    if n_unchanged > 0:
        raise RuntimeError(f'Failed to set {n_unchanged} patch ints for filename={filename}')

    tab = tab.add_column(tab.schema.get_field_index('patch'), 'patch_int', pa.array(patches_int))
    tab = tab.remove_column(tab.schema.get_field_index('patch'))
    tab = tab.rename_columns([x if x != 'patch_int' else 'patch' for x in tab.column_names])
    # Re-write metadata
    tab = pa.Table.from_pandas(tab.to_pandas())

    filename_new = f'truth_patchint_tract{tract}.parq'
    print(f'Writing {filename_new}')
    pq.write_table(tab, filename_new)
    
