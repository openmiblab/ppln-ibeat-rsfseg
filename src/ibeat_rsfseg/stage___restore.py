import os
import logging

import dbdicom as db


def dixons(archivepath, datapath, group, site=None):
    datapath = os.path.join(datapath, 'dixon', 'stage_2_data')
    archivepath = os.path.join(archivepath, 'dixon', 'stage_2_data')
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, 'Controls')
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, 'Patients', site)
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitedatapath)


def segmentations(archivepath, datapath, group, site=None):
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, 'Controls')
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, 'Patients', site)
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitedatapath)



if __name__=='__main__':

    LOCALPATH = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    SHAREDPATH = r"G:\\Shared drives\iBEAt_Build"
    os.makedirs(LOCALPATH, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(LOCALPATH, 'error.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    dixons(SHAREDPATH, LOCALPATH, 'Controls')
    # dixons(SHAREDPATH, LOCALPATH, 'Patients', site='Sheffield')