from timer import Timer
from os.path import exists, join
from os import makedirs


class Folder:
    run_id = str(Timer.start.strftime('%s'))
    DST = "runs/run_" + run_id
    if not exists(DST):
        makedirs(DST)
    DST_ARC = join(DST, "archive")
    DST_IND = join(DST, "inds")
