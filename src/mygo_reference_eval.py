"""Reference ranking functions of the MyGO evaluation.

Source : https://github.com/zjukg/MyGO/blob/main/utils.py (functions `calculate_rank`, `metrics`)
Blob   : git blob sha1 1cc2cb2db90f3c56e1810df29a7ba598e56bd9b1 (458 bytes), last commit
         50276d8d2bdb43496ca8685c748c59b64934efd4
The two functions below are reproduced verbatim for equivalence testing. The upstream repository publishes no
license; see baselines/NOTICE. Tie policy of the reference: rank = #{score > target} + #{score == target} // 2 + 1.
"""

MYGO_COMMIT = "50276d8d2bdb43496ca8685c748c59b64934efd4"  # last commit touching utils.py
MYGO_UTILS_BLOB_SHA1 = "1cc2cb2db90f3c56e1810df29a7ba598e56bd9b1"  # == GitHub API, 458 B
MYGO_SOURCE_URL = "https://github.com/zjukg/MyGO/blob/main/utils.py"

# fmt: off
# ----------------------------------------------------------------------------------------------
# BEGIN VERBATIM COPY FROM zjukg/MyGO utils.py  (do not edit)
# ----------------------------------------------------------------------------------------------
import numpy as np


def calculate_rank(score, target, filter_list):
	score_target = score[target]
	score[filter_list] = score_target - 1
	rank = np.sum(score > score_target) + np.sum(score == score_target) // 2 + 1
	return rank

def metrics(rank):
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank < 11) / len(rank)
    hit3 = np.sum(rank < 4) / len(rank)
    hit1 = np.sum(rank < 2) / len(rank)
    return mr, mrr, hit10, hit3, hit1
# ----------------------------------------------------------------------------------------------
# END VERBATIM COPY
# ----------------------------------------------------------------------------------------------
# fmt: on
