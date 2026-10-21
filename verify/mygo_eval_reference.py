"""Independent reference implementation of the MyGO filtered-ranking evaluation.

SOURCE
    Repository : https://github.com/zjukg/MyGO        (local mirror: repos/MyGO-main)
    File       : utils.py
    Functions  : calculate_rank, metrics
    Paper      : Zhang et al., "Tokenization, Fusion, and Augmentation: Towards Fine-grained
                 Multi-modal Entity Representation", AAAI 2025
                 (BibTeX key in the repo README: DBLP:conf/aaai/ZhangCGXHLZC25)
    License    : see the upstream repository.

COMMIT (resolved 2026-10-02 via the GitHub API, not by guessing)
    The local mirror has no .git directory, so the hash was resolved in two steps:
      1. git blob SHA-1 of the local `repos/MyGO-main/utils.py`
             = 1cc2cb2db90f3c56e1810df29a7ba598e56bd9b1   (458 bytes)
         GitHub API `/repos/zjukg/MyGO/contents/utils.py` returns the SAME blob sha and size,
         so the local copy is byte-identical to the file on the default branch.
      2. GitHub API `/repos/zjukg/MyGO/commits?path=utils.py` gives the last commit that touched
         it: 50276d8d2bdb43496ca8685c748c59b64934efd4, 2024-04-16, "update code and data",
         consistent with the README's latest dated entry (2024-04).

VERBATIM
    The two functions below are copied **verbatim**, byte for byte, from the source file. They are
    NOT rewritten, reformatted or "improved". Tabs, spacing and integer division are preserved,
    because the integer division is exactly the tie-break policy under test.

TIE-BREAK POLICY OF THIS REFERENCE (read off the code, not assumed)
    rank = #{score > score_target} + #{score == score_target} // 2 + 1
    That is the **mean rank** convention: a tie of k candidates (including the target) contributes
    floor(k/2) to the rank, i.e. halfway between the optimistic (#> + 1) and the pessimistic
    (#> + #= ) conventions. Any evaluator using a stable argsort instead will differ on ties, and
    that difference is what check V2 quantifies.
"""

MYGO_COMMIT = "50276d8d2bdb43496ca8685c748c59b64934efd4"  # last commit touching utils.py
MYGO_UTILS_BLOB_SHA1 = "1cc2cb2db90f3c56e1810df29a7ba598e56bd9b1"  # == GitHub API, 458 B
MYGO_SOURCE_URL = "https://github.com/zjukg/MyGO/blob/main/utils.py"

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
