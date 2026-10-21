# Patches applied to the third-party baselines

The reported MoMoK and TSAM rows were produced with the upstream code plus the small local patches below. The
upstream repositories are not redistributed; fetch them at the stated commit and apply the patch.

| baseline | upstream | commit | patch | what it does |
|---|---|---|---|---|
| MoMoK | https://github.com/zjukg/MoMoK | `99c2df114d48c79708ea0608644b685183d81bd9` | `momok.diff` (2 files, +6/-1 lines) | text-segment count derived from the text width (needed for the 768-d text of ViProKG; unchanged for 384-d); optional per-triple rank dump (ranking arithmetic unchanged) |
| TSAM | https://github.com/2391134843/TSAM | `81b1b4ab297f01b12a82e7124ac9f25ed703196f` | `tsam/merge_tokens.diff` | pad entity token lists to `max_num` with a mask; per-dataset vocabulary/pad ids (ViProKG: 64001 / 8255; W/Y keep BERT defaults 30522 / 14999) |

```bash
git clone https://github.com/zjukg/MoMoK && cd MoMoK && git checkout 99c2df114d48c79708ea0608644b685183d81bd9
git apply /path/to/release/baselines/patches/momok.diff

git clone https://github.com/2391134843/TSAM && cd TSAM && git checkout 81b1b4ab297f01b12a82e7124ac9f25ed703196f
git apply /path/to/release/baselines/patches/tsam/merge_tokens.diff
```

Verification done for this release: both patches apply cleanly to the stated commits, and the patched files are
AST-identical to the files that were actually run.

## License of the patches

The lines added by these diffs were written by the authors of this repository and are released under the MIT license
(`../../LICENSE`). The context and removed lines belong to the upstream projects, which publish no license (GitHub API,
17/10/2026); no license is granted here for the upstream code. The files are the authors' own `merge_tokens.py` edits
only in the sense of the added lines: the file as a whole is upstream code, which is why it is shipped as a diff and
not as a `.py` file.
