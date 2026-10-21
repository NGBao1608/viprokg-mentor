# ViProKG — dataset files

ViProKG is a Vietnamese multimodal knowledge graph: Wikidata triples, Wikimedia Commons image metadata and
Vietnamese Wikipedia text. This directory holds the files used by MENTOR and the baselines.
The directory name keeps the development name `ViMMKG2`; the public name of the dataset is ViProKG.

## Size

| Item | Count |
|---|---|
| Entities (8,207 graph entities + 48 attribute-value nodes) | 8,255 |
| Relations (72 relational + 9 attribute) | 81 |
| Triples | 35,602 |
| train / valid / test | 29,355 / 3,610 / 2,637 |
| Entities with an image | 7,790 |
| Attribute ("slice") triples in test | 987 (`test_lat.txt`) |

Images are **not** distributed; `data/final/ViMMKG2_core/manifest.tsv` logs, per image, the license, author and source page.

## Files

| File | Content |
|---|---|
| `entities.txt`, `relations.txt` | numeric entity / relation ids (0..N-1) |
| `entity2id.tsv`, `relation2id.tsv` | name ↔ id maps (`entity2id.tsv`: QID or attribute-value label, id, type) |
| `train.txt`, `valid.txt`, `test.txt` | numeric triples (head, relation, tail) |
| `*_thuong.txt`, `*_lat.txt` | the same splits restricted to relational ("thường") and attribute ("lát") triples; `test_phu_thuong.txt` is the auxiliary relational test subset (1,918 triples) |
| `lat_cand.tsv`, `lat_rel_names.tsv`, `lat_baseline.tsv` | candidate values, names and frequency baselines of the attribute relations |
| `texts_by_id.tsv` | entity text (Vietnamese Wikipedia summary + Wikidata description), keyed by id and QID |
| `ViMMKG2-textual-phobert-seg.json` | PhoBERT token ids of the word-segmented text, keyed by integer entity id |
| `image_index.tsv` | per entity: has_image, image tier, image kind, number of entities sharing the file, default-view flag |
| `image_paths.tsv` | source label and original path of each image (images themselves not shipped) |
| `V.pt`, `V_mask.pt` | image-token features and mask (`True` = no image) |
| `L_feat.pt` | lexical-channel features |
| `folds_k3.json` | 3-fold out-of-fold assignment of the training triples (seed 42) |

## License

Data: CC BY-SA 4.0 (see `LICENSE-DATA` at the repository root for scope and provenance of text and image metadata).
