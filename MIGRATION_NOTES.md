# Migration from the previous domain-aware patch

Old `protein_generator.pt` checkpoints are intentionally rejected. They contain the v2 domain-ID-only architecture and cannot be converted into the v3 sequence-conditioned architecture without retraining.

Re-run `download_domains.py` because the old `domain_pairs.csv` does not contain `domain_a_sequence`, `domain_b_sequence`, `left_context`, or `right_context`. Then run `train_compatibility.py` and `train_generator.py`.

The old `score` column was an almost continuous weighted average of correlated pLDDT terms. In v3, `absolute_score` is the comparable absolute 0–100 quality measure and `score` is the quantized within-pair percentile rank. Manuscript text and plots that refer to the previous score formula should therefore be updated before publication or competition submission.
