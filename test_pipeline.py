import unittest

import torch

from domain_fusion import (
    Candidate,
    ConditionalLSTM,
    apply_scores,
    features,
    plddt_by_region,
    structural_score,
    ca_lddt,
)


class PipelineTests(unittest.TestCase):
    def test_features(self):
        f = features("AVILMFWYDEKRPGS")
        self.assertAlmostEqual(f["hydrophobic_fraction"], 8 / 15)
        self.assertAlmostEqual(f["charged_fraction"], 4 / 15)
        self.assertAlmostEqual(f["proline_fraction"], 1 / 15)
        self.assertAlmostEqual(f["gly_ser_fraction"], 2 / 15)

    def test_structural_score_uses_broader_components(self):
        score = structural_score(
            whole_plddt=80,
            junction_plddt=70,
            junction_drop=5,
            linker_plddt=65,
            domain_a_plddt=90,
            domain_b_plddt=85,
            domain_a_baseline_plddt=92,
            domain_b_baseline_plddt=88,
        )
        self.assertAlmostEqual(score, 80.9375)

    def test_negative_junction_drop_is_not_penalized(self):
        # With only whole/junction/drop available, weights are renormalized.
        score = structural_score(80, 60, -5)
        expected = (0.25 * 60 + 0.10 * 80 + 0.10 * 100) / 0.45
        self.assertAlmostEqual(score, expected)

    def test_rank_score_ignores_micro_differences(self):
        base_kwargs = dict(
            domain_a="IPR1",
            domain_b="IPR2",
            linker="GGSGG",
            category="heuristic",
            sequence="AAAAAGGSGGAAAAA",
            linker_length=5,
            hydrophobic_fraction=0.0,
            charged_fraction=0.0,
            proline_fraction=0.0,
            gly_ser_fraction=1.0,
            junction_plddt=80.0,
            linker_plddt=75.0,
            domain_a_plddt=85.0,
            domain_b_plddt=85.0,
            junction_drop=2.0,
        )
        a = Candidate(**base_kwargs, whole_plddt=82.000001)
        b = Candidate(**{**base_kwargs, "linker": "GSGSG", "sequence": "AAAAAGSGSGAAAAA"}, whole_plddt=82.000006)
        apply_scores([a, b])
        self.assertAlmostEqual(a.score, b.score)

    def test_junction_metrics_use_three_residues_each_side(self):
        scores = [90, 90, 90, 90, 90, 80, 80, 80, 80, 70, 70, 70, 70, 70]
        lines = []
        for serial, (residue, bfactor) in enumerate(enumerate(scores, start=1), start=1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA A{residue:4d}    "
                f"  0.000   0.000   0.000  1.00{bfactor:6.2f}           C"
            )
        pdb = "\n".join(lines)
        result = plddt_by_region(pdb, 5, 4, 5)
        self.assertAlmostEqual(result["junction_plddt"], 80.0)
        self.assertAlmostEqual(result["flank_plddt"], 80.0)
        self.assertAlmostEqual(result["junction_drop"], 0.0)


    def test_ca_lddt_is_orientation_invariant_and_penalizes_distortion(self):
        reference = torch.tensor(
            [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0], [7.6, 3.8, 0.0]],
            dtype=torch.float32,
        ).numpy()
        translated = reference + 25.0
        self.assertAlmostEqual(ca_lddt(reference, translated), 100.0)
        distorted = translated.copy()
        distorted[-1] += [8.0, 0.0, 0.0]
        self.assertLess(ca_lddt(reference, distorted), 100.0)

    def test_sequence_conditioned_lstm_shapes(self):
        model = ConditionalLSTM(
            vocab_size=23,
            num_domains=5,
            pad_idx=0,
            embedding_dim=32,
            context_embedding_dim=16,
            context_hidden_dim=8,
            domain_embedding_dim=4,
            condition_dim=24,
            hidden_dim=40,
            num_layers=2,
            dropout=0.1,
            max_target_length=30,
        )
        tokens = torch.tensor([[1, 3, 4], [1, 5, 6]])
        left_context = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]])
        right_context = torch.tensor([[8, 9, 0], [10, 11, 12]])
        left_id = torch.tensor([1, 0])
        right_id = torch.tensor([2, 3])
        lengths = torch.tensor([[100.0, 80.0], [75.0, 120.0]])
        target_lengths = torch.tensor([5, 7])
        logits, length_logits = model(
            tokens,
            left_context,
            right_context,
            left_id,
            right_id,
            lengths,
            target_lengths,
        )
        self.assertEqual(tuple(logits.shape), (2, 3, 23))
        self.assertEqual(tuple(length_logits.shape), (2, 31))
        self.assertEqual(model.lstm.input_size, 32 + 24)


if __name__ == "__main__":
    unittest.main()
