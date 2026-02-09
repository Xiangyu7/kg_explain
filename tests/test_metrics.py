"""Unit tests for kg_explain.evaluation.metrics module.

Tests cover:
    - hit_at_k: basic, miss, edge cases
    - reciprocal_rank: first, second, not found
    - precision_at_k: all positive, none, partial
    - average_precision: perfect, partial, empty
    - ndcg_at_k: perfect, partial, empty
    - auroc: perfect, random, edge cases
    - Input validation: type checking, k validation
    - Boundary cases: empty lists, empty positives, k > len
"""
import pytest
import math

from kg_explain.evaluation.metrics import (
    hit_at_k,
    reciprocal_rank,
    precision_at_k,
    average_precision,
    ndcg_at_k,
    auroc,
    _validate_inputs,
)


# ===== _validate_inputs =====

class TestValidateInputs:
    def test_valid(self):
        result = _validate_inputs(["a", "b"], {"a"}, k=5)
        assert result == {"a"}

    def test_list_positives_auto_convert(self):
        result = _validate_inputs(["a"], ["a", "b"])
        assert result == {"a", "b"}

    def test_invalid_ranked_type(self):
        with pytest.raises(TypeError, match="ranked"):
            _validate_inputs("not_a_list", {"a"})

    def test_invalid_positives_type(self):
        with pytest.raises(TypeError, match="positives"):
            _validate_inputs(["a"], "not_a_set")

    def test_invalid_k_zero(self):
        with pytest.raises(ValueError, match="k"):
            _validate_inputs(["a"], {"a"}, k=0)

    def test_invalid_k_negative(self):
        with pytest.raises(ValueError, match="k"):
            _validate_inputs(["a"], {"a"}, k=-1)


# ===== hit_at_k =====

class TestHitAtK:
    def test_hit(self):
        assert hit_at_k(["a", "b", "c"], {"a"}, 3) == 1.0

    def test_miss(self):
        assert hit_at_k(["a", "b", "c"], {"d"}, 3) == 0.0

    def test_k_1_hit(self):
        assert hit_at_k(["pos", "neg"], {"pos"}, 1) == 1.0

    def test_k_1_miss(self):
        assert hit_at_k(["neg", "pos"], {"pos"}, 1) == 0.0

    def test_empty_ranked(self):
        assert hit_at_k([], {"a"}, 5) == 0.0

    def test_empty_positives(self):
        assert hit_at_k(["a", "b"], set(), 5) == 0.0

    def test_k_larger_than_list(self):
        assert hit_at_k(["a"], {"a"}, 100) == 1.0


# ===== reciprocal_rank =====

class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["b", "a", "c"], {"a"}) == 0.5

    def test_third_position(self):
        assert reciprocal_rank(["b", "c", "a"], {"a"}) == pytest.approx(1/3)

    def test_not_found(self):
        assert reciprocal_rank(["a", "b"], {"x"}) == 0.0

    def test_empty_ranked(self):
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_empty_positives(self):
        assert reciprocal_rank(["a", "b"], set()) == 0.0

    def test_multiple_positives_first(self):
        """Should return rank of FIRST positive found."""
        assert reciprocal_rank(["neg", "pos1", "pos2"], {"pos1", "pos2"}) == 0.5


# ===== precision_at_k =====

class TestPrecisionAtK:
    def test_all_positive(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_none_positive(self):
        assert precision_at_k(["a", "b"], {"c"}, 2) == 0.0

    def test_half_positive(self):
        assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5

    def test_k_1(self):
        assert precision_at_k(["pos", "neg"], {"pos"}, 1) == 1.0

    def test_empty_ranked(self):
        assert precision_at_k([], {"a"}, 5) == 0.0


# ===== average_precision =====

class TestAveragePrecision:
    def test_perfect(self):
        """All positives at the top."""
        assert average_precision(["a", "b", "c"], {"a", "b"}) == 1.0

    def test_one_positive_at_top(self):
        assert average_precision(["pos", "neg1", "neg2"], {"pos"}) == 1.0

    def test_one_positive_at_bottom(self):
        assert average_precision(["neg1", "neg2", "pos"], {"pos"}) == pytest.approx(1/3)

    def test_empty_ranked(self):
        assert average_precision([], {"a"}) == 0.0

    def test_empty_positives(self):
        assert average_precision(["a", "b"], set()) == 0.0

    def test_two_positives_spread(self):
        """pos1 at rank 1, pos2 at rank 3."""
        ranked = ["pos1", "neg", "pos2"]
        positives = {"pos1", "pos2"}
        # P@1 for pos1 = 1/1, P@3 for pos2 = 2/3
        # AP = (1 + 2/3) / 2 = 5/6
        assert average_precision(ranked, positives) == pytest.approx(5/6)


# ===== ndcg_at_k =====

class TestNdcgAtK:
    def test_perfect_ranking(self):
        """All positives in top-K."""
        assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_worst_ranking(self):
        """No positives in top-K."""
        assert ndcg_at_k(["neg1", "neg2"], {"pos"}, 2) == 0.0

    def test_empty_ranked(self):
        assert ndcg_at_k([], {"a"}, 5) == 0.0

    def test_empty_positives(self):
        assert ndcg_at_k(["a"], set(), 5) == 0.0

    def test_single_positive_at_1(self):
        assert ndcg_at_k(["pos", "neg"], {"pos"}, 2) == 1.0

    def test_single_positive_at_2(self):
        ranked = ["neg", "pos"]
        ndcg = ndcg_at_k(ranked, {"pos"}, 2)
        # DCG = 1/log2(3) = 0.6309
        # IDCG = 1/log2(2) = 1.0
        assert ndcg == pytest.approx(1/math.log2(3), abs=0.001)


# ===== auroc =====

class TestAuroc:
    def test_perfect_ranking(self):
        """All positives before all negatives."""
        assert auroc(["pos1", "pos2", "neg1", "neg2"], {"pos1", "pos2"}) == 1.0

    def test_worst_ranking(self):
        """All negatives before all positives."""
        assert auroc(["neg1", "neg2", "pos1", "pos2"], {"pos1", "pos2"}) == 0.0

    def test_random_ranking(self):
        """Interleaved: P(positive ranks above negative) = 0.5."""
        auc = auroc(["pos", "neg", "pos", "neg"], {"pos"})
        # Only one unique positive, so just the first 'pos'
        # Wait, there are duplicates. Use unique items.
        auc = auroc(["p1", "n1", "p2", "n2"], {"p1", "p2"})
        # p1: 2 neg below, p2: 1 neg below. total = 3. n_pos*n_neg = 4.
        assert auc == pytest.approx(3/4)

    def test_empty_ranked(self):
        assert auroc([], {"a"}) == 0.0

    def test_empty_positives(self):
        assert auroc(["a", "b"], set()) == 0.0

    def test_all_positive(self):
        """Edge case: no negatives."""
        assert auroc(["a", "b"], {"a", "b"}) == 0.0

    def test_all_negative(self):
        """Edge case: no positives in ranked."""
        assert auroc(["a", "b"], {"c"}) == 0.0

    def test_single_positive_at_top(self):
        assert auroc(["pos", "n1", "n2", "n3"], {"pos"}) == 1.0

    def test_single_positive_at_bottom(self):
        assert auroc(["n1", "n2", "n3", "pos"], {"pos"}) == 0.0
