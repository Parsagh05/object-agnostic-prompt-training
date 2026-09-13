"""The balanced/full split protocols and the manifest provenance guard."""

from dataclasses import replace
from pathlib import Path
import unittest

from object_agnostic_prompt_attack.config import (
    LABEL_BALANCE_POLICIES,
    DataConfig,
    ExperimentConfig,
)
from object_agnostic_prompt_attack.data import (
    PromptTrainingSample,
    automatic_attack_train_split,
    automatic_evaluation_split,
    automatic_protocol_split,
    load_attack_train_manifest,
)
from object_agnostic_prompt_attack.training import dataset_output_directory


def make_samples(counts: dict[str, tuple[int, int]], dataset: str = "mvtec"):
    """One sample per image for categories described as (normal, abnormal)."""

    samples = []
    for category, (normal, abnormal) in counts.items():
        for label, total in ((0, normal), (1, abnormal)):
            for index in range(total):
                name = "good" if label == 0 else "defect"
                samples.append(
                    PromptTrainingSample(
                        protocol_id=f"test/{category}/{name}/{index:04d}",
                        dataset=dataset,
                        category=category,
                        defect_type=name,
                        image_path=Path(f"/root/{category}/{name}/{index:04d}.png"),
                        mask_path=None if label == 0 else Path("/root/mask.png"),
                        label=label,
                    )
                )
    return samples


def count_by_label(selected):
    return {
        label: sum(sample.label == label for sample in selected) for label in (0, 1)
    }


# Real MVTec AD test-split sizes, so the protocol totals are checked against the
# dataset the pipeline actually runs on rather than a toy fixture.
MVTEC_TEST_COUNTS = {
    "bottle": (20, 63), "cable": (58, 92), "capsule": (23, 109),
    "carpet": (28, 89), "grid": (21, 57), "hazelnut": (40, 70),
    "leather": (32, 92), "metal_nut": (22, 93), "pill": (26, 141),
    "screw": (41, 119), "tile": (33, 84), "toothbrush": (12, 30),
    "transistor": (60, 40), "wood": (19, 60), "zipper": (32, 119),
}


class BalancedProtocolTests(unittest.TestCase):
    def test_reproduces_the_shipped_checkpoint_cohort(self):
        """448 samples, 224 per label -- the cohort in results/prompts/mvtec."""

        selected = automatic_attack_train_split(
            make_samples(MVTEC_TEST_COUNTS), seed=111, evaluation_fraction=0.5
        )
        self.assertEqual(len(selected), 448)
        self.assertEqual(count_by_label(selected), {0: 224, 1: 224})

    def test_default_protocol_is_balanced(self):
        samples = make_samples(MVTEC_TEST_COUNTS)
        default = automatic_attack_train_split(
            samples, seed=111, evaluation_fraction=0.5
        )
        explicit = automatic_attack_train_split(
            samples, seed=111, evaluation_fraction=0.5, protocol="balanced"
        )
        self.assertEqual(
            [sample.protocol_id for sample in default],
            [sample.protocol_id for sample in explicit],
        )

    def test_discards_the_surplus_of_a_lopsided_category(self):
        selected = automatic_attack_train_split(
            make_samples({"bottle": (20, 63)}), seed=111, evaluation_fraction=0.5
        )
        self.assertEqual(count_by_label(selected), {0: 10, 1: 10})


class FullProtocolTests(unittest.TestCase):
    def test_keeps_every_image_and_holds_out_half_of_each_label(self):
        selected = automatic_attack_train_split(
            make_samples(MVTEC_TEST_COUNTS),
            seed=111,
            evaluation_fraction=0.5,
            protocol="full",
        )
        self.assertEqual(len(selected), 864)

    def test_evaluation_count_follows_the_kept_images_not_the_minimum(self):
        """The regression the `basis` line exists to prevent.

        For 20 normal / 63 abnormal the held-out counts are 10 and 32, leaving
        41 for training. Deriving both from min(20, 63) would hold out 10 of
        each and hand 53 abnormal images to prompt training -- 22 of them
        reserved by the attack pipeline for evaluation.
        """

        selected = automatic_attack_train_split(
            make_samples({"bottle": (20, 63)}),
            seed=111,
            evaluation_fraction=0.5,
            protocol="full",
        )
        self.assertEqual(count_by_label(selected), {0: 10, 1: 31})
        self.assertEqual(len(selected), 41)
        self.assertNotEqual(len(selected), 63)

    def test_never_selects_more_than_the_balanced_complement_allows(self):
        """Every category keeps at least one image on each side of the split."""

        for category, counts in MVTEC_TEST_COUNTS.items():
            selected = automatic_attack_train_split(
                make_samples({category: counts}),
                seed=111,
                evaluation_fraction=0.5,
                protocol="full",
            )
            per_label = count_by_label(selected)
            for label, available in enumerate(counts):
                self.assertGreaterEqual(per_label[label], 1, category)
                self.assertLess(per_label[label], available, category)

    def test_rejects_an_unknown_protocol(self):
        with self.assertRaises(ValueError):
            automatic_attack_train_split(
                make_samples({"bottle": (20, 63)}),
                seed=111,
                evaluation_fraction=0.5,
                protocol="stratified",
            )


class AttackTrainFractionTests(unittest.TestCase):
    """Shrinks the training cohort without moving the split boundary."""

    def cohort(self, fraction, protocol="balanced"):
        return automatic_attack_train_split(
            make_samples(MVTEC_TEST_COUNTS),
            seed=111,
            evaluation_fraction=0.5,
            protocol=protocol,
            attack_train_fraction=fraction,
        )

    def test_full_fraction_is_the_default_and_changes_nothing(self):
        self.assertEqual(len(self.cohort(1.0)), 448)
        default = automatic_attack_train_split(
            make_samples(MVTEC_TEST_COUNTS), seed=111, evaluation_fraction=0.5
        )
        self.assertEqual(
            [sample.protocol_id for sample in default],
            [sample.protocol_id for sample in self.cohort(1.0)],
        )

    def test_reproduces_the_attack_pipelines_cohort_sizes(self):
        """Verified image-for-image against select_attack_train_fraction."""

        for fraction, expected in ((0.05, 34), (0.10, 58), (0.25, 124), (0.50, 232)):
            self.assertEqual(len(self.cohort(fraction)), expected, fraction)
        for fraction, expected in ((0.05, 56), (0.10, 99), (0.25, 228), (0.50, 441)):
            self.assertEqual(len(self.cohort(fraction, "full")), expected, fraction)

    def test_smaller_fractions_nest_inside_larger_ones(self):
        """Rank order is stable, so a cheaper run is a prefix of a richer one."""

        previous = None
        for fraction in (0.05, 0.10, 0.25, 0.50, 1.0):
            current = {sample.protocol_id for sample in self.cohort(fraction)}
            if previous is not None:
                self.assertTrue(previous <= current, fraction)
            previous = current

    def test_keeps_at_least_one_image_per_stratum(self):
        cohort = self.cohort(0.01)
        strata = {(sample.category, sample.label) for sample in cohort}
        self.assertEqual(len(strata), len(MVTEC_TEST_COUNTS) * 2)

    def test_rejects_a_fraction_outside_its_range(self):
        for fraction in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                self.cohort(fraction)


class EvaluationSplitTests(unittest.TestCase):
    """The held-out half must be exactly the complement of the trained half."""

    def both(self, protocol="balanced", fraction=1.0):
        return automatic_protocol_split(
            make_samples(MVTEC_TEST_COUNTS),
            seed=111,
            evaluation_fraction=0.5,
            protocol=protocol,
            attack_train_fraction=fraction,
        )

    def test_halves_never_overlap(self):
        for protocol in ("balanced", "full"):
            train, evaluation = self.both(protocol)
            ids_train = {s.protocol_id for s in train}
            ids_eval = {s.protocol_id for s in evaluation}
            self.assertEqual(ids_train & ids_eval, set(), protocol)

    def test_halves_partition_the_kept_images(self):
        train, evaluation = self.both("full")
        # "full" keeps every test image, so the two halves must cover all 1725.
        self.assertEqual(len(train) + len(evaluation), 1725)
        self.assertEqual(len(train), 864)
        self.assertEqual(len(evaluation), 861)

    def test_balanced_halves_cover_only_the_balanced_cohort(self):
        train, evaluation = self.both("balanced")
        self.assertEqual(len(train), 448)
        self.assertEqual(len(train) + len(evaluation), 894)

    def test_evaluation_half_is_never_subsetted_by_the_train_fraction(self):
        """A cheaper training cohort must still be scored on the same images."""

        _, full_eval = self.both("full", fraction=1.0)
        for fraction in (0.5, 0.25, 0.05):
            train, evaluation = self.both("full", fraction=fraction)
            self.assertEqual(
                [s.protocol_id for s in evaluation],
                [s.protocol_id for s in full_eval],
                fraction,
            )
            self.assertLess(len(train), 864)

    def test_evaluation_half_carries_its_partition_label(self):
        _, evaluation = self.both("full")
        self.assertEqual({s.partition for s in evaluation}, {"evaluation"})

    def test_both_labels_present_in_the_evaluation_half(self):
        for protocol in ("balanced", "full"):
            _, evaluation = self.both(protocol)
            self.assertEqual(count_by_label(evaluation).keys(), {0, 1})
            self.assertTrue(all(v > 0 for v in count_by_label(evaluation).values()))

    def test_convenience_wrapper_matches(self):
        _, evaluation = self.both("full")
        direct = automatic_evaluation_split(
            make_samples(MVTEC_TEST_COUNTS),
            seed=111, evaluation_fraction=0.5, protocol="full",
        )
        self.assertEqual([s.protocol_id for s in direct],
                         [s.protocol_id for s in evaluation])

    def test_attack_train_wrapper_is_unchanged(self):
        train, _ = self.both("balanced")
        legacy = automatic_attack_train_split(
            make_samples(MVTEC_TEST_COUNTS), seed=111, evaluation_fraction=0.5
        )
        self.assertEqual([s.protocol_id for s in legacy],
                         [s.protocol_id for s in train])


class ConfigTests(unittest.TestCase):
    def test_default_is_balanced_with_its_policy_string(self):
        config = DataConfig()
        self.assertEqual(config.split_protocol, "balanced")
        self.assertEqual(
            config.label_balance_policy, "per_dataset_category_equal_labels_v1"
        )

    def test_full_reports_the_attack_pipelines_full_policy(self):
        config = DataConfig(split_protocol="full")
        self.assertEqual(
            config.label_balance_policy, "per_dataset_category_all_images_v1"
        )

    def test_rejects_an_unknown_protocol(self):
        with self.assertRaises(ValueError):
            DataConfig(split_protocol="stratified")

    def test_rejects_a_fraction_outside_its_range(self):
        for fraction in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                DataConfig(attack_train_fraction=fraction)

    def test_cohort_directory_matches_the_attack_setup_id_convention(self):
        cases = {
            ("balanced", 1.0): "balanced",
            ("full", 1.0): "full",
            ("balanced", 0.25): "balanced_train25",
            ("full", 0.2): "full_train20",
            ("full", 0.125): "full_train12p5",
            ("balanced", 0.05): "balanced_train5",
        }
        for (protocol, fraction), expected in cases.items():
            config = DataConfig(
                split_protocol=protocol, attack_train_fraction=fraction
            )
            self.assertEqual(config.cohort_directory, expected)

    def test_output_directory_carries_the_cohort_segment(self):
        config = ExperimentConfig.from_mapping(
            {"artifacts": {"output_root": "artifacts/prompts"}}
        )
        for protocol in ("balanced", "full"):
            scoped = replace(config, data=replace(config.data, split_protocol=protocol))
            directory = dataset_output_directory(scoped, "mvtec")
            self.assertEqual(directory.parts[-2:], (protocol, "mvtec"))

        scoped = replace(
            config,
            data=replace(
                config.data, split_protocol="full", attack_train_fraction=0.25
            ),
        )
        self.assertEqual(
            dataset_output_directory(scoped, "visa").parts[-2:],
            ("full_train25", "visa"),
        )


class ManifestProvenanceTests(unittest.TestCase):
    """A manifest carries the settings it was built under; they must agree."""

    header = (
        "protocol_id,dataset,category,label,partition,"
        "split_seed,evaluation_fraction,label_balance_policy\n"
    )

    def write_manifest(self, policy, seed=111, fraction=0.5):
        path = Path(self.directory.name) / "attack_train_indices.csv"
        rows = "".join(
            f"test/bottle/{name}/{index:04d},mvtec,bottle,{label},attack_train,"
            f"{seed},{fraction},{policy}\n"
            for label, name in ((0, "good"), (1, "defect"))
            for index in range(2)
        )
        path.write_text(self.header + rows, encoding="utf-8")
        return path

    def setUp(self):
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.discovered = make_samples({"bottle": (2, 2)})

    def load(self, path, **expected):
        return load_attack_train_manifest(
            path,
            dataset="mvtec",
            root=Path("/root"),
            discovered=self.discovered,
            **expected,
        )

    def test_accepts_a_manifest_whose_policy_matches(self):
        path = self.write_manifest(LABEL_BALANCE_POLICIES["balanced"])
        selected = self.load(
            path,
            expected_policy=LABEL_BALANCE_POLICIES["balanced"],
            expected_seed=111,
            expected_evaluation_fraction=0.5,
        )
        self.assertEqual(len(selected), 4)

    def test_rejects_a_full_manifest_under_the_balanced_protocol(self):
        path = self.write_manifest(LABEL_BALANCE_POLICIES["full"])
        with self.assertRaises(ValueError) as caught:
            self.load(path, expected_policy=LABEL_BALANCE_POLICIES["balanced"])
        self.assertIn("split_protocol", str(caught.exception))

    def test_rejects_a_mismatched_seed(self):
        path = self.write_manifest(LABEL_BALANCE_POLICIES["balanced"], seed=42)
        with self.assertRaises(ValueError) as caught:
            self.load(path, expected_seed=111)
        self.assertIn("split_seed", str(caught.exception))

    def test_rejects_a_mismatched_evaluation_fraction(self):
        path = self.write_manifest(LABEL_BALANCE_POLICIES["balanced"], fraction=0.3)
        with self.assertRaises(ValueError) as caught:
            self.load(path, expected_evaluation_fraction=0.5)
        self.assertIn("evaluation_fraction", str(caught.exception))

    def test_tolerates_a_manifest_without_provenance_columns(self):
        path = Path(self.directory.name) / "legacy.csv"
        path.write_text(
            "protocol_id,dataset,category,label,partition\n"
            + "".join(
                f"test/bottle/{name}/{index:04d},mvtec,bottle,{label},attack_train\n"
                for label, name in ((0, "good"), (1, "defect"))
                for index in range(2)
            ),
            encoding="utf-8",
        )
        selected = self.load(
            path,
            expected_policy=LABEL_BALANCE_POLICIES["balanced"],
            expected_seed=111,
            expected_evaluation_fraction=0.5,
        )
        self.assertEqual(len(selected), 4)


class ManifestFractionTests(unittest.TestCase):
    """The fraction is applied from the manifest's own rank columns."""

    header = (
        "protocol_id,dataset,category,label,partition,"
        "attack_train_rank,attack_train_stratum_size\n"
    )

    def setUp(self):
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.discovered = make_samples({"bottle": (4, 4)})

    def write(self, path_name, header, ranked=True):
        path = Path(self.directory.name) / path_name
        rows = ""
        for label, name in ((0, "good"), (1, "defect")):
            for rank in range(1, 5):
                row = f"test/bottle/{name}/{rank - 1:04d},mvtec,bottle,{label},attack_train"
                if ranked:
                    row += f",{rank},4"
                rows += row + "\n"
        path.write_text(header + rows, encoding="utf-8")
        return path

    def load(self, path, fraction):
        return load_attack_train_manifest(
            path,
            dataset="mvtec",
            root=Path("/root"),
            discovered=self.discovered,
            attack_train_fraction=fraction,
        )

    def test_keeps_the_rank_prefix_of_each_stratum(self):
        path = self.write("ranked.csv", self.header)
        self.assertEqual(len(self.load(path, 1.0)), 8)
        self.assertEqual(len(self.load(path, 0.5)), 4)
        self.assertEqual(len(self.load(path, 0.25)), 2)
        # ceil() keeps the cohort from rounding away to nothing.
        self.assertEqual(len(self.load(path, 0.01)), 2)

    def test_selects_the_lowest_ranks(self):
        path = self.write("ranked.csv", self.header)
        selected = {sample.protocol_id for sample in self.load(path, 0.5)}
        self.assertEqual(
            selected,
            {"test/bottle/good/0000", "test/bottle/good/0001",
             "test/bottle/defect/0000", "test/bottle/defect/0001"},
        )

    def test_full_fraction_does_not_need_the_rank_columns(self):
        path = self.write(
            "plain.csv", "protocol_id,dataset,category,label,partition\n", ranked=False
        )
        self.assertEqual(len(self.load(path, 1.0)), 8)

    def test_refuses_a_partial_fraction_without_the_rank_columns(self):
        path = self.write(
            "plain.csv", "protocol_id,dataset,category,label,partition\n", ranked=False
        )
        with self.assertRaises(ValueError) as caught:
            self.load(path, 0.5)
        self.assertIn("attack_train_rank", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
