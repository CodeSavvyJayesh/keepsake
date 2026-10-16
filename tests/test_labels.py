"""Tests for matching selected copies back to originals. Run: ks test"""
from pathlib import Path

from engine.labels import match_selected


def make(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_matches_by_name_ignoring_case(tmp_path):
    raw = [make(tmp_path / "raw" / "DSC_0001.JPG"), make(tmp_path / "raw" / "DSC_0002.JPG")]
    sel = [make(tmp_path / "sel" / "Haldi" / "dsc_0002.jpg")]
    r = match_selected(raw, sel)
    assert r.matches == {raw[1]: sel[0]}
    assert r.how["file name"] == 1 and not r.unmatched


def test_matches_edited_export_by_name_without_extension(tmp_path):
    raw = [make(tmp_path / "raw" / "IMG_1043.CR3")]
    sel = [make(tmp_path / "sel" / "IMG_1043.jpg")]
    assert match_selected(raw, sel).matches == {raw[0]: sel[0]}


def test_shared_name_resolved_by_size(tmp_path):
    a = make(tmp_path / "raw" / "camA" / "DSC_8858.JPG", b"aaaa")
    b = make(tmp_path / "raw" / "camB" / "DSC_8858.JPG", b"bbbbbbbb")
    sel = [make(tmp_path / "sel" / "DSC_8858.JPG", b"bbbbbbbb")]
    r = match_selected([a, b], sel)
    assert r.matches == {b: sel[0]} and r.how["file name + size"] == 1


def test_shared_name_resolved_by_camera_and_time(tmp_path):
    a = make(tmp_path / "raw" / "camA" / "DSC_8858.JPG", b"aaaa")
    b = make(tmp_path / "raw" / "camB" / "DSC_8858.JPG", b"bbbb")
    s = make(tmp_path / "sel" / "DSC_8858.JPG", b"edited")      # edited: size differs
    meta = {a: ("NIKON Z 9", "2026:08:27 18:01:02", "10"),
            b: ("NIKON Z 8", "2026:08:27 19:15:40", "55"),
            s: ("NIKON Z 8", "2026:08:27 19:15:40", "55")}
    r = match_selected([a, b], [s], meta_fn=lambda files: {f: meta[f] for f in files})
    assert r.matches == {b: s} and r.how["file name + camera + time"] == 1


def test_unresolvable_and_missing_are_reported_not_guessed(tmp_path):
    a = make(tmp_path / "raw" / "camA" / "DSC_1.JPG", b"same")
    b = make(tmp_path / "raw" / "camB" / "DSC_1.JPG", b"same")
    s1 = make(tmp_path / "sel" / "DSC_1.JPG", b"same")
    s2 = make(tmp_path / "sel" / "NOT_IN_RAW.JPG")
    r = match_selected([a, b], [s1, s2])
    assert r.matches == {} and sorted(r.unmatched) == sorted([s1, s2])


def test_second_copy_of_same_original_is_a_repeat(tmp_path):
    raw = [make(tmp_path / "raw" / "DSC_5.JPG")]
    sel = [make(tmp_path / "sel" / "Day1" / "DSC_5.JPG"), make(tmp_path / "sel" / "Day2" / "DSC_5.JPG")]
    r = match_selected(raw, sel)
    assert len(r.matches) == 1 and len(r.repeats) == 1


def test_typo_in_extension_matched_by_size(tmp_path):
    a = make(tmp_path / "raw" / "camA" / "DSC_0342.JPG", b"aaaa")
    b = make(tmp_path / "raw" / "camB" / "DSC_0342.JPG", b"bbbbbbbb")
    typo = make(tmp_path / "sel" / "DSC_0342.JPGG.JPG", b"bbbbbbbb")
    r = match_selected([a, b], [typo])
    assert r.matches == {b: typo} and r.how["file name with typo + size"] == 1


def test_typo_with_different_size_is_not_guessed(tmp_path):
    a = make(tmp_path / "raw" / "DSC_0509.JPG", b"aaaa")
    typo = make(tmp_path / "sel" / "DSC_0509.JPGG.JPG", b"edited!")
    assert match_selected([a], [typo]).unmatched == [typo]
