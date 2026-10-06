"""Integration tests for the per-vessel position track (GET /track)."""
from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.ais import decode_payload
from app.storage import SnapshotStore
from tests.conftest import (
    AMBIG_F1A,
    AMBIG_F1B,
    AMBIG_F2,
    INCOMPLETE_F1,
    MULTI_FRAG1,
    MULTI_FRAG2,
    MULTI_PAYLOAD,
    OTHER_POSITION,
    POS_SPLIT_F1A,
    POS_SPLIT_F1B,
    POS_SPLIT_F2,
    SINGLE_FRAGMENT,
    SINGLE_PAYLOAD,
    VDO_SENTINELS,
)

T1 = "2026-10-05T11:00:00Z"
T2 = "2026-10-05T12:00:00Z"
T3 = "2026-10-05T13:00:00Z"

MMSI = 366999999

# Two fragment-1s sharing one fragment-2 make an ambiguous pool whose
# candidates are both decodable position reports (same MMSI, different
# longitudes): [POS_SPLIT_F1A, POS_SPLIT_F2] -> -73.9835,
# [POS_SPLIT_F1B, POS_SPLIT_F2] -> -73.980087.
POS_F1A = POS_SPLIT_F1A
POS_F1B = POS_SPLIT_F1B
POS_F2 = POS_SPLIT_F2


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by a fresh snapshot database per test."""
    monkeypatch.setenv("AIS_DB_PATH", str(tmp_path / "snapshots.db"))
    monkeypatch.setattr(main, "_store", None)
    yield TestClient(main.app)
    monkeypatch.setattr(main, "_store", None)


def one(source_id: str, sentence: str) -> dict:
    return {"source_id": source_id, "sentence": sentence}


def ingest(client: TestClient, sentences: list[dict], observed_at: str):
    response = client.post(
        "/ingest", json={"observed_at": observed_at, "sentences": sentences}
    )
    return response.status_code, response.json()


def track(client: TestClient, source_id: str, mmsi: int = MMSI) -> list[dict]:
    response = client.get("/track", params={"source_id": source_id, "mmsi": mmsi})
    assert response.status_code == 200
    return response.json()


def create_review(client: TestClient, sentences: list[dict], observed_at: str):
    return client.post(
        "/reviews", json={"observed_at": observed_at, "sentences": sentences}
    )


def resolve(client: TestClient, batch_id: str, selections: list[list[int]]):
    return client.post(f"/reviews/{batch_id}/resolve", json={"selections": selections})


def ambig_position_batch(source_id: str = "rx-2") -> list[dict]:
    return [one(source_id, POS_F1A), one(source_id, POS_F1B), one(source_id, POS_F2)]


# -- plain ingest --------------------------------------------------------------


def test_ingested_positions_form_the_track(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-1", OTHER_POSITION)], T3)
    points = track(client, "rx-1")
    assert len(points) == 2
    assert points[0]["observed_at"].startswith("2026-10-05T12:00:00")
    assert points[0]["data"]["longitude"] == pytest.approx(-73.9835)
    assert points[0]["data"]["message_type"] == 1
    assert points[1]["observed_at"].startswith("2026-10-05T13:00:00")
    assert points[1]["data"]["longitude"] == pytest.approx(-73.980087)


def test_late_old_observation_extends_track_without_rolling_back_snapshot(client):
    ingest(client, [one("rx-1", OTHER_POSITION)], T3)  # newer arrives first
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)  # late old observation
    points = track(client, "rx-1")
    # the track is ordered by observation time, not arrival order
    assert [p["observed_at"][:19] for p in points] == [
        "2026-10-05T12:00:00",
        "2026-10-05T13:00:00",
    ]
    assert points[0]["data"]["longitude"] == pytest.approx(-73.9835)
    # ... but the newer snapshot is not rolled back
    (snapshot,) = client.get("/vessels", params={"source_id": "rx-1"}).json()
    assert snapshot["position"]["observed_at"].startswith("2026-10-05T13:00:00")
    assert snapshot["position"]["data"]["longitude"] == pytest.approx(-73.980087)


def test_same_observed_at_first_saved_wins_and_adds_no_point(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-1", OTHER_POSITION)], T2)  # re-submitted instant
    points = track(client, "rx-1")
    assert len(points) == 1
    assert points[0]["data"]["longitude"] == pytest.approx(-73.9835)


def test_same_batch_first_sentence_index_wins_for_the_track(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT), one("rx-1", OTHER_POSITION)], T2)
    points = track(client, "rx-1")
    assert len(points) == 1
    assert points[0]["data"]["longitude"] == pytest.approx(-73.9835)


def test_static_messages_create_no_track_points(client):
    status, body = ingest(client, [one("rx-1", MULTI_FRAG1), one("rx-1", MULTI_FRAG2)], T2)
    assert status == 200
    assert body["snapshots"][0]["static"] is not None
    assert track(client, "rx-1") == []


def test_null_coordinates_are_kept_in_the_track(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)  # real coordinates
    ingest(client, [one("rx-1", VDO_SENTINELS)], T2)  # null longitude
    points = track(client, "rx-1")
    assert len(points) == 2
    assert points[0]["data"]["longitude"] == pytest.approx(-73.9835)
    # the gap is kept as decoded, not borrowed from the neighbouring point
    assert points[1]["data"]["longitude"] is None
    assert points[1]["data"]["latitude"] == pytest.approx(40.7484)


def test_errors_incomplete_and_pending_reviews_create_no_points(client):
    bad = SINGLE_FRAGMENT.replace("15Mwqgh", "15Mwqgi")  # checksum mismatch
    ingest(client, [one("rx-1", bad)], T2)
    ingest(client, [one("rx-1", INCOMPLETE_F1)], T2)
    response = create_review(
        client, [one("rx-2", AMBIG_F1A), one("rx-2", AMBIG_F1B), one("rx-2", AMBIG_F2)], T2
    )
    assert response.status_code == 201
    assert track(client, "rx-1") == []
    assert track(client, "rx-2", 366000001) == []
    assert track(client, "rx-2", 366000002) == []


def test_ambiguous_ingest_still_tracks_the_unambiguous_decodes(client):
    status, _body = ingest(
        client,
        [
            one("rx-2", SINGLE_FRAGMENT),  # decodes unambiguously
            one("rx-2", AMBIG_F1A),
            one("rx-2", AMBIG_F1B),
            one("rx-2", AMBIG_F2),
        ],
        T2,
    )
    assert status == 409
    points = track(client, "rx-2")
    assert len(points) == 1
    assert points[0]["data"]["longitude"] == pytest.approx(-73.9835)
    # the ambiguous candidates produced no points
    assert track(client, "rx-2", 366000001) == []


def test_track_is_kept_per_source(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", OTHER_POSITION)], T2)
    assert [p["data"]["longitude"] for p in track(client, "rx-1")] == [
        pytest.approx(-73.9835)
    ]
    assert [p["data"]["longitude"] for p in track(client, "rx-2")] == [
        pytest.approx(-73.980087)
    ]


# -- review resolution ----------------------------------------------------------


def test_resolve_appends_selected_and_unambiguous_positions(client):
    batch_id = create_review(
        client,
        ambig_position_batch()  # ambiguous pool of position candidates
        + [one("rx-9", OTHER_POSITION)],  # unambiguous decode, other source
        T1,
    ).json()["batch_id"]
    # nothing is tracked while the batch is pending
    assert track(client, "rx-2") == []
    assert track(client, "rx-9") == []

    response = resolve(client, batch_id, [[0, 2]])
    assert response.status_code == 200

    # the selected candidate became a track point; the loser did not
    points = track(client, "rx-2")
    assert len(points) == 1
    assert points[0]["observed_at"].startswith("2026-10-05T11:00:00")
    assert points[0]["data"]["longitude"] == pytest.approx(-73.9835)
    # the batch's unambiguous decode was tracked as well
    points = track(client, "rx-9")
    assert len(points) == 1
    assert points[0]["data"]["longitude"] == pytest.approx(-73.980087)


def test_late_resolution_extends_track_without_rolling_back_snapshot(client):
    batch_id = create_review(client, ambig_position_batch(), T1).json()["batch_id"]
    # a newer observation for the same vessel arrives before the resolution
    ingest(client, [one("rx-2", OTHER_POSITION)], T2)

    response = resolve(client, batch_id, [[0, 2]])
    assert response.status_code == 200

    # the track gained the old point in its time position ...
    points = track(client, "rx-2")
    assert [p["observed_at"][:19] for p in points] == [
        "2026-10-05T11:00:00",
        "2026-10-05T12:00:00",
    ]
    assert points[0]["data"]["longitude"] == pytest.approx(-73.9835)
    assert points[1]["data"]["longitude"] == pytest.approx(-73.980087)
    # ... but the newer snapshot was not rolled back
    (snapshot,) = client.get("/vessels", params={"source_id": "rx-2"}).json()
    assert snapshot["position"]["observed_at"].startswith("2026-10-05T12:00:00")
    assert snapshot["position"]["data"]["longitude"] == pytest.approx(-73.980087)


def test_unselected_candidate_and_second_resolution_add_no_points(client):
    batch_id = create_review(client, ambig_position_batch(), T1).json()["batch_id"]
    assert resolve(client, batch_id, [[0, 2]]).status_code == 200
    assert resolve(client, batch_id, [[1, 2]]).status_code == 409  # already resolved
    points = track(client, "rx-2")
    assert len(points) == 1
    assert points[0]["data"]["longitude"] == pytest.approx(-73.9835)


def test_resolve_same_observed_at_keeps_first_track_point(client):
    # an equal-time point already stored wins over the resolved candidate
    ingest(client, [one("rx-2", OTHER_POSITION)], T1)
    batch_id = create_review(client, ambig_position_batch(), T1).json()["batch_id"]
    assert resolve(client, batch_id, [[0, 2]]).status_code == 200
    points = track(client, "rx-2")
    assert len(points) == 1
    assert points[0]["data"]["longitude"] == pytest.approx(-73.980087)


# -- querying -------------------------------------------------------------------


def test_track_query_requires_source_and_mmsi(client):
    assert client.get("/track").status_code == 422
    assert client.get("/track", params={"source_id": "rx-1"}).status_code == 422
    assert client.get("/track", params={"mmsi": MMSI}).status_code == 422


def test_unknown_vessel_has_an_empty_track(client):
    assert track(client, "rx-nobody") == []


def test_track_survives_restart(client, monkeypatch):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", OTHER_POSITION)], T2)
    # simulate a container restart: drop the cached store, reopen the same file
    monkeypatch.setattr(main, "_store", None)
    assert len(track(client, "rx-1")) == 2


# -- store level -----------------------------------------------------------------


def test_store_track_dedup_and_order(tmp_path):
    store = SnapshotStore(str(tmp_path / "store.db"))
    position = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    other = dict(position, longitude=-70.0)
    store.upsert_if_newer("rx-1", MMSI, "position", "2026-10-05T12:00:00.000000Z", position)
    # equal instant: first saved wins, no extra point
    store.upsert_if_newer("rx-1", MMSI, "position", "2026-10-05T12:00:00.000000Z", other)
    # older instant: still tracked (late observation)
    store.upsert_if_newer("rx-1", MMSI, "position", "2026-10-05T11:00:00.000000Z", other)
    points = store.get_track("rx-1", MMSI)
    assert [p.observed_at.isoformat()[:19] for p in points] == [
        "2026-10-05T11:00:00",
        "2026-10-05T12:00:00",
    ]
    assert points[0].data.longitude == pytest.approx(-70.0)
    assert points[1].data.longitude == pytest.approx(-73.9835)
    store.close()


def test_existing_snapshots_seed_one_starting_point_per_vessel(tmp_path):
    """A database written before tracks existed: the stored position
    snapshots become the first track points; no earlier history is invented."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE vessel_state (
            source_id   TEXT NOT NULL,
            mmsi        INTEGER NOT NULL,
            kind        TEXT NOT NULL CHECK (kind IN ('position', 'static')),
            observed_at TEXT NOT NULL,
            data        TEXT NOT NULL,
            PRIMARY KEY (source_id, mmsi, kind)
        );
        """
    )
    position = decode_payload(SINGLE_PAYLOAD, 0).model_dump(mode="json")
    static = decode_payload(MULTI_PAYLOAD, 2).model_dump(mode="json")
    conn.execute(
        "INSERT INTO vessel_state VALUES (?, ?, 'position', ?, ?)",
        ("rx-1", MMSI, "2026-10-05T10:00:00.000000Z", json.dumps(position)),
    )
    conn.execute(
        "INSERT INTO vessel_state VALUES (?, ?, 'static', ?, ?)",
        ("rx-1", MMSI, "2026-10-05T10:00:00.000000Z", json.dumps(static)),
    )
    conn.commit()
    conn.close()

    store = SnapshotStore(path)
    points = store.get_track("rx-1", MMSI)
    assert len(points) == 1  # one starting point, no invented earlier history
    assert points[0].observed_at.isoformat().startswith("2026-10-05T10:00:00")
    assert points[0].data.longitude == pytest.approx(-73.9835)
    # the snapshot side is untouched and new observations still append
    assert store.get_snapshot("rx-1", MMSI).static is not None
    store.upsert_if_newer(
        "rx-1", MMSI, "position", "2026-10-05T12:00:00.000000Z", position
    )
    assert len(store.get_track("rx-1", MMSI)) == 2
    store.close()
