from maritime_nsrl.confirmatory import STAGES, method_definitions


def test_confirmatory_methods_isolate_tag_penalty_arm():
    methods=method_definitions()
    assert set(methods)=={"neural","temporal_state_only","softened_guidance","softened_guidance_no_tag_penalty"}
    a,b=methods["softened_guidance"],methods["softened_guidance_no_tag_penalty"]
    assert {k:v for k,v in a.items() if k!="tag_penalty"} == {k:v for k,v in b.items() if k!="tag_penalty"}
    assert a["tag_penalty"] == 1. and b["tag_penalty"] == 0.


def test_curriculum_starts_patrol_and_uses_shared_policy_stage_later():
    assert STAGES["02-1v1-patrol"]["opponent"] == "patrol"
    assert STAGES["04-3v3-static"]["team_size"] == 3
