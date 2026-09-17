import pytest
from scripts.review_moran_road_candidates import distance_to_line_m, review


def test_distance_uses_segment_interior_and_clamps_endpoints():
    assert distance_to_line_m(0,0,[[-.001,.001],[.001,.001]])==pytest.approx(111.195)
    assert distance_to_line_m(0,0,[[.001,0],[.002,0]])==pytest.approx(111.195)
    assert distance_to_line_m(0,0,[[0,0],[0,0]])==0


def test_conflicting_nearest_cameras_stay_unmapped():
    bundle={'features':[{'id':str(i),'geometry':{'coordinates':[0,0]},
                         'properties':{'measurement':{'verdict':{'pump':v}}}}
                        for i,v in enumerate(['PASS','FAIL'])]}
    roads={'features':[{'geometry':{'type':'LineString','coordinates':[[0,0],[.001,0]]},'properties':{'id':'road'}}]}
    result=review(bundle,roads)
    assert result['nearest_candidate_overlaps'][0]['mixed_verdicts']
    assert result['confirmed_mapping_count']==0 and not result['routing_ready']
    assert all(r['edge_id'] is None for r in result['camera_candidates'])
    empty=review(bundle,{'features':[]})
    assert all(not r['nearest_within_review_distance'] for r in empty['camera_candidates'])
