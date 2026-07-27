
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from atp_model.model_service import load_state,load_bundle,predict_match

def test_prediction_smoke():
    df=load_state(); bundle=load_bundle()
    names=list(df.player.unique())
    a,b=names[0],names[1]
    r=predict_match(df,bundle,a,b,"Hard",100,120,1.8,2.05)
    assert 0 < r["probability_a"] < 1
    assert r["fair_odds_a"] > 1
