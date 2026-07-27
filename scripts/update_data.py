
from pathlib import Path
from datetime import datetime
import urllib.request, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"; RAW.mkdir(parents=True,exist_ok=True)
sources=[
 ("tml","https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"),
 ("sackmann","https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv")
]
for year in range(2020,datetime.utcnow().year+1):
    target=RAW/f"{year}.csv"; success=False
    for name,template in sources:
        try:
            urllib.request.urlretrieve(template.format(year=year),target)
            print(f"{year}: downloaded from {name}"); success=True; break
        except Exception as exc:
            print(f"{year}: {name} failed: {exc}")
    if not success and not target.exists():
        print(f"{year}: no source available")
subprocess.check_call([sys.executable,str(ROOT/"scripts"/"train_model.py")])
