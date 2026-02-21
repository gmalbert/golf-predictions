"""One-shot script: add yardage field to every entry in course_metadata.json."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "data_files" / "course_metadata.json"
data = json.loads(p.read_text())

YARDAGE = {
    "masters tournament": 7510, "the masters": 7510, "masters": 7510,
    "u.s. open": 7300,
    "the open championship": 7180, "open championship": 7180, "the open": 7180,
    "pga championship": 7400,
    "the players championship": 7215, "players championship": 7215,
    "tour championship": 7346, "the tour championship": 7346,
    "bmw championship": 7500,
    "fedex st. jude championship": 7239, "fedex st. jude classic": 7239,
    "wgc-fedex st. jude invitational": 7239,
    "northern trust": 7200, "the northern trust": 7200,
    "genesis invitational": 7322, "genesis open": 7322,
    "at&t pebble beach pro-am": 6972, "at&t pebble beach": 6972,
    "the american express": 7060, "desert classic": 7060,
    "the american express championship": 7060,
    "sony open in hawaii": 7044, "sony open": 7044,
    "arnold palmer invitational presented by mastercard": 7466,
    "arnold palmer invitational": 7466,
    "the honda classic": 7140, "honda classic": 7140,
    "valspar championship": 7340,
    "waste management phoenix open": 7261, "wmphoenix open": 7261, "phoenix open": 7261,
    "farmers insurance open": 7765,
    "sentry tournament of champions": 7596, "sentry": 7596,
    "tournament of champions": 7596,
    "memorial tournament presented by workday": 7392, "memorial tournament": 7392,
    "rbc canadian open": 7108, "canadian open": 7108,
    "rbc heritage": 7099,
    "wells fargo championship": 7521,
    "the cj cup byron nelson": 7468, "at&t byron nelson": 7468,
    "charles schwab challenge": 7204, "the charles schwab challenge": 7204,
    "rocket mortgage classic": 7376,
    "john deere classic": 7268,
    "travelers championship": 6841, "the travelers championship": 6841,
    "wyndham championship": 7131,
    "sanderson farms championship": 7461,
    "shriners children's open": 7255, "shriners hospitals for children open": 7255,
    "zozo championship": 7041, "the zozo championship": 7041,
    "world wide technology championship at mayakoba": 7008,
    "mayakoba golf classic": 7008,
    "the rsm classic": 7005, "rsm classic": 7005,
    "hero world challenge": 7302,
    "wgc-dell technologies match play": 7108, "dell technologies match play": 7108,
    "wgc-workday championship": 7474,
    "wgc-mexico championship": 7049,
    "wgc-hsbc champions": 7241,
    "3m open": 7431, "the 3m open": 7431,
    "cognizant classic in the palm beaches": 7054,
    "puerto rico open": 7513,
    "valero texas open": 7494, "texas open": 7494,
    "corales puntacana championship": 7677,
    "korn ferry tour championship": 7300,
    "butterfield bermuda championship": 7000, "bermuda championship": 7000,
    "nitto atp finals": 7200,
    "ace group classic": 7021,
    "truist championship": 7521,
    "cj cup": 7500, "the cj cup": 7500,
}

data["_schema"]["yardage"] = "Course yardage (yards) on championship layout"

updated, fallback = 0, 0
for k, v in data.items():
    if k.startswith("_") or not isinstance(v, dict):
        continue
    if k in YARDAGE:
        v["yardage"] = YARDAGE[k]
    else:
        v["yardage"] = 7250  # PGA Tour average fallback
        fallback += 1
        print(f"  [fallback] {k!r} -> 7250")
    updated += 1

p.write_text(json.dumps(data, indent=2))
print(f"Done: {updated} entries updated, {fallback} used fallback yardage")
