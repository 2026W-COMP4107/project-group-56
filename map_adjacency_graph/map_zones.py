"""
map_zones.py
------------
Zone definitions, adjacency graphs, and pathfinding for Valorant maps.

Usage:
    from map_zones import get_zone, get_path, get_zone_centroid

    zone = get_zone(x, y, "Bind")           # raw game coords → zone name
    path = get_path("attacker_spawn", "b_site", "Bind")  # ["attacker_spawn", "b_long", ...]
    cx, cy = get_zone_centroid("a_site", "Bind")         # normalized center of zone
"""

import math
import networkx as nx
from typing import Optional

# ── Coordinate normalization ───────────────────────────────────────────────────

MAP_SCALARS: dict[str, dict[str, float]] = {
    "Bind":     {"xMultiplier": 5.9e-05,  "xScalarToAdd": 0.576941, "yMultiplier": -5.9e-05,  "yScalarToAdd": 0.967566},
    "Split":    {"xMultiplier": 7.8e-05,  "xScalarToAdd": 0.842188, "yMultiplier": -7.8e-05,  "yScalarToAdd": 0.697578},
    "Breeze":   {"xMultiplier": 7.0e-05,  "xScalarToAdd": 0.465123, "yMultiplier": -7.0e-05,  "yScalarToAdd": 0.833078},
    "Fracture": {"xMultiplier": 7.8e-05,  "xScalarToAdd": 0.556952, "yMultiplier": -7.8e-05,  "yScalarToAdd": 1.155886},
    "Pearl":    {"xMultiplier": 7.8e-05,  "xScalarToAdd": 0.480469, "yMultiplier": -7.8e-05,  "yScalarToAdd": 0.916016},
}

def normalize_coord(x: float, y: float, map_name: str) -> tuple[float, float]:
    """Convert raw game coords to normalized [0,1] minimap coords."""
    s = MAP_SCALARS[map_name]
    return (
        s["xMultiplier"] * x + s["xScalarToAdd"],
        s["yMultiplier"] * y + s["yScalarToAdd"],
    )

# ── Point-in-polygon (ray casting) ────────────────────────────────────────────

def point_in_polygon(px: float, py: float, polygon: list[list[float]]) -> bool:
    """
    Ray casting algorithm — returns True if (px, py) is inside the polygon.
    polygon is a list of [x, y] pairs in normalized coords.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

# ── Zone polygon definitions ───────────────────────────────────────────────────
# Points are already in normalized coords (output of normalize_coord).
# Add other maps here as your partner fills them in.

ZONE_POLYGONS: dict[str, list[dict]] = {
    "Bind": [
        {"name": "defender_spawn", "points": [[-0.5402,-0.0604],[-0.5414,0.0073],[-0.4336,0.0476],[-0.3637,0.0488],[-0.365,0.0844],[-0.2485,0.0856],[-0.2473,0.0405],[-0.36,-0.0093],[-0.3588,-0.0343],[-0.3061,-0.0817],[-0.3074,-0.1707],[-0.3686,-0.1719],[-0.3723,-0.109],[-0.4287,-0.1078]]},
        {"name": "heaven",         "points": [[-0.3711,0.0903],[-0.3123,0.0903],[-0.2853,0.1935],[-0.2841,0.2517],[-0.3061,0.26],[-0.3699,0.2612]]},
        {"name": "a_site",         "points": [[-0.2853,0.0891],[-0.2853,0.2671],[-0.224,0.2695],[-0.1419,0.2683],[-0.1468,0.1722],[-0.1162,0.1686],[-0.1088,0.0986],[-0.1934,0.0927],[-0.1971,0.1164],[-0.2424,0.1129],[-0.24,0.0903]]},
        {"name": "lamps",          "points": [[-0.2363,0.1034],[-0.1983,0.1034],[-0.1995,0.0808],[-0.1603,0.082],[-0.1554,0.0191],[-0.2412,0.0179]]},
        {"name": "a_short",        "points": [[-0.1542,0.0595],[-0.1542,0.0963],[-0.0414,0.0974],[-0.0414,0.1781],[0.0407,0.1781],[0.0395,0.0215],[-0.1064,0.0227]]},
        {"name": "a_lobby",        "points": [[0.0456,0.1105],[0.0431,0.2291],[0.0064,0.2279],[0.0039,0.2588],[-0.039,0.26],[-0.0377,0.3217],[0.1069,0.3264],[0.1081,0.1093]]},
        {"name": "showers",        "points": [[-0.1358,0.1876],[-0.1395,0.33],[-0.0684,0.33],[-0.0672,0.3169],[-0.0426,0.3157],[-0.0414,0.2623],[-0.0966,0.2623],[-0.0953,0.1888]]},
        {"name": "a_teleporter",   "points": [[-0.0549,0.324],[0.0051,0.3252],[0.0088,0.3845],[-0.0623,0.3869]]},
        {"name": "b_teleporter",   "points": [[-0.137,-0.1149],[-0.0623,-0.1126],[-0.0623,-0.0497],[-0.1395,-0.0485]]},
        {"name": "cave",           "points": [[0.1449,0.1496],[0.2061,0.1461],[0.2343,0.0334],[0.1963,-0.0509],[0.1608,-0.0437],[0.1473,-0.0129]]},
        {"name": "a_link",         "points": [[0.0358,-0.0876],[0.0762,-0.0876],[0.075,-0.007],[0.0297,-0.0034],[0.0162,0.0215],[-0.039,0.0227],[-0.0402,-0.0437],[0.0309,-0.0449]]},
        {"name": "b_link",         "points": [[0.0358,-0.0876],[0.1473,-0.0829],[0.151,-0.2348],[0.0346,-0.2288]]},
        {"name": "b_lobby",        "points": [[-0.0059,-0.3842],[-0.0059,-0.2265],[0.1608,-0.2098],[0.1522,-0.3866]]},
        {"name": "b_short",        "points": [[0.0309,-0.23],[-0.0267,-0.2324],[-0.0279,-0.1885],[-0.1223,-0.1826],[-0.1407,-0.1422],[-0.1284,-0.1161],[0.0284,-0.1126]]},
        {"name": "attacker_spawn", "points": [[0.1449,-0.1031],[0.1424,-0.0627],[0.1191,-0.058],[0.1118,-0.0402],[0.151,-0.0378],[0.1706,-0.0615],[0.2196,-0.0544],[0.3225,0.0429],[0.1963,0.1496],[0.1069,0.1544],[0.1081,0.2161],[0.227,0.2196],[0.4279,0.0785],[0.4316,0.0144],[0.2527,-0.0829]]},
        {"name": "b_long",         "points": [[0.0002,-0.3771],[-0.001,-0.3226],[-0.1468,-0.3261],[-0.148,-0.4032],[-0.0794,-0.4044],[-0.0733,-0.3783]]},
        {"name": "b_garden",       "points": [[-0.1493,-0.3285],[-0.1505,-0.2656],[-0.088,-0.2644],[-0.0794,-0.281],[-0.0819,-0.3178],[-0.0966,-0.3273]]},
        {"name": "b_halls",        "points": [[-0.3686,-0.1778],[-0.3135,-0.1766],[-0.3196,-0.3095],[-0.3699,-0.3107]]},
        {"name": "b_elbow",        "points": [[-0.3699,-0.3119],[-0.24,-0.3131],[-0.24,-0.3439],[-0.1983,-0.3487],[-0.1983,-0.4044],[-0.2988,-0.4032],[-0.3,-0.3439],[-0.3699,-0.3427]]},
        {"name": "hookah",         "points": [[-0.1787,-0.2526],[-0.1775,-0.198],[-0.1174,-0.1861],[-0.0463,-0.1873],[-0.0451,-0.2573]]},
        {"name": "b_site",         "points": [[-0.3172,-0.3095],[-0.3172,-0.1885],[-0.2853,-0.1861],[-0.2804,-0.1766],[-0.2314,-0.1766],[-0.2314,-0.1968],[-0.1799,-0.198],[-0.1799,-0.2573],[-0.1517,-0.2763],[-0.1505,-0.3475],[-0.235,-0.3475],[-0.235,-0.3083]]},
    ],
    "Breeze": [
        {"name": "b_site",         "points": [[-0.1253,-0.2795],[0.1253,-0.276],[0.1253,-0.1925],[0.1418,-0.1913],[0.1429,-0.1299],[0.1206,-0.1345],[0.1229,-0.0788],[0.0535,-0.0927],[0.0335,0.0314],[-0.0312,0.0372],[-0.0382,-0.0046],[-0.1006,-0.0081],[-0.1324,0.0128],[-0.1865,-0.0579],[-0.1418,-0.0962]]},
        {"name": "b_main",         "points": [[0.1394,-0.1902],[0.1571,-0.1925],[0.1594,-0.2227],[0.3324,-0.2273],[0.3324,-0.1101],[0.2112,-0.1055],[0.21,-0.0939],[0.1606,-0.0962],[0.1606,-0.1368],[0.1371,-0.1345]]},
        {"name": "b_window",       "points": [[0.3312,-0.2099],[0.4088,-0.2111],[0.4088,-0.0266],[0.3288,-0.0278]]},
        {"name": "mid_canon",      "points": [[0.33,-0.0823],[0.33,-0.0289],[0.4418,0.0894],[0.3512,0.0963],[0.2441,-0.0231],[0.2418,-0.0893]]},
        {"name": "b_elbow",        "points": [[0.1618,-0.0974],[0.2135,-0.0939],[0.2112,-0.0197],[0.1453,-0.0092],[0.1441,0.1207],[0.0676,0.123],[0.0676,0.065],[0.0818,0.0638],[0.0794,-0.0336],[0.1182,-0.0753],[0.1524,-0.0742]]},
        {"name": "mid_bottom",     "points": [[0.35,0.0894],[0.4829,0.087],[0.4829,0.2889],[0.35,0.2854]]},
        {"name": "attacker_spawn", "points": [[0.49,0.087],[0.6924,0.0847],[0.6876,0.239],[0.5629,0.2355],[0.5629,0.3248],[0.4841,0.3248]]},
        {"name": "a_lobby",        "points": [[0.49,0.2738],[0.5665,0.275],[0.5629,0.3237],[0.6088,0.3585],[0.6135,0.4559],[0.4512,0.4571],[0.3982,0.4153],[0.3488,0.4107],[0.3441,0.3747],[0.3429,0.3237],[0.4818,0.3202]]},
        {"name": "a_elbow",        "points": [[0.2547,0.2773],[0.3006,0.2761],[0.2971,0.3202],[0.3547,0.3167],[0.3429,0.3759],[0.2512,0.3736]]},
        {"name": "a_main",         "points": [[0.4112,0.4199],[0.4512,0.4431],[0.4041,0.5127],[0.4041,0.6392],[0.3076,0.6392],[0.3135,0.442],[0.3818,0.4362]]},
        {"name": "a_site",         "points": [[0.3124,0.4895],[0.3135,0.6438],[0.1735,0.7018],[0.0676,0.7018],[0.0724,0.6195],[-0.0406,0.522],[-0.0406,0.3863],[0.2665,0.3852],[0.2582,0.4907]]},
        {"name": "a_elbow_site",   "points": [[0.0288,0.3898],[0.0853,0.3898],[0.0771,0.3225],[-0.0429,0.3248],[-0.0418,0.3689],[0.0253,0.3701]]},
        {"name": "a_bridge",       "points": [[-0.0441,0.4756],[-0.0441,0.3921],[-0.0829,0.3991],[-0.0841,0.4803]]},
        {"name": "defender_spawn", "points": [[-0.2335,0.1566],[-0.1147,0.1555],[-0.1124,0.3828],[-0.2829,0.3828],[-0.2876,0.2332]]},
        {"name": "spawn_a",        "points": [[-0.2159,0.355],[-0.1041,0.3492],[-0.0818,0.4083],[-0.0806,0.4977],[-0.2171,0.4327]]},
        {"name": "spawn_b",        "points": [[-0.23,0.1624],[-0.1182,0.159],[-0.1206,0.0267],[-0.19,-0.0568],[-0.2371,-0.0243]]},
        {"name": "spawn_mid",      "points": [[-0.1147,0.1682],[-0.0394,0.1659],[-0.0429,0.3944],[-0.0841,0.3933],[-0.0853,0.3132],[-0.1171,0.3156]]},
        {"name": "mid_nest",       "points": [[-0.1124,0.094],[-0.0347,0.0894],[-0.0312,0.1868],[-0.1124,0.1926]]},
        {"name": "mid_top",        "points": [[-0.0276,0.0186],[0.0359,0.0151],[0.0312,0.0743],[0.0735,0.0731],[0.0724,0.13],[0.0712,0.2088],[-0.0335,0.2193]]},
        {"name": "a_doors",        "points": [[0.1312,0.3805],[0.2241,0.384],[0.2206,0.297],[0.13,0.3005]]},
        {"name": "mid_pillar",     "points": [[0.07,0.13],[0.0735,0.2135],[0.1324,0.232],[0.1394,0.2877],[0.31,0.2854],[0.3171,0.1207]]},
    ],
    "Fracture": [
        {"name": "attacker_side_bridge", "points": [[-0.6357,-0.4238],[-0.5341,-0.338],[-0.5487,-0.2657],[-0.5068,-0.2719],[-0.5068,-0.1148],[-0.5603,-0.1127],[-0.5592,-0.0786],[-0.6536,-0.0796],[-0.6766,-0.1551],[-0.6745,-0.3835]]},
        {"name": "b_bench",        "points": [[-0.6326,-0.4207],[-0.5477,-0.3339],[-0.5288,-0.3556],[-0.5058,-0.3576],[-0.5005,-0.3215],[-0.4722,-0.2956],[-0.3884,-0.2956],[-0.3873,-0.3545],[-0.3318,-0.3535],[-0.3318,-0.4259],[-0.4272,-0.4228],[-0.468,-0.4507],[-0.6242,-0.4527]]},
        {"name": "b_arcade",       "points": [[-0.4272,-0.4238],[-0.3412,-0.4269],[-0.3464,-0.4434],[-0.2521,-0.4465],[-0.2542,-0.5241],[-0.3108,-0.5189],[-0.3118,-0.5582],[-0.3873,-0.5623],[-0.4303,-0.523]]},
        {"name": "b_tower",        "points": [[-0.3685,-0.5592],[-0.2573,-0.5623],[-0.2552,-0.6946],[-0.2972,-0.6905],[-0.2982,-0.5995],[-0.3737,-0.5954]]},
        {"name": "b_site",         "points": [[-0.2552,-0.6698],[-0.2563,-0.4321],[-0.0949,-0.43],[-0.1043,-0.6781]]},
        {"name": "b_main",         "points": [[-0.0907,-0.6295],[0.1682,-0.5912],[0.1745,-0.3597],[0.0791,-0.3576],[-0.0393,-0.4269],[-0.0026,-0.5189],[-0.0865,-0.5396]]},
        {"name": "b_tunnel",       "points": [[-0.0215,-0.5023],[-0.2217,-0.5034],[-0.2227,-0.4052],[-0.3265,-0.4042],[-0.3265,-0.3483],[-0.164,-0.3504],[-0.1609,-0.4279],[-0.032,-0.4248]]},
        {"name": "spawn_b",        "points": [[0.0886,-0.5054],[0.0079,-0.3928],[0.0875,-0.3628],[0.0341,-0.3277],[0.0288,-0.2615],[0.185,-0.2688],[0.1766,-0.5085]]},
        {"name": "attacker_spawn", "points": [[0.0383,-0.2584],[0.2479,-0.2667],[0.2489,-0.123],[0.032,-0.1282]]},
        {"name": "spawn_a",        "points": [[0.1221,-0.1551],[0.2479,-0.1509],[0.2416,-0.0011],[0.1211,0.001]]},
        {"name": "a_main",         "points": [[0.1284,-0.1075],[0.0026,-0.1055],[-0.032,0.0268],[-0.0288,0.092],[-0.1179,0.0909],[-0.1127,0.1798],[0.1127,0.1777],[0.1106,0.0186],[0.0791,0.0196],[0.0802,-0.0393],[0.1326,-0.0393]]},
        {"name": "a_gate",         "points": [[-0.6494,-0.1427],[-0.5425,-0.1282],[-0.5446,0.0517],[-0.6431,0.0506]]},
        {"name": "a_dish",         "points": [[-0.5498,-0.0579],[-0.5519,0.0496],[-0.4345,0.0475],[-0.3685,0.1405],[-0.3695,-0.0589]]},
        {"name": "a_drop",         "points": [[-0.3705,0.0279],[-0.3716,0.1405],[-0.2437,0.1302],[-0.2427,0.0206]]},
        {"name": "a_site",         "points": [[-0.2427,0.0279],[-0.2437,0.125],[-0.293,0.1312],[-0.2898,0.2367],[-0.1053,0.2408],[-0.1116,0.1064],[-0.1368,0.0868],[-0.12,0.0692],[-0.1211,-0.0228],[-0.2416,-0.0207]]},
        {"name": "defenders_tower","points": [[-0.0498,-0.0931],[-0.0529,-0.1509],[-0.1735,-0.154],[-0.1756,-0.09]]},
        {"name": "a_rope",         "points": [[-0.1724,-0.0889],[0.0079,-0.0931],[0.0089,-0.0114],[-0.0372,0.0413],[-0.0938,-0.0207],[-0.1263,-0.0176],[-0.1882,-0.0166],[-0.1882,-0.0868]]},
        {"name": "defenders_spawn","points": [[-0.1714,-0.0941],[-0.1703,-0.1757],[-0.2385,-0.1747],[-0.2353,-0.1995],[-0.1693,-0.2067],[-0.1651,-0.277],[-0.2248,-0.3277],[-0.336,-0.3246],[-0.3286,-0.0434],[-0.2416,-0.06],[-0.2385,-0.0931]]},
        {"name": "b_canteen",      "points": [[-0.2123,-0.3266],[-0.1473,-0.3887],[-0.1473,-0.4724],[-0.0917,-0.4796],[-0.0687,-0.3535],[-0.163,-0.2677]]},
        {"name": "b_generator",    "points": [[-0.2783,-0.3184],[-0.337,-0.3215],[-0.336,-0.399],[-0.2563,-0.4031],[-0.2552,-0.4434],[-0.1504,-0.4362],[-0.1494,-0.3856],[-0.1913,-0.3432]]},
    ],
    "Pearl": [
        {"name": "a_secret",       "points": [[-0.3065,0.3144],[-0.2562,0.3155],[-0.255,0.3845],[-0.1986,0.4055],[-0.1606,0.4464],[-0.1582,0.4791],[-0.206,0.5223],[-0.3077,0.4265]]},
        {"name": "defender_spawn", "points": [[-0.3751,0.1146],[-0.3763,0.2431],[-0.3101,0.2478],[-0.3089,0.3038],[-0.206,0.3027],[-0.2072,0.2384],[-0.1729,0.2396],[-0.1704,0.1683],[-0.1459,0.1683],[-0.1423,0.1029]]},
        {"name": "a_flowers",      "points": [[-0.1067,0.3728],[-0.1668,0.4078],[-0.2525,0.3716],[-0.2538,0.3085],[-0.168,0.3085]]},
        {"name": "a_dugout",       "points": [[-0.1435,0.4873],[-0.1275,0.5001],[-0.0528,0.4978],[-0.0516,0.548],[-0.1668,0.5492],[-0.1864,0.5305]]},
        {"name": "a_site",         "points": [[0.0318,0.4043],[0.0318,0.5363],[-0.0552,0.5363],[-0.0565,0.4966],[-0.1288,0.4943],[-0.1533,0.4732],[-0.1557,0.4078],[-0.1116,0.3856]]},
        {"name": "a_link",         "points": [[-0.0687,0.3903],[-0.0087,0.3903],[-0.0062,0.3447],[0.0734,0.3447],[0.0661,0.1754],[0.0195,0.1765],[0.0171,0.2595],[-0.0957,0.2606],[-0.0969,0.3015]]},
        {"name": "mid_connector",  "points": [[-0.0712,0.1473],[-0.07,0.2092],[-0.0295,0.2232],[0.06,0.2209],[0.0612,0.1789],[0.0097,0.1777],[0.0097,0.159],[0.0403,0.0901],[0.0759,0.0866],[0.0722,0.048],[-0.0454,0.0457]]},
        {"name": "water",          "points": [[-0.07,0.152],[-0.0712,0.2057],[-0.1349,0.2127],[-0.1398,0.2396],[-0.1704,0.2373],[-0.1692,0.1695],[-0.1398,0.166]]},
        {"name": "mid_doors",      "points": [[0.1568,0.0691],[0.0685,0.0667],[0.0698,0.1415],[0.1666,0.1426]]},
        {"name": "b_link",         "points": [[0.0747,0.0632],[0.1531,0.0585],[0.1543,0.0083],[0.1874,0.006],[0.1899,-0.0793],[0.1629,-0.1073],[0.1629,-0.1891],[0.0881,-0.1879],[0.082,-0.0302],[0.0575,-0.0279],[0.0575,0.0632]]},
        {"name": "b_halls",        "points": [[-0.0601,-0.3013],[-0.0577,-0.1272],[-0.1275,-0.126],[-0.13,-0.3024]]},
        {"name": "art",            "points": [[0.1702,0.1613],[0.1837,0.2665],[0.1825,0.3611],[0.0673,0.3611],[0.0685,0.263],[0.1053,0.2641],[0.1114,0.1602]]},
        {"name": "mid",            "points": [[0.1678,0.0574],[0.1653,0.1543],[0.1163,0.1578],[0.1163,0.2127],[0.1923,0.2209],[0.1997,0.2525],[0.3222,0.2513],[0.3259,0.1064],[0.2903,0.0667]]},
        {"name": "top_mid",        "points": [[0.3271,0.1625],[0.3271,0.2945],[0.3859,0.2933],[0.3847,0.1625]]},
        {"name": "b_shops",        "points": [[0.2193,-0.0314],[0.2229,0.0609],[0.4325,0.0691],[0.4276,-0.0326]]},
        {"name": "attacker_spawn", "points": [[0.4313,-0.0256],[0.4987,-0.0291],[0.5048,0.0293],[0.6102,0.0363],[0.6053,0.2454],[0.4742,0.2478],[0.4717,0.3167],[0.3896,0.3868],[0.3859,0.2945],[0.3835,0.0866],[0.4313,0.0714]]},
        {"name": "restaurant",     "points": [[0.3712,0.3003],[0.3712,0.4043],[0.18,0.4043],[0.1813,0.298]]},
        {"name": "b_main",         "points": [[0.3259,-0.0443],[0.4962,-0.0478],[0.4987,-0.2136],[0.3136,-0.2148],[0.3124,-0.0489]]},
        {"name": "b_long",         "points": [[0.4215,-0.2487],[0.0379,-0.2487],[0.044,-0.3527],[0.1359,-0.355],[0.1935,-0.3959],[0.2278,-0.3959],[0.2364,-0.3585],[0.4215,-0.3585]]},
        {"name": "b_site",         "points": [[-0.0785,-0.3503],[-0.0761,-0.3071],[-0.0589,-0.2907],[-0.0614,-0.126],[0.0832,-0.126],[0.0832,-0.265],[0.0244,-0.2627],[0.0379,-0.2989],[0.0354,-0.3527]]},
        {"name": "b_heaven",       "points": [[0.055,-0.077],[-0.1766,-0.0746],[-0.1802,-0.126],[0.0575,-0.1249]]},
        {"name": "defender_b_side","points": [[-0.0087,-0.0735],[-0.1839,-0.0746],[-0.2158,0.0188],[-0.2133,0.1134],[-0.0859,0.1076],[-0.0859,0.006],[-0.1018,-0.0069],[-0.0062,-0.0151]]},
        {"name": "a_main",         "points": [[-0.1839,0.0188],[-0.1849,0.1134],[-0.0859,0.1076],[-0.0859,0.006],[-0.1018,-0.0069],[-0.0062,-0.0151],[-0.0087,-0.0735],[-0.1766,-0.0746],[-0.1802,-0.126],[0.0575,-0.1249]]},###
    ],
    "Split": [
        {"name": "b_alley",        "points": [[0.3085,-0.2721],[0.0922,-0.2733],[0.0874,-0.3836],[0.3073,-0.3861]]},
        {"name": "b_stairs",       "points": [[0.0839,-0.3273],[0.0589,-0.326],[0.0553,-0.1949],[0.1504,-0.1961],[0.1504,-0.2353],[0.0969,-0.2341]]},
        {"name": "a_screens",      "points": [[-0.199,-0.3383],[-0.2775,-0.3383],[-0.2763,-0.3677],[-0.4011,-0.3701],[-0.4035,-0.4412],[-0.1931,-0.4449]]},
        {"name": "defender_spawn", "points": [[-0.1943,-0.3346],[-0.1919,-0.3971],[-0.0671,-0.3934],[-0.0623,-0.5282],[0.0339,-0.5282],[0.0327,-0.3273],[0.0518,-0.3285],[0.053,-0.2586],[-0.0362,-0.2599],[-0.0445,-0.3248],[-0.123,-0.3309]]},
        {"name": "b_heaven",       "points": [[0.1742,-0.1814],[0.173,-0.0025],[0.1053,-0.0025],[0.1029,-0.1961],[0.1742,-0.1974]]},
        {"name": "mail",           "points": [[0.1029,-0.0429],[0.1029,-0.1312],[0.0339,-0.1238],[0.0363,-0.0417]]},
        {"name": "vents",          "points": [[-0.1348,-0.2072],[-0.1372,-0.103],[-0.0029,-0.1091],[0.0268,-0.1606],[0.0137,-0.1839],[-0.0849,-0.1839],[-0.0837,-0.2108]]},
        {"name": "top_mid",        "points": [[0.0351,-0.1103],[-0.0505,-0.1079],[-0.0516,0.0355],[0.0589,0.0379],[0.0506,-0.0062],[0.0375,-0.0074]]},
        {"name": "bottom_mid",     "points": [[0.1421,0.0477],[0.1421,0.1605],[-0.1253,0.1605],[-0.1253,0.0367]]},
        {"name": "sewers",         "points": [[-0.1253,0.0306],[-0.1241,0.1433],[-0.3488,0.1409],[-0.3476,0.0343]]},
        {"name": "b_rafters",      "points": [[0.2277,-0.2795],[0.2277,-0.1447],[0.2098,-0.1447],[0.211,-0.2795]]},
        {"name": "b_hell",         "points": [[0.1766,-0.1447],[0.2669,-0.1447],[0.2693,-0.1054],[0.2479,-0.0785],[0.1754,-0.076]]},
        {"name": "back_b_site",    "points": [[0.3751,-0.3358],[0.3751,-0.228],[0.4464,-0.2292],[0.4464,-0.342]]},
        {"name": "b_site",         "points": [[0.3727,-0.2746],[0.3762,-0.2047],[0.4523,-0.2084],[0.4511,-0.1005],[0.274,-0.1042],[0.2705,-0.1447],[0.2253,-0.1447],[0.2277,-0.2709]]},
        {"name": "b_main",         "points": [[0.4143,-0.1018],[0.2871,-0.0956],[0.236,-0.038],[0.2372,0.0477],[0.3251,0.049],[0.3275,0.0808],[0.4155,0.0784]]},
        {"name": "b_lobby",        "points": [[0.3156,0.0453],[0.2015,0.0477],[0.198,0.1825],[0.1445,0.1887],[0.1445,0.3014],[0.3216,0.2941]]},
        {"name": "b_link",         "points": [[0.2003,0.0416],[0.1302,0.0379],[0.1314,0.1813],[0.2039,0.1776]]},
        {"name": "attacker_spawn", "points": [[-0.344,0.1458],[-0.3428,0.2218],[-0.2466,0.2193],[-0.2442,0.2916],[-0.0802,0.2941],[-0.0754,0.4178],[0.0827,0.408],[0.1373,0.3002],[0.1362,0.2193],[-0.0516,0.1997],[-0.142,0.2205],[-0.1479,0.147]]},
        {"name": "a_lobby",        "points": [[-0.3452,0.212],[-0.344,0.0955],[-0.3654,0.0968],[-0.3642,0.0306],[-0.4819,0.0294],[-0.4807,0.1029],[-0.4379,0.1066],[-0.4356,0.2144]]},
        {"name": "a_main",         "points": [[-0.268,-0.1128],[-0.2703,0.0294],[-0.4546,0.0318],[-0.4534,-0.1165]]},
        {"name": "a_ramps",        "points": [[-0.2703,-0.1054],[-0.2727,0.0134],[-0.1753,0.0147],[-0.142,-0.0197],[-0.1455,-0.1667],[-0.2489,-0.163]]},
        {"name": "a_heaven",       "points": [[-0.1337,-0.1692],[-0.1325,-0.2035],[-0.1646,-0.2047],[-0.1669,-0.25],[-0.1277,-0.2525],[-0.1277,-0.3407],[-0.2644,-0.3371],[-0.2668,-0.2844],[-0.2632,-0.2035],[-0.2382,-0.1679]]},
        {"name": "a_rafters",      "points": [[-0.2632,-0.2856],[-0.2632,-0.1937],[-0.3036,-0.1949],[-0.3036,-0.2157],[-0.287,-0.2133],[-0.2858,-0.2868]]},
        {"name": "a_elbow",        "points": [[-0.3595,-0.3628],[-0.3607,-0.3187],[-0.4558,-0.3162],[-0.4558,-0.3714]]},
        {"name": "a_site",         "points": [[-0.3084,-0.3677],[-0.3642,-0.3689],[-0.363,-0.277],[-0.4237,-0.2746],[-0.426,-0.3175],[-0.5152,-0.3187],[-0.514,-0.1214],[-0.4807,-0.1214],[-0.3975,-0.163],[-0.3607,-0.1373],[-0.2715,-0.1349],[-0.2703,-0.1949],[-0.306,-0.1937],[-0.3084,-0.2096],[-0.2917,-0.2108],[-0.2882,-0.3236]]},
    ],
}

# ── Adjacency graphs ───────────────────────────────────────────────────────────

ADJACENCY: dict[str, dict[str, list[str]]] = {
    "Bind": {
        "attacker_spawn": ["a_lobby", "b_long"],
        "a_lobby":        ["attacker_spawn", "a_short", "a_link"],
        "a_short":        ["a_lobby", "a_site"],
        "a_link":         ["a_lobby", "showers", "a_site"],
        "showers":        ["a_link", "b_link"],
        "a_site":         ["a_short", "a_link", "heaven", "lamps", "a_teleporter"],
        "heaven":         ["a_site", "defender_spawn"],
        "lamps":          ["a_site"],
        "a_teleporter":   ["a_site", "b_teleporter"],   # teleporter connection
        "b_teleporter":   ["a_teleporter", "hookah"],
        "b_long":         ["attacker_spawn", "b_lobby"],
        "b_lobby":        ["b_long", "b_short", "b_garden"],
        "b_garden":       ["b_lobby", "b_halls"],
        "b_halls":        ["b_garden", "b_elbow"],
        "b_elbow":        ["b_halls", "hookah", "b_short"],
        "b_short":        ["b_lobby", "b_elbow", "b_site"],
        "hookah":         ["b_teleporter", "b_elbow", "b_site"],
        "b_site":         ["b_short", "hookah", "b_link", "cave"],
        "b_link":         ["b_site", "showers", "cave"],
        "cave":           ["b_link", "b_site", "defender_spawn"],
        "defender_spawn": ["heaven", "cave"],
    },
    "Breeze": {
        "attacker_spawn": ["mid_bottom", "a_lobby"],
        "mid_bottom":     ["attacker_spawn", "mid_canon", "mid_pillar"],
        "mid_canon":      ["mid_bottom", "b_window","b_elbow"],
        "b_window":       ["mid_canon", "b_main"],
        "b_main":         ["b_window", "b_elbow", "b_site"],
        "b_elbow":        ["b_main", "mid_top", "mid_canon"],
        "b_site":         ["b_main", "mid_top","spawn_b"],
        "mid_pillar":     ["a_elbow", "mid_top","a_doors","mid_bottom"],
        "mid_top":        ["mid_pillar", "mid_nest", "b_site","b_elbow"],
        "mid_nest":       ["mid_top", "spawn_mid"],
        "spawn_mid":      ["mid_nest", "defender_spawn", "a_bridge", "a_elbow_site"],
        "spawn_b":        ["b_site", "defender_spawn"],
        "spawn_a":        ["defender_spawn", "a_bridge", "a_site"],
        "a_main":         ["a_lobby", "a_site"],
        "defender_spawn": ["spawn_b", "spawn_mid", "spawn_a"],
        "a_bridge":       ["spawn_a", "a_site", "a_elbow_site", "spawn_mid"],
        "a_elbow_site":   ["a_site", "a_bridge","spawn_mid"],
        "a_doors":        ["mid_pillar", "a_site"],
        "a_elbow":        ["mid_pillar", "a_lobby"],
        "a_lobby":        ["attacker_spawn", "a_elbow", "a_main"],
        "a_site":         ["a_bridge", "a_elbow_site", "a_doors", "a_main","spawn_a"],
    },
    "Fracture": {
        "attacker_spawn":       ["attacker_side_bridge","spawn_b","spawn_a"],
        "spawn_b":              ["attacker_spawn","b_main"],
        "spawn_a":              ["attacker_spawn", "a_main"],
        "attacker_side_bridge": ["attacker_spawn", "a_gate", "b_bench"],
        "b_main":               ["spawn_b", "b_tunnel", "b_site"],
        "b_tunnel":             ["b_main", "b_arcade"],
        "b_canteen":            ["b_site", "b_generator", "defenders_spawn"],
        "b_generator":          ["defenders_spawn", "b_canteen", "b_site"],
        "b_bench":              ["b_arcade", "attacker_side_bridge", "b_tunnel"],
        "b_arcade":             ["b_bench", "b_tower","b_tunnel","b_site"],
        "b_tower":              ["b_arcade", "b_site"],
        "b_site":               ["b_generator", "b_tower", "b_canteen","b_main","b_arcade"],
        "a_gate":               ["attacker_side_bridge", "a_dish"],
        "a_dish":               ["a_gate", "a_drop"],
        "a_drop":               ["a_dish", "a_site"],
        "a_main":               ["spawn_a", "a_rope","a_site"],
        "a_rope":               ["a_main", "a_site", "defenders_tower"],
        "defenders_tower":      ["a_rope", "defenders_spawn"],
        "a_site":               ["a_drop", "a_rope", "defenders_spawn","a_main"],
        "defenders_spawn":      ["b_canteen", "defenders_tower", "a_site", "b_generator"],
    },
    "Pearl": {
        "attacker_spawn": ["b_shops", "restaurant", "top_mid"],
        "b_shops":        ["attacker_spawn", "b_main", "mid"],
        "b_main":         ["b_shops", "b_long"],
        "b_long":         ["b_main", "b_site"],
        "b_site":         ["b_long", "b_halls", "b_heaven","b_link"],
        "b_heaven":       ["b_site", "defender_b_side"],
        "b_halls":        ["b_site", "defender_b_side"],
        "defender_b_side":["b_heaven", "defender_spawn"],
        "defender_spawn": ["defender_b_side", "a_secret", "water","a_flowers"],
        "a_secret":       ["defender_spawn", "a_flowers","a_site","a_dugout"],
        "a_flowers":      ["a_secret", "defender_spawn", "a_site"],
        "water":          ["defender_spawn", "mid_connector"],
        "mid_connector":  ["water", "mid_doors", "a_link", "b_link"],
        "mid_doors":      ["mid_connector", "mid"],
        "mid":            ["mid_doors", "b_shops", "top_mid", "art"],
        "top_mid":        ["mid", "restaurant","attacker_spawn"],
        "art":            ["mid", "a_link"],
        "restaurant":     ["attacker_spawn", "top_mid", "a_main"],
        "a_link":         ["mid_connector", "a_site", "art"],
        "a_dugout":       ["a_site", "a_secret"],
        "a_site":         ["a_link", "a_dugout"," a_flowers", "a_secret", "a_main"],
        "b_link":         ["mid_connector", "mid_doors", "b_site"],###
        "a_main":         ["restaurant", "a_site"],
    },
    "Split": {
        "attacker_spawn": ["b_lobby", "a_lobby"],
        "b_lobby":        ["attacker_spawn", "b_link", "b_main"],
        "b_link":         ["b_lobby", "bottom_mid"],
        "b_main":         ["b_lobby", "b_site"],
        "b_hell":         ["b_site", "b_heaven"],
        "b_heaven":       ["b_hell", "b_site", "mail","b_stairs","b_rafters"],
        "b_site":         ["b_main", "b_hell", "b_heaven", "back_b_site", "b_rafters","b_alley"],
        "back_b_site":    ["b_site"],
        "b_rafters":      ["b_site", "b_heaven"],
        "b_alley":        ["b_stairs", "defender_spawn","b_site"],
        "b_stairs":       ["b_alley", "defender_spawn"],
        "mail":           ["b_heaven", "top_mid"],
        "vents":          ["top_mid", "a_heaven"],
        "top_mid":        ["mail", "vents", "bottom_mid"],
        "bottom_mid":     ["b_link", "top_mid", "sewers"],
        "sewers":         ["bottom_mid", "a_lobby"],
        "a_lobby":        ["attacker_spawn", "a_main","sewers"],
        "a_main":         ["a_lobby", "a_ramps","a_site"],
        "a_ramps":        ["a_main", "a_heaven"],
        "a_heaven":       ["vents","a_ramps", "a_rafters", "a_site","defender_spawn"],
        "a_rafters":      ["a_heaven", "a_site"],
        "a_elbow":        ["a_site", "a_screens"],
        "a_screens":      ["a_elbow", "defender_spawn"],
        "a_site":         ["a_elbow", "a_rafters","a_main","a_heaven"],
        "defender_spawn": ["a_screens", "a_heaven", "b_stairs", "b_alley"],
    },
}

# Teleporter edges have near-zero travel time on Bind
TELEPORTER_EDGES = {
    "Bind": {("a_teleporter", "b_teleporter"), ("b_teleporter", "a_teleporter")}
}

# ── Build NetworkX graphs ──────────────────────────────────────────────────────

_graphs: dict[str, nx.Graph] = {}

def _get_graph(map_name: str) -> nx.Graph:
    if map_name not in _graphs:
        G = nx.Graph()
        for node, neighbors in ADJACENCY[map_name].items():
            for nb in neighbors:
                G.add_edge(node, nb)
        _graphs[map_name] = G
    return _graphs[map_name]

# ── Zone centroid cache ────────────────────────────────────────────────────────

_centroids: dict[str, dict[str, tuple[float, float]]] = {}

def get_zone_centroid(zone_name: str, map_name: str) -> tuple[float, float]:
    """Return the average (x, y) of a zone's polygon points."""
    if map_name not in _centroids:
        _centroids[map_name] = {}
    if zone_name not in _centroids[map_name]:
        zones = {z["name"]: z["points"] for z in ZONE_POLYGONS.get(map_name, [])}
        pts = zones.get(zone_name, [])
        if not pts:
            return (0.0, 0.0)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        _centroids[map_name][zone_name] = (cx, cy)
    return _centroids[map_name][zone_name]

# ── Public API ─────────────────────────────────────────────────────────────────

def get_zone(x: float, y: float, map_name: str) -> str:
    """
    Given raw game coordinates and a map name, return the zone the player is in.
    Returns "unknown" if the point doesn't fall inside any defined polygon.
    """
    if map_name not in MAP_SCALARS:
        return "unknown"

    nx_coord, ny_coord = normalize_coord(x, y, map_name)

    for zone in ZONE_POLYGONS.get(map_name, []):
        if point_in_polygon(nx_coord, ny_coord, zone["points"]):
            return zone["name"]

    return "unknown"


def get_path(start: str, end: str, map_name: str) -> list[str]:
    """
    Return shortest zone-to-zone path respecting map topology.
    Returns empty list if no path exists or map not defined.

    Example:
        get_path("attacker_spawn", "b_site", "Bind")
        → ["attacker_spawn", "b_long", "b_lobby", "b_short", "b_site"]
    """
    if map_name not in ADJACENCY:
        return []
    if start == end:
        return [start]
    try:
        G = _get_graph(map_name)
        return nx.shortest_path(G, start, end)
    except nx.NetworkXNoPath:
        return []
    except nx.NodeNotFound:
        return []


def interpolate_position(
    start_zone: str,
    end_zone: str,
    t: float,
    decay_constant: float,
    map_name: str,
) -> tuple[str, tuple[float, float]]:
    """
    Given a start zone, end zone, elapsed time t (seconds), and decay constant,
    return the predicted (zone_name, (norm_x, norm_y)) using decay-weighted
    pathfinding — never jumps through walls.

    weight = 1 - exp(-t / decay_constant)
    weight=0 → at start_zone, weight=1 → at end_zone
    """
    weight = 1 - math.exp(-t / decay_constant)
    path   = get_path(start_zone, end_zone, map_name)

    if not path:
        # No path found — stay at start
        return start_zone, get_zone_centroid(start_zone, map_name)

    # Map weight to position along path
    idx   = int(weight * (len(path) - 1))
    idx   = max(0, min(idx, len(path) - 1))
    zone  = path[idx]
    coord = get_zone_centroid(zone, map_name)
    return zone, coord


def label_site(zone_name: str) -> Optional[str]:
    """
    Map a zone name to a site label (A / B / Mid / None).
    Used by the tactical labeler.
    """
    if "a_site" in zone_name or zone_name in ("heaven", "lamps", "a_short"):
        return "A"
    if "b_site" in zone_name or zone_name in ("hookah", "b_short", "b_halls"):
        return "B"
    if zone_name in ("cave", "a_link", "b_link"):
        return "Mid"
    return None


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test zone lookup with a known kill location from your data
    test_cases = [
        (6277, -8492, "Bind"),    # from your sample JSON
        (-3193, -7266, "Bind"),
        (1953, -5847,  "Bind"),
    ]
    print("Zone lookup test:")
    for x, y, m in test_cases:
        zone = get_zone(x, y, m)
        cx, cy = get_zone_centroid(zone, m) if zone != "unknown" else (0, 0)
        print(f"  ({x:7}, {y:7}) → {zone:20s}  centroid=({cx:.3f}, {cy:.3f})")

    print("\nPathfinding test (Bind):")
    pairs = [
        ("attacker_spawn", "b_site"),
        ("attacker_spawn", "a_site"),
        ("defender_spawn", "b_site"),
        ("b_teleporter",   "a_site"),   # should use teleporter shortcut
    ]
    for start, end in pairs:
        path = get_path(start, end, "Bind")
        print(f"  {start:20s} → {end:15s} : {' → '.join(path)}")

    print("\nInterpolation test (attacker → b_site, t=15s, decay=20):")
    zone, coord = interpolate_position("attacker_spawn", "b_site", 15, 20, "Bind")
    print(f"  predicted zone: {zone}  coord: ({coord[0]:.3f}, {coord[1]:.3f})")
