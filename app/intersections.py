"""轮廓交叉清单（intersections，只读即时检查）。

对一份已存 SVG path 即时解析（复用 :func:`app.svggeo.analyze_contours`
的解析与曲线细分折线，见 :func:`app.svggeo.flatten_subpath`），在
**加工前**对每条**闭合子路径**做两类边对检查：

- **自交**：同一闭合环内所有**非相邻边**对（相邻边仅共享端点，
  属正常连接，跳过；闭合环含末点回首点的闭合边）；
- **互交**：不同闭合子路径之间的全部边对。

边对搜索分两层（粗筛 + 精判）：

- **粗筛（包围盒扫掠）**：对每条细分边建立**轴对齐包围盒（AABB）**，
  按 x 区间做事件扫掠（边的 xmin/xmax 处进入/退出活动集），仅对
  y 区间也重叠的活动边对做进一步检查——稀疏大路径的精确分类次数可由
  边数平方降到接近线性；
- **精判（现有相交分类）**：粗筛候选仍交给:func:`_classify` 做
  crossing / endpoint_contact / collinear_overlap 三类判定，密集重叠时
  所有包围盒相交的边对都会被枚举，清单完整不漏。

包围盒按 :data:`_BBOX_PAD_REL` × 边长**保守外扩**：精判所用的参数域
端点容差（:data:`_EPS_PAR`）、角域与线域相对容差（:data:`_EPS_ANGLE` /
:data:`_EPS_LINE`）都可能把**包围盒本不相交**的边对判成交点（近端点
接触、近共线重叠），外扩量取这些容差并留出安全余量，保证粗筛只可能
多报候选、绝不漏掉精判会报告的边对。

开放轮廓（未以 ``Z`` 闭合）不参与（列入 ``skipped_contours``）；
闭合但退化的子路径（面积为 0 或顶点不足）仍按字面意义参与边检查。

交点分三类：

- ``crossing``：两线段在内部真正交叉（交点严格位于两条边内部）；
- ``endpoint_contact``：端点接触（某条边的端点落在另一条边上，
  含环内非相邻边共享端点的“收尖/捏合”构型、T 型接触与零长度边点接触）；
- ``collinear_overlap``：共线重叠（两条边共线且共享一段非零长度
  区间；仅共享一点时归为 ``endpoint_contact``）。

**同坐标、同类型**的交点合并为一条，合并其涉及的全部（轮廓, 边索引）；
``collinear_overlap`` 的合并键为（字典序排列的）重叠区间两端。
最终清单按涉及的轮廓/边索引稳定排序。整个计算只读、即时完成：
不依赖已保存的规范化结果，也不写库。
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from typing import Any, NamedTuple

from .svggeo import Contour

# 参数域容差：t/u 距端点 1e-9 以内视为端点接触，吸收细分浮点误差
_EPS_PAR = 1e-9
# 角域（无量纲）容差：|cross|/(|u||v|) <= 该值视为平行，即 sin(夹角) 阈值
_EPS_ANGLE = 1e-9
# 线域（无量纲）相对容差：判共线/点落边时的垂直距离阈值 = 该值 ×
# 对应长度基准（共线取较短边、点落边取被落边）
_EPS_LINE = 1e-9
# 包围盒保守外扩（无量纲相对值）：每条边的 AABB 各轴外扩 该值 × 边长。
# 粗筛必须覆盖精判容差所能接受的“最近距离”：
#   非平行分支 |t|,|s| 可放宽到 _EPS_PAR，两端点越界合计 ≤ 2·_EPS_PAR·长边；
#   平行/共线分支垂直间距 ≤ _EPS_LINE·较短边，纵向参数越界 ≤ _EPS_PAR·长边；
# 角域阈值 _EPS_ANGLE 另带来 O(_EPS_ANGLE) 量级的投影偏移。取 4e-9 >
# 2·_EPS_PAR + _EPS_LINE + _EPS_ANGLE = 3e-9（含 1e-9 余量吸收浮点舍入），
# 使近端点接触 / 近共线重叠等包围盒本不相交的边对不被粗筛漏掉。
_BBOX_PAD_REL = 4e-9
# 输出量化位数（仅影响 JSON 展示，不改变判定与合并键）
_DECIMALS = 9

# 类型在输出中的固定排序（同一点位的不同类型不再合并，类型仅参与次序）
_TYPE_ORDER = {"crossing": 0, "endpoint_contact": 1, "collinear_overlap": 2}


def find_intersections(contours: list[Contour]) -> dict[str, Any]:
    """对已分类的轮廓列表生成加工前轮廓交叉清单。

    参数:
        contours: :func:`app.svggeo.analyze_contours` 返回的轮廓
            （已带 ``role``）。

    返回:
        ``{checked_contour_count, intersection_count, intersections,
        skipped_contours}``；``intersections`` 每项给出 ``type``、
        交点坐标（``point``；共线重叠为区间两端 ``points``）以及涉及的
        轮廓与边索引（``contours[].edge_indices``，边 k 为细分顶点
        ``points[k] -> points[(k+1) mod n]``，含闭合环的回首边）。
    """
    closed = [c for c in contours if c.closed]
    closed.sort(key=lambda c: c.original_index)

    skipped = [
        {"original_index": c.original_index, "role": c.role,
         "reason": ("open" if c.role == "open" else "degenerate")}
        for c in sorted(contours, key=lambda c: c.original_index)
        if not c.closed
    ]

    # 合并键 -> 聚合条目（同坐标同类型交点合并其涉及边索引）
    merged: dict[tuple, dict[str, Any]] = {}

    def add(kind: str, key: tuple, coord: Any,
            contour_idx: int, edge: int) -> None:
        item = merged.get(key)
        if item is None:
            item = {"kind": kind, "coord": coord,
                    "groups": {contour_idx: set()}}
            merged[key] = item
        item["groups"].setdefault(contour_idx, set()).add(edge)

    def classify_pair(e1: _Edge, e2: _Edge) -> None:
        """粗筛候选交精判；同环非相邻边 / 跨环全部候选均在此分类。

        调用方已按旧遍历次序（自交 (轮廓, i<j)；互交 (小轮廓, 大轮廓, i,
        j)）规范化参数顺序——精判在其首条边上插值交点，固定次序可保证
        大坐标下输出坐标逐位一致。
        """
        for kind, coord in _classify(e1.a, e1.b, e2.a, e2.b):
            key = _merge_key(kind, coord)
            add(kind, key, coord, e1.contour_idx, e1.edge)
            add(kind, key, coord, e2.contour_idx, e2.edge)

    # 粗筛：全部闭合环的细分边统一做 AABB x 扫掠 + y 区间过滤；
    # 同环仅保留非相邻边对，跨环候选全部交给精判。
    edges: list[_Edge] = []
    ring_sizes: dict[int, int] = {}
    for c in closed:
        n = len(c.points)
        ring_sizes[c.original_index] = n
        for k in range(n):
            edges.append(_make_edge(c.original_index, c.points, k, n))

    # 候选按旧遍历次序（环内自交在前：(轮廓, i<j)；跨环互交在后：
    # (较小轮廓, 较大轮廓, i, j)）排序后再精判——合并条目的代表坐标取
    # 首次插入该合并键的边对插值结果，固定次序可保证大坐标下输出与
    # 逐位一致（同一点由不同边对插值可有 ULP 级差异）。
    ordered: list[tuple[tuple, _Edge, _Edge]] = []
    for ea, eb in _candidate_pairs(edges):
        if ea.contour_idx == eb.contour_idx:
            if _adjacent(ea.edge, eb.edge,
                         ring_sizes[ea.contour_idx]):
                continue  # 相邻边仅共享端点，属正常连接
            e1, e2 = ((ea, eb) if ea.edge < eb.edge else (eb, ea))
            order = (0, e1.contour_idx, e1.edge, e2.edge)
        else:
            e1, e2 = ((ea, eb) if ea.contour_idx < eb.contour_idx
                      else (eb, ea))
            order = (1, e1.contour_idx, e2.contour_idx,
                     e1.edge, e2.edge)
        ordered.append((order, e1, e2))
    ordered.sort(key=lambda t: t[0])

    for _order, e1, e2 in ordered:
        classify_pair(e1, e2)

    role_by_index = {c.original_index: c.role for c in contours}
    intersections = [
        _entry(item, role_by_index)
        for item in sorted(merged.values(),
                           key=lambda it: _sort_key(it, role_by_index))
    ]

    return {
        "checked_contour_count": len(closed),
        "intersection_count": len(intersections),
        "intersections": intersections,
        "skipped_contours": skipped,
    }


# ---------------------------------------------------------------------------
# 粗筛：边包围盒（AABB）x 扫掠 + y 区间过滤
# ---------------------------------------------------------------------------

class _Edge(NamedTuple):
    contour_idx: int          # 所属轮廓的 original_index
    edge: int                 # 边索引（points[edge] -> points[(edge+1) mod n]）
    a: tuple[float, float]
    b: tuple[float, float]
    x0: float                 # 外扩后包围盒
    y0: float
    x1: float
    y1: float


def _make_edge(contour_idx: int,
               pts: list[tuple[float, float]],
               k: int, n: int) -> _Edge:
    """细分边 k 的保守 AABB：各轴按 :data:`_BBOX_PAD_REL` × 边长外扩。"""
    a, b = pts[k], pts[(k + 1) % n]
    ax, ay = a
    bx, by = b
    minx, maxx = (ax, bx) if ax <= bx else (bx, ax)
    miny, maxy = (ay, by) if ay <= by else (by, ay)
    pad = _BBOX_PAD_REL * math.hypot(bx - ax, by - ay)
    return _Edge(contour_idx, k, a, b,
                 minx - pad, miny - pad, maxx + pad, maxy + pad)


def _adjacent(i: int, j: int, n: int) -> bool:
    """同一闭合环上的边 i、j 是否相邻（共享一个细分顶点）。"""
    return (i - j) % n == 1 or (j - i) % n == 1


def _candidate_pairs(edges: list[_Edge]):
    """枚举外扩 AABB 相交的边对。

    x 轴事件扫掠：每条边在 x0 处进入活动集、x1 处退出（同一 x 先进后出，
    使仅在端点相切的包围盒也同处活动集）；进入时仅与活动集中 **y 区间
    重叠** 的边配对，因此候选对的 x、y 区间均相交（含边界）。
    """
    # (x, kind, edge_index)：kind 0=进入、1=退出；进入先于退出（同一 x
    # 时进入事件在前），使仅在端点相切的包围盒也同处活动集；edge_index
    # 唯一且单调，顺序确定。
    events = [(e.x0, 0, idx) for idx, e in enumerate(edges)]
    events += [(e.x1, 1, idx) for idx, e in enumerate(edges)]
    events.sort()

    # 活动集按 (y0, 边序号) 排序；y1 仅用于区间重叠判断
    active: list[tuple[float, int]] = []
    for _x, kind, idx in events:
        e = edges[idx]
        if kind == 1:
            active.pop(bisect_left(active, (e.y0, idx)))
            continue
        # y0_other <= e.y1 的活动边才可能 y 区间重叠
        stop = bisect_left(active, (e.y1, math.inf))
        for _y0, other in active[:stop]:
            if edges[other].y1 >= e.y0:  # 另一侧 y 区间须搭接到 e.y0
                yield edges[other], e
        insort(active, (e.y0, idx))


# ---------------------------------------------------------------------------
# 线段对分类
# ---------------------------------------------------------------------------

def _classify(a, b, c, d):
    """分类线段 ab 与 cd 的关系。

    返回 ``(kind, coord)`` 元组的列表（一对线段至多一条），
    ``coord`` 为交点元组（共线重叠为两端点的元组）；无交点返回空列表。
    """
    ux, uy = b[0] - a[0], b[1] - a[1]
    vx, vy = d[0] - c[0], d[1] - c[1]
    wx, wy = a[0] - c[0], a[1] - c[1]
    uu = ux * ux + uy * uy
    vv = vx * vx + vy * vy
    lu, lv = math.sqrt(uu), math.sqrt(vv)
    # 角域容差无量纲（sin 夹角阈值）；线域容差也是无量纲相对值，由各判定
    # 乘上与自身量纲一致的长度基准（共线按较短边、点落边按被落边）。
    # 二者只依赖边向量（与原点位置无关），故同一构型整体平移或等比放大后
    # 判定不变：长度比 1e9 的长边遇短边正交相交不会再被吞进“平行”分支。
    sin_tol = _EPS_ANGLE
    line_rel = _EPS_LINE

    # 零长度边：退化为点是否落在另一条边上；距离容差按被落边长度缩放
    if uu == 0.0:
        return [("endpoint_contact", (a,))] if _point_on_segment(c, d, a, line_rel) else []
    if vv == 0.0:
        return [("endpoint_contact", (c,))] if _point_on_segment(a, b, c, line_rel) else []

    denom = ux * vy - uy * vx
    if abs(denom) > sin_tol * lu * lv:
        # 非平行：解 a + t·u = c + s·v（w = a - c）
        t = (wy * vx - wx * vy) / denom
        s = (wy * ux - wx * uy) / denom
        if t < -_EPS_PAR or t > 1 + _EPS_PAR or s < -_EPS_PAR or s > 1 + _EPS_PAR:
            return []
        tc = min(1.0, max(0.0, t))
        point = (a[0] + tc * ux, a[1] + tc * uy)
        at_end = (t <= _EPS_PAR or t >= 1 - _EPS_PAR
                  or s <= _EPS_PAR or s >= 1 - _EPS_PAR)
        kind = "endpoint_contact" if at_end else "crossing"
        return [(kind, (point,))]

    # 平行：垂直距离 |u×w|/|u| 超过相对容差 × 较短边长度则不共线
    if abs(ux * wy - uy * wx) > line_rel * lu * min(lu, lv):
        return []

    # 共线：沿 u 方向参数化（q = a + s·u），比较两线段的参数区间
    def param(p):
        return ((p[0] - a[0]) * ux + (p[1] - a[1]) * uy) / uu

    s3, s4 = param(c), param(d)
    lo = max(0.0, min(s3, s4))
    hi = min(1.0, max(s3, s4))
    if hi < lo - _EPS_PAR:
        return []  # 共线但不重叠
    if hi - lo <= _EPS_PAR:
        t = min(1.0, max(0.0, lo))
        return [("endpoint_contact", ((a[0] + t * ux, a[1] + t * uy),))]

    p_lo = (a[0] + lo * ux, a[1] + lo * uy)
    p_hi = (a[0] + hi * ux, a[1] + hi * uy)
    if (p_hi[0], p_hi[1]) < (p_lo[0], p_lo[1]):
        p_lo, p_hi = p_hi, p_lo  # 端点按字典序排列
    return [("collinear_overlap", (p_lo, p_hi))]


def _point_on_segment(a, b, p, line_rel) -> bool:
    """点 p 是否在线段 ab 上（含端点）。

    ``line_rel`` 为无量纲相对容差：垂直距离阈值 ``line_rel·|ab|``，
    沿轴包围盒的纵向余量同口径（长度量纲），故等比放大后判定不变。
    """
    ux, uy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(ux, uy)
    dist_eps = line_rel * length
    cross = ux * (p[1] - a[1]) - uy * (p[0] - a[0])
    if abs(cross) > dist_eps * length:
        return False
    return (min(a[0], b[0]) - dist_eps <= p[0] <= max(a[0], b[0]) + dist_eps
            and min(a[1], b[1]) - dist_eps <= p[1] <= max(a[1], b[1]) + dist_eps)


# ---------------------------------------------------------------------------
# 合并 / 输出
# ---------------------------------------------------------------------------

def _merge_key(kind: str, coord: tuple) -> tuple:
    """同坐标同类型交点的合并键（相对量化，吸收表示层浮点噪声）。"""
    if kind == "collinear_overlap":
        p1, p2 = coord
        return (kind, _qm(p1), _qm(p2))
    return (kind, _qm(coord[0]))


def _q(p: tuple[float, float]) -> tuple[float, float]:
    """显示量化：绝对 9 位小数量化（仅影响 JSON 展示）。"""
    qx, qy = round(p[0], _DECIMALS), round(p[1], _DECIMALS)
    return (qx + 0.0 if qx == 0 else qx,
            qy + 0.0 if qy == 0 else qy)


def _qm(p: tuple[float, float]) -> tuple[float, float]:
    """合并键量化：按坐标量级取**相对**精度（量纲一致）。

    同一几何交点由不同边对独立插值或由顶点字面量给出时，在大坐标
    （如 1e13 处 ULP≈5e-4）下绝对 1e-9 量化无法吸收其差，会把同一
    交点拆成多条；按相对 1e-9（即 ``_EPS_PAR``，与参数域容差同口径）
    取分格则与位置、尺度无关：整体平移或等比放大后合并结果一致。
    """
    def one(v: float) -> float:
        if v == 0.0:
            return 0.0
        step = 10 ** math.floor(math.log10(abs(v))) * _EPS_PAR
        return round(v / step) * step
    return (one(p[0]), one(p[1]))


def _pt(p: tuple[float, float]) -> dict[str, float]:
    qx, qy = _q(p)
    return {"x": qx, "y": qy}


def _sort_key(item: dict[str, Any], role_by_index: dict[int, str]) -> tuple:
    """按涉及的（轮廓索引, 最小边索引）序列稳定排序，坐标/类型兜底。"""
    groups = sorted(
        (idx, min(edges)) for idx, edges in item["groups"].items())
    coord = item["coord"]
    if item["kind"] == "collinear_overlap":
        tail = (_q(coord[0]) + _q(coord[1]))
    else:
        tail = _q(coord[0])
    return (tuple(groups), _TYPE_ORDER[item["kind"]], tail)


def _entry(item: dict[str, Any],
           role_by_index: dict[int, str]) -> dict[str, Any]:
    kind, coord = item["kind"], item["coord"]
    groups = [
        {"original_index": idx,
         "role": role_by_index.get(idx, "unknown"),
         "edge_indices": sorted(item["groups"][idx])}
        for idx in sorted(item["groups"])
    ]
    if kind == "collinear_overlap":
        return {"type": kind, "points": [_pt(coord[0]), _pt(coord[1])],
                "contours": groups}
    return {"type": kind, "point": _pt(coord[0]), "contours": groups}
