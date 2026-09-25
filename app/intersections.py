"""轮廓交叉清单（intersections，只读即时检查）。

对一份已存 SVG path 即时解析（复用 :func:`app.svggeo.analyze_contours`
的解析与曲线细分折线，见 :func:`app.svggeo.flatten_subpath`），在
**加工前**对每条**闭合子路径**做两类边对检查：

- **自交**：同一闭合环内所有**非相邻边**对（相邻边仅共享端点，
  属正常连接，跳过；闭合环含末点回首点的闭合边）；
- **互交**：不同闭合子路径之间的全部边对。

开放轮廓（未以 ``Z`` 闭合）不参与（列入 ``skipped_contours``）；
闭合但退化的子路径（面积为 0 或顶点不足）仍按字面意义参与边检查。

交点分三类：

- ``crossing``：两线段在内部真正交叉（交点严格位于两条边内部）；
- ``endpoint_contact``：端点接触（某条边的端点落在另一条边上，
  含环内非相邻边共享端点的“收尖/捏合”构型、T 型接触与零长度边点接触）；
- ``collinear_overlap``：共线重叠（两条边共线且共享一段非零长度
  区间；仅共享一点时归为 ``endpoint_contact``）。

**边对搜索（扫掠宽相）**：先为每条细分边建立轴对齐包围盒（AABB），
按交点容差**保守外扩**（见 :func:`_candidate_pairs`），再沿 x 轴扫掠、
以 y 区间重叠过滤，只把包围盒外扩后仍可能接触的边对交给
:func:`_classify` 精确分类——稀疏、彼此远离的大路径只需极少次精确
分类，而密集重叠（包围盒大面积交叠）时全部边对仍是候选，清单完整。
同环候选仍只取**非相邻边**对，跨环候选全部保留；宽相只裁剪边对、
不改变判定，故三类交点的合并键、边索引与输出次序均与全枚举一致。

**同坐标、同类型**的交点合并为一条，合并其涉及的全部（轮廓, 边索引）；
``collinear_overlap`` 的合并键为（字典序排列的）重叠区间两端。
最终清单按涉及的轮廓/边索引稳定排序。整个计算只读、即时完成：
不依赖已保存的规范化结果，也不写库。
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Iterator

from .svggeo import Contour

# 参数域容差：t/u 距端点 1e-9 以内视为端点接触，吸收细分浮点误差
_EPS_PAR = 1e-9
# 角域（无量纲）容差：|cross|/(|u||v|) <= 该值视为平行，即 sin(夹角) 阈值
_EPS_ANGLE = 1e-9
# 线域（无量纲）相对容差：判共线/点落边时的垂直距离阈值 = 该值 ×
# 对应长度基准（共线取较短边、点落边取被落边）
_EPS_LINE = 1e-9
# 宽相包围盒的单边相对外扩量：pad = 该值 × 边长（零长度边为 0）。
# 取 4·(_EPS_PAR + _EPS_LINE) 的对称并集上界并留 4× 浮点余量——
# 凡 _classify 可能报告的边对，其（未外扩的）包围盒在任一坐标轴上的
# 间隙都不超过 pad_i + pad_j（详见 _candidate_pairs 的证明注释），
# 故近端点接触与近共线重叠绝不会被宽相漏裁。
_BBOX_PAD_REL = 4.0 * (_EPS_PAR + _EPS_LINE)
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

    # ---- 宽相：AABB 沿 x 扫掠 + y 区间过滤，只留下可能接触的边对 ----------
    # 同环候选仍只取非相邻边对；跨环候选全部保留。每对边至多分类一次，
    # 且分类结果与边对枚举次序无关（交点随后统一合并、排序）。
    for r, s in _candidate_pairs(closed):
        cr, er = r.contour, r.edge
        cs, es = s.contour, s.edge
        for kind, coord in _classify(r.a, r.b, s.a, s.b):
            key = _merge_key(kind, coord)
            add(kind, key, coord, cr.original_index, er)
            add(kind, key, coord, cs.original_index, es)

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
# 宽相：保守外扩的轴对齐包围盒（AABB）+ x 扫掠 / y 区间过滤
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Edge:
    """一条细分边（闭合环含末点回首点的闭合边）及其保守外扩 AABB。"""
    contour: Contour
    edge: int
    a: tuple[float, float]
    b: tuple[float, float]
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def _build_edges(contours: list[Contour]) -> list[_Edge]:
    """为全部闭合轮廓的每条细分边建立外扩后的轴对齐包围盒。

    外扩量 ``pad = _BBOX_PAD_REL·边长``（零长度边为 0，外扩盒退化为点）。
    """
    edges: list[_Edge] = []
    for c in contours:
        pts, n = c.points, len(c.points)
        for k in range(n):
            a, b = pts[k], pts[(k + 1) % n]
            pad = _BBOX_PAD_REL * math.hypot(b[0] - a[0], b[1] - a[1])
            edges.append(_Edge(
                contour=c, edge=k, a=a, b=b,
                xmin=min(a[0], b[0]) - pad,
                ymin=min(a[1], b[1]) - pad,
                xmax=max(a[0], b[0]) + pad,
                ymax=max(a[1], b[1]) + pad,
            ))
    return edges


def _adjacent(r: _Edge, s: _Edge) -> bool:
    """同一闭合环上的两条边是否相邻（共享一个顶点；边 0 与边 n-1 相邻）。

    仅在两条边属于同一轮廓时有意义；跨轮廓恒为 False。
    """
    if r.contour is not s.contour:
        return False
    n = len(r.contour.points)
    i, j = r.edge, s.edge
    return ((i + 1) % n == j) or ((j + 1) % n == i)


def _candidate_pairs(contours: list[Contour]) -> Iterator[tuple[_Edge, _Edge]]:
    """枚举包围盒保守外扩后仍可能接触的边对（每对至多一次）。

    薄封装：先 :func:`_build_edges`，再交给 :func:`_sweep_pairs`。
    沿 x 轴扫掠、以 y 区间过滤的算法与保守性（为何外扩绝不会漏掉
    _classify 报告的边对）见 :func:`_sweep_pairs`。
    """
    edges = _build_edges(contours)
    yield from _sweep_pairs(edges)


def _sweep_pairs(edges: list[_Edge]) -> Iterator[tuple[_Edge, _Edge]]:
    """在已建立的外扩 AABB 边序列上做 x 扫掠 / y 过滤的候选枚举。

    沿 x 轴扫掠：边按 ``xmin`` 依次进入活动集，``xmax < 当前 xmin``
    （严格小于才过期；相等时包围盒在 x 上相切，仍须交给精确分类）的活动边
    惰性过期。活动集始终按 ``(ymin, 序号)`` 有序：当前边按 ymin 定位插入点，
    交叠候选分两侧枚举——**前驱**（ymin ≤ 当前 ymin）从插入点向前回扫，
    一旦某活动边的 ``ymin + 全局最大 y 跨度 < 当前 ymin`` 即安全停止
    （再往前的边 ymin 更小，其包围盒向上无论如何都够不到当前边下沿，
    停止界取全部边跨度的最大值，长短边混排也保守）；**后继**
    （ymin > 当前 ymin）从插入点向后扫描，``ymin > 当前 ymax`` 后停止。
    同环相邻边对在此跳过（与原全枚举的跳过规则一致），跨环候选全保留。

    **保守性证明**（为何外扩绝不会漏掉 _classify 报告的边对）：

    - 非平行分支：报告交点时两条边的参数均落在
      ``[-_EPS_PAR, 1+_EPS_PAR]``，故交点到每条边本体的纵向距离
      ≤ ``_EPS_PAR·边长``；交点又是两延长线的公共点（垂直方向无分离），
      报告点的越界只可能沿边方向——任一坐标轴上的原始包围盒间隙单边
      不超过 ``_EPS_PAR·边长``。
    - 平行/共线分支：不报告时垂直间距 > ``_EPS_LINE·min(|u|,|v|)``；
      报告（端点接触/共线重叠）时两线段最近点的距离由
      “垂直 ≤ _EPS_LINE·min(|u|,|v|) + 纵向 ≤ _EPS_PAR·max(|u|,|v|)”
      夹逼，分解到任一坐标轴、再按对称并集摊到单边，均不超过
      ``(_EPS_PAR + _EPS_LINE)·边长``。
    - 零长度边：退化为点落边判定，其距离容差 = ``_EPS_LINE·被落边长``；
      点边自身外扩为 0，被落边单侧按上式外扩已足够。

    因此任一被报告边对在任一坐标轴上的原始 AABB 间隙
    ``gap_axis ≤ pad_i + pad_j``。取 ``pad = _BBOX_PAD_REL·边长``
    （= ``4·(_EPS_PAR+_EPS_LINE)·边长``，4× 系数把上述相对上界的
    舍入余量一并吸收）后两个外扩盒必相交；外扩盒不相交的边对交给
    _classify 也必为空，裁剪安全。``_EPS_ANGLE`` 只决定非平行分支的
    入口，不引入位置容差。
    """
    if len(edges) < 2:
        return

    order = sorted(range(len(edges)),
                   key=lambda h: (edges[h].xmin, h))
    # 活动边序号（边 id），按 (ymin, id) 有序；过期边惰性保留、定期压实
    active: list[int] = []
    present: set[int] = set()
    # (xmax, id) 最小堆：用于识别最早过期位置（压实移除后残留项弹出即弃）
    end_heap: list[tuple[float, int]] = []
    dead: set[int] = set()
    # 全部已进入边的 y 跨度（外扩后）最大值：回扫的安全停止界
    span_max = 0.0

    for hid in order:
        cur = edges[hid]

        while end_heap and end_heap[0][0] < cur.xmin:
            expired = heapq.heappop(end_heap)[1]
            if expired in present:  # 压实已移除的残留堆项直接丢弃
                dead.add(expired)
        if dead and len(active) >= 2 * (len(active) - len(dead)):
            active = [h for h in active if h not in dead]
            present = set(active)
            dead.clear()

        # 按 (ymin, id) 定位并插入活动集；交叠候选分两侧枚举：
        #   前驱（ymin ≤ 当前 ymin）——只能靠其 ymax 向上覆盖到当前下沿；
        #   后继（ymin > 当前 ymin）——ymin 一旦超过当前 ymax 即不可能交叠。
        # 前驱侧用全局最大 y 跨度做停止界：ymin 再小的边，若
        # other.ymin + span_max 仍够不到当前 ymin，则更前面（ymin 更小）
        # 的边也不可能交叠。两侧都跳过已过期边与同环相邻边。
        key = (cur.ymin, hid)
        ins = bisect_left(active, key, key=lambda h: (edges[h].ymin, h))

        for pos in range(ins - 1, -1, -1):
            oh = active[pos]
            other = edges[oh]
            if other.ymin + span_max < cur.ymin:
                break
            if oh in dead or other.ymax < cur.ymin:
                continue
            if not _adjacent(cur, other):
                yield cur, other

        for pos in range(ins, len(active)):
            oh = active[pos]
            other = edges[oh]
            if other.ymin > cur.ymax:
                break
            if oh in dead or other.ymax < cur.ymin:
                continue
            if not _adjacent(cur, other):
                yield cur, other

        active.insert(ins, hid)
        present.add(hid)
        heapq.heappush(end_heap, (cur.xmax, hid))
        span = cur.ymax - cur.ymin
        if span > span_max:
            span_max = span


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
