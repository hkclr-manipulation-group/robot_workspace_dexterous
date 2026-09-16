"""GPU FK, static triangle-BVH intersection, and closed-STL containment."""
import warp as wp

TRIANGLE_LANES = 128
CONTACT_TOLERANCE = 1.0e-6


@wp.func
def separates(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, e: wp.vec3,
              f: wp.vec3, axis: wp.vec3) -> bool:
    length = wp.length(axis)
    if length < 1.0e-15:
        return False
    pa, pb, pc = wp.dot(a, axis), wp.dot(b, axis), wp.dot(c, axis)
    pd, pe, pf = wp.dot(d, axis), wp.dot(e, axis), wp.dot(f, axis)
    margin = CONTACT_TOLERANCE * length
    return (wp.min(pa, wp.min(pb, pc)) > wp.max(pd, wp.max(pe, pf)) + margin or
            wp.min(pd, wp.min(pe, pf)) > wp.max(pa, wp.max(pb, pc)) + margin)


@wp.func
def triangles_intersect(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3,
                        b0: wp.vec3, b1: wp.vec3, b2: wp.vec3) -> bool:
    # Translate before projecting to reduce cancellation. Normalized face axes
    # keep the tolerance in metres, including very small CAD triangles.
    a = wp.vec3(0.0)
    b, c, d, e, f = a1-a0, a2-a0, b0-a0, b1-a0, b2-a0
    na = wp.normalize(wp.cross(b, c))
    nb = wp.normalize(wp.cross(e-d, f-d))
    if separates(a, b, c, d, e, f, na) or separates(a, b, c, d, e, f, nb):
        return False
    for i in range(3):
        ea = b-a
        eb = e-d
        if i == 1:
            ea = c-b
            eb = f-e
        elif i == 2:
            ea = a-c
            eb = d-f
        # In-plane edge axes also handle coplanar triangles and touching edges.
        if separates(a, b, c, d, e, f, wp.cross(na, ea)):
            return False
        if separates(a, b, c, d, e, f, wp.cross(nb, eb)):
            return False
        for j in range(3):
            other = e-d
            if j == 1:
                other = f-e
            elif j == 2:
                other = d-f
            if separates(a, b, c, d, e, f, wp.cross(ea, other)):
                return False
    return True


@wp.kernel(enable_backward=False)
def forward_kinematics(
    q: wp.array2d(dtype=float), active: wp.array(dtype=int),
    parents: wp.array(dtype=int), origins: wp.array(dtype=wp.transform),
    joint_indices: wp.array(dtype=int), joint_types: wp.array(dtype=int),
    axes: wp.array(dtype=wp.vec3), multipliers: wp.array(dtype=float),
    offsets: wp.array(dtype=float), transforms: wp.array2d(dtype=wp.transform),
    colliding: wp.array(dtype=int),
):
    batch = wp.tid()
    colliding[batch] = 0
    if active[batch] == 0:
        colliding[batch] = 1
        return
    for j in range(q.shape[1]):
        if not wp.isfinite(q[batch, j]):
            colliding[batch] = 1
            return
    for link in range(parents.shape[0]):
        pose = origins[link]
        if joint_types[link] != 0:
            value = q[batch, joint_indices[link]] * multipliers[link] + offsets[link]
            motion = wp.transform_identity()
            if joint_types[link] == 2:
                motion = wp.transform(axes[link] * value, wp.quat_identity())
            else:
                motion = wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(axes[link], value))
            pose = wp.transform_multiply(pose, motion)
        if parents[link] >= 0:
            pose = wp.transform_multiply(transforms[batch, parents[link]], pose)
        transforms[batch, link] = pose


@wp.kernel(enable_backward=False)
def world_bounds(
    transforms: wp.array2d(dtype=wp.transform), part_links: wp.array(dtype=int),
    local_lo: wp.array(dtype=wp.vec3), local_hi: wp.array(dtype=wp.vec3),
    colliding: wp.array(dtype=int), lows: wp.array2d(dtype=wp.vec3),
    highs: wp.array2d(dtype=wp.vec3),
):
    batch, part = wp.tid()
    if colliding[batch] != 0:
        return
    pose = transforms[batch, part_links[part]]
    lo, hi = wp.vec3(1.0e30), wp.vec3(-1.0e30)
    for corner in range(8):
        point = local_lo[part]
        for axis in range(3):
            if (corner & (1 << axis)) != 0:
                point[axis] = local_hi[part][axis]
        world = wp.transform_point(pose, point)
        lo, hi = wp.min(lo, world), wp.max(hi, world)
    lows[batch, part] = lo - wp.vec3(CONTACT_TOLERANCE)
    highs[batch, part] = hi + wp.vec3(CONTACT_TOLERANCE)


@wp.func
def boxes_overlap(lo_a: wp.vec3, hi_a: wp.vec3, lo_b: wp.vec3, hi_b: wp.vec3) -> bool:
    return (lo_a[0] <= hi_b[0] and lo_b[0] <= hi_a[0] and
            lo_a[1] <= hi_b[1] and lo_b[1] <= hi_a[1] and
            lo_a[2] <= hi_b[2] and lo_b[2] <= hi_a[2])


@wp.kernel(enable_backward=False)
def intersect_mesh_pairs(
    transforms: wp.array2d(dtype=wp.transform), part_links: wp.array(dtype=int),
    mesh_ids: wp.array(dtype=wp.uint64), triangle_counts: wp.array(dtype=int),
    pairs: wp.array2d(dtype=int), lows: wp.array2d(dtype=wp.vec3),
    highs: wp.array2d(dtype=wp.vec3), colliding: wp.array(dtype=int),
):
    batch, pair, lane = wp.tid()
    if colliding[batch] != 0:
        return
    a, b = pairs[pair, 0], pairs[pair, 1]
    if not boxes_overlap(lows[batch, a], highs[batch, a], lows[batch, b], highs[batch, b]):
        return
    relative = wp.transform_multiply(wp.transform_inverse(transforms[batch, part_links[b]]),
                                     transforms[batch, part_links[a]])
    mesh_a, mesh_b = mesh_ids[a], mesh_ids[b]
    for face in range(lane, triangle_counts[a], TRIANGLE_LANES):
        if colliding[batch] != 0:
            return
        v0 = wp.transform_point(relative, wp.mesh_get_point(mesh_a, wp.mesh_get_index(mesh_a, 3*face)))
        v1 = wp.transform_point(relative, wp.mesh_get_point(mesh_a, wp.mesh_get_index(mesh_a, 3*face+1)))
        v2 = wp.transform_point(relative, wp.mesh_get_point(mesh_a, wp.mesh_get_index(mesh_a, 3*face+2)))
        lo = wp.min(v0, wp.min(v1, v2)) - wp.vec3(CONTACT_TOLERANCE)
        hi = wp.max(v0, wp.max(v1, v2)) + wp.vec3(CONTACT_TOLERANCE)
        query = wp.mesh_query_aabb(mesh_b, lo, hi)
        other = int(0)
        while wp.mesh_query_aabb_next(query, other):
            u0 = wp.mesh_get_point(mesh_b, wp.mesh_get_index(mesh_b, 3*other))
            u1 = wp.mesh_get_point(mesh_b, wp.mesh_get_index(mesh_b, 3*other+1))
            u2 = wp.mesh_get_point(mesh_b, wp.mesh_get_index(mesh_b, 3*other+2))
            if triangles_intersect(v0, v1, v2, u0, u1, u2):
                wp.atomic_max(colliding, batch, 1)
                return


@wp.kernel(enable_backward=False)
def contained_mesh_pairs(
    transforms: wp.array2d(dtype=wp.transform), part_links: wp.array(dtype=int),
    mesh_ids: wp.array(dtype=wp.uint64), closed: wp.array(dtype=int),
    representatives: wp.array(dtype=wp.vec3), representative_offsets: wp.array(dtype=int),
    pairs: wp.array2d(dtype=int), lows: wp.array2d(dtype=wp.vec3),
    highs: wp.array2d(dtype=wp.vec3), colliding: wp.array(dtype=int),
):
    batch, pair = wp.tid()
    if colliding[batch] != 0:
        return
    a, b = pairs[pair, 0], pairs[pair, 1]
    if not boxes_overlap(lows[batch, a], highs[batch, a], lows[batch, b], highs[batch, b]):
        return
    # Surface intersections have already been checked. One representative per
    # connected source component detects complete containment in either direction.
    for direction in range(2):
        source = int(a)
        target = int(b)
        if direction == 1:
            source = int(b)
            target = int(a)
        if closed[target] != 0:
            relative = wp.transform_multiply(
                wp.transform_inverse(transforms[batch, part_links[target]]),
                transforms[batch, part_links[source]])
            for k in range(representative_offsets[source], representative_offsets[source+1]):
                point = wp.transform_point(relative, representatives[k])
                query = wp.mesh_query_point(mesh_ids[target], point, 1.0e6)
                if query.result and query.sign < 0.0:
                    wp.atomic_max(colliding, batch, 1)
                    return
