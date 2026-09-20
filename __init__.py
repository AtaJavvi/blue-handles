# -*- coding: utf-8 -*-
bl_info = {
    "name": "Blue Handles",
    "author": "Ata_Javvi",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "Edit Mode: Ctrl+Shift+Alt+D / Tool Header BH",
    "description": "Blue Handles — Vertex + Spine mesh deform with Bezier handles (Edit Mode)",
    "category": "Mesh",
}

import bpy
import bmesh
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from bpy.types import Operator, AddonPreferences
from bpy.props import IntProperty, StringProperty, BoolProperty, EnumProperty
from mathutils import Vector, Matrix, Quaternion
from mathutils.geometry import intersect_point_line
from mathutils.kdtree import KDTree
from bpy_extras import view3d_utils
import math
import json
import time

# Global smooth mode for R / Shift+R popup (while modal may be running)
VDH_SMOOTH_MODE = 'LAPLACIAN'
VDH_SMOOTH_ITEMS = (
    ('LAPLACIAN', 'Laplacian', 'Classic neighbor average (can shrink a bit)'),
    ('TAUBIN', 'Taubin', 'Smooth with less volume loss'),
    ('TANGENTIAL', 'Tangential', 'Smooth along the surface, better volume keep'),
)
_ATTR_INTERP_ORDER = ('SMOOTH', 'SPHERE', 'LINEAR', 'SHARP', 'CONSTANT')
_active_vdh_op = None  # running modal operator instance
# Per-mesh spine recall / bind-rest keyed by mesh datablock pointer
# (NOT object name — deleting Object "Cylinder" and creating a new one must not revive old curves)
_vdh_spine_recall = {}
_vdh_spine_bind_rest = {}


def _vdh_cache_key(obj):
    """Stable RAM key for this mesh datablock (unique per mesh instance in session)."""
    if obj is None or getattr(obj, 'type', None) != 'MESH' or obj.data is None:
        return None
    try:
        return int(obj.data.as_pointer())
    except Exception:
        return obj.name_full


def _vdh_set_recall(obj, data):
    global _vdh_spine_recall
    key = _vdh_cache_key(obj)
    if key is None or not data:
        return
    _vdh_spine_recall[key] = data


def _vdh_set_bind_rest(obj, data):
    global _vdh_spine_bind_rest
    key = _vdh_cache_key(obj)
    if key is None or not data:
        return
    _vdh_spine_bind_rest[key] = data


# Mesh custom-property keys (persist inside .blend)
_VDH_RECALL_PROP = "vdh_spine_recall_json"
_VDH_BIND_REST_PROP = "vdh_spine_bind_rest_json"
_VDH_VERTEX_SCALE_PROP = "vdh_vertex_display_scale"
_vdh_vertex_scale = {}


def _vdh_get_vertex_display_scale(obj, default=1.0):
    key = _vdh_cache_key(obj)
    if key is not None and key in _vdh_vertex_scale:
        try:
            return float(_vdh_vertex_scale[key])
        except Exception:
            pass
    try:
        if obj is not None and obj.type == 'MESH' and obj.data is not None:
            if _VDH_VERTEX_SCALE_PROP in obj.data:
                return float(obj.data[_VDH_VERTEX_SCALE_PROP])
    except Exception:
        pass
    return float(default)


def _vdh_set_vertex_display_scale(obj, value):
    val = max(0.05, min(5.0, float(value or 1.0)))
    key = _vdh_cache_key(obj)
    if key is not None:
        _vdh_vertex_scale[key] = val
    try:
        if obj is not None and obj.type == 'MESH' and obj.data is not None:
            obj.data[_VDH_VERTEX_SCALE_PROP] = val
    except Exception:
        pass
    return val


def _vdh_vec3(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return [float(v[0]), float(v[1]), float(v[2])]
    try:
        return [float(v.x), float(v.y), float(v.z)]
    except Exception:
        return None


def _vdh_from_vec3(v):
    if v is None:
        return None
    try:
        return Vector((float(v[0]), float(v[1]), float(v[2])))
    except Exception:
        return None


def _vdh_serialize_bez(pts):
    out = []
    for bp in (pts or []):
        if not isinstance(bp, dict):
            continue
        out.append({
            'co': _vdh_vec3(bp.get('co')),
            'hl': _vdh_vec3(bp.get('hl')),
            'hr': _vdh_vec3(bp.get('hr')),
        })
    return out


def _vdh_deserialize_bez(pts):
    out = []
    for bp in (pts or []):
        if not isinstance(bp, dict):
            continue
        co = _vdh_from_vec3(bp.get('co')) or Vector()
        hl = _vdh_from_vec3(bp.get('hl')) or co.copy()
        hr = _vdh_from_vec3(bp.get('hr')) or co.copy()
        out.append({'co': co, 'hl': hl, 'hr': hr})
    return out


def _vdh_serialize_bind(bind):
    out = []
    for item in (bind or []):
        try:
            vidx = int(item[0])
            t = float(item[1])
            off = _vdh_vec3(item[2]) if len(item) > 2 else None
            tan = _vdh_vec3(item[3]) if len(item) > 3 else None
            dist = float(item[4]) if len(item) > 4 else 0.0
            out.append([vidx, t, off, tan, dist])
        except Exception:
            continue
    return out


def _vdh_deserialize_bind(bind):
    out = []
    for item in (bind or []):
        try:
            vidx = int(item[0])
            t = float(item[1])
            off = _vdh_from_vec3(item[2]) if len(item) > 2 else None
            tan = _vdh_from_vec3(item[3]) if len(item) > 3 else None
            dist = float(item[4]) if len(item) > 4 else 0.0
            out.append((vidx, t, off, tan, dist))
        except Exception:
            continue
    return out


def _vdh_serialize_recall_entry(data):
    """Convert in-memory recall dict to JSON-safe structure."""
    if not data:
        return None
    chains_out = []
    for ch in (data.get('chains') or []):
        chains_out.append({
            'bez': _vdh_serialize_bez(ch.get('bez')),
            'rest_bez': _vdh_serialize_bez(ch.get('rest_bez')),
            'modes': list(ch.get('modes') or []),
            'tilt': [float(x) for x in (ch.get('tilt') or [])],
            'radius': [float(x) for x in (ch.get('radius') or [])],
            'handle_params': [float(x) for x in (ch.get('handle_params') or [])],
            'bind': _vdh_serialize_bind(ch.get('bind')),
            'influence': float(ch.get('influence') or 0.1),
            'point_influence': [float(x) for x in (ch.get('point_influence') or [])],
            'point_influence_default': [float(x) for x in (ch.get('point_influence_default') or [])],
            'point_inf_falloff': list(ch.get('point_inf_falloff') or []),
            # Persist the Shrink/Inflate/Tilt interpolation falloff in the
            # .blend JSON recall payload. Without this, Constant/Sphere/etc.
            # exists only in RAM and is lost when the tool is re-created.
            'attr_interp': (str(ch.get('attr_interp') or 'SMOOTH').upper()
                            if str(ch.get('attr_interp') or 'SMOOTH').upper() in _ATTR_INTERP_ORDER
                            else 'SMOOTH'),
            'origin_ids': list(ch.get('origin_ids') or []),
            'chain_id': ch.get('chain_id'),
            'in_front': bool(ch.get('in_front', True)),
            'vg_name': ch.get('vg_name') or '',
        })
    mesh_snap = {}
    for k, v in (data.get('mesh_snap') or {}).items():
        mesh_snap[str(int(k))] = _vdh_vec3(v)
    rest_snap = {}
    for k, v in (data.get('rest_snap') or {}).items():
        rest_snap[str(int(k))] = _vdh_vec3(v)
    all_rest_snap = {}
    for k, v in (data.get('all_rest_snap') or {}).items():
        all_rest_snap[str(int(k))] = _vdh_vec3(v)
    out = {
        'chains': chains_out,
        'active_chain': int(data.get('active_chain', 0) or 0),
        'display_scale': float(data.get('display_scale', 1.0) or 1.0),
        'vert_count': int(data.get('vert_count', -1)),
        'mesh_snap': mesh_snap,
        'rest_snap': rest_snap,
        'all_rest_snap': all_rest_snap,
        'attr_layers': {
            'tilt': {str(int(k)): _vdh_vec3(v) for k, v in (data.get('attr_layers', {}).get('tilt', {}) or {}).items()},
            'inflate': {str(int(k)): _vdh_vec3(v) for k, v in (data.get('attr_layers', {}).get('inflate', {}) or {}).items()},
        },
    }
    # One previous confirm snapshot so Blender Undo can restore spine pose.
    prev = data.get('prev')
    if prev and prev is not data:
        try:
            prev_copy = dict(prev)
            prev_copy.pop('prev', None)
            out['prev'] = _vdh_serialize_recall_entry(prev_copy)
        except Exception:
            pass
    return out



def _vdh_deserialize_recall_entry(data):
    if not data:
        return None
    chains = []
    for ch in (data.get('chains') or []):
        chains.append({
            'bez': _vdh_deserialize_bez(ch.get('bez')),
            'rest_bez': _vdh_deserialize_bez(ch.get('rest_bez')),
            'modes': list(ch.get('modes') or []),
            'tilt': [float(x) for x in (ch.get('tilt') or [])],
            'radius': [float(x) for x in (ch.get('radius') or [])],
            'handle_params': [float(x) for x in (ch.get('handle_params') or [])],
            'bind': _vdh_deserialize_bind(ch.get('bind')),
            'influence': float(ch.get('influence') or 0.1),
            'point_influence': [float(x) for x in (ch.get('point_influence') or [])],
            'point_influence_default': [float(x) for x in (ch.get('point_influence_default') or [])],
            'point_inf_falloff': list(ch.get('point_inf_falloff') or []),
            'attr_interp': (str(ch.get('attr_interp') or 'SMOOTH').upper()
                            if str(ch.get('attr_interp') or 'SMOOTH').upper() in _ATTR_INTERP_ORDER
                            else 'SMOOTH'),
            'origin_ids': list(ch.get('origin_ids') or []),
            'chain_id': ch.get('chain_id'),
            'in_front': bool(ch.get('in_front', True)),
            'vg_name': ch.get('vg_name') or '',
        })
    mesh_snap = {}
    for k, v in (data.get('mesh_snap') or {}).items():
        vec = _vdh_from_vec3(v)
        if vec is not None:
            mesh_snap[int(k)] = vec
    rest_snap = {}
    for k, v in (data.get('rest_snap') or {}).items():
        vec = _vdh_from_vec3(v)
        if vec is not None:
            rest_snap[int(k)] = vec
    all_rest_snap = {}
    for k, v in (data.get('all_rest_snap') or {}).items():
        vec = _vdh_from_vec3(v)
        if vec is not None:
            all_rest_snap[int(k)] = vec
    attr_layers_raw = data.get('attr_layers') or {}
    attr_layers = {'tilt': {}, 'inflate': {}}
    for _kind in ('tilt', 'inflate'):
        for k, v in (attr_layers_raw.get(_kind) or {}).items():
            vec = _vdh_from_vec3(v)
            if vec is not None:
                attr_layers[_kind][int(k)] = vec
    out = {
        'chains': chains,
        'active_chain': int(data.get('active_chain', 0) or 0),
        'display_scale': float(data.get('display_scale', 1.0) or 1.0),
        'vert_count': int(data.get('vert_count', -1)),
        'mesh_snap': mesh_snap,
        'rest_snap': rest_snap,
        'all_rest_snap': all_rest_snap,
        'attr_layers': attr_layers,
    }
    prev = data.get('prev')
    if prev:
        try:
            if isinstance(prev, dict):
                prev_copy = dict(prev)
                prev_copy.pop('prev', None)
                out['prev'] = _vdh_deserialize_recall_entry(prev_copy)
        except Exception:
            pass
    return out


def _vdh_recall_snap_error(bm, mesh_snap):
    """Average distance of saved mesh_snap to current edit verts. None if unusable."""
    if not bm or not mesh_snap:
        return None
    try:
        bm.verts.ensure_lookup_table()
    except Exception:
        return None
    err = 0.0
    n = 0
    nverts = len(bm.verts)
    for vidx, co in mesh_snap.items():
        try:
            i = int(vidx)
        except Exception:
            continue
        if i < 0 or i >= nverts or co is None:
            continue
        try:
            err += (bm.verts[i].co - co).length
        except Exception:
            continue
        n += 1
    if n <= 0:
        return None
    return err / float(n)


def _vdh_iter_recall_candidates(*entries):
    """Yield unique recall dicts, including one-level prev snapshots."""
    seen = set()
    for data in entries:
        if not data or not isinstance(data, dict):
            continue
        cur = data
        for _ in range(2):
            if not cur:
                break
            marker = id(cur)
            if marker not in seen:
                seen.add(marker)
                yield cur
            cur = cur.get('prev') if isinstance(cur.get('prev'), dict) else None


def _vdh_write_mesh_json(mesh, prop_name, obj_data):
    """Write JSON string custom property on mesh datablock (saved in .blend)."""
    if mesh is None:
        return
    try:
        mesh[prop_name] = json.dumps(obj_data, separators=(',', ':'))
    except Exception:
        pass


def _vdh_read_mesh_json(mesh, prop_name):
    if mesh is None:
        return None
    try:
        raw = mesh.get(prop_name)
        if not raw:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode('utf-8')
        return json.loads(raw)
    except Exception:
        return None


def _vdh_persist_recall_to_mesh(obj, data):
    if obj is None or obj.type != 'MESH' or not data:
        return
    ser = _vdh_serialize_recall_entry(data)
    if ser:
        _vdh_write_mesh_json(obj.data, _VDH_RECALL_PROP, ser)


def _vdh_load_recall_from_mesh(obj):
    if obj is None or obj.type != 'MESH':
        return None
    raw = _vdh_read_mesh_json(obj.data, _VDH_RECALL_PROP)
    return _vdh_deserialize_recall_entry(raw) if raw else None


def _vdh_persist_bind_rest_to_mesh(obj, data):
    if obj is None or obj.type != 'MESH' or not data:
        return
    try:
        chains_out = []
        for ch in (data.get('chains') or []):
            chains_out.append({
                'chain_id': ch.get('chain_id'),
                'bez': _vdh_serialize_bez(ch.get('bez')),
                'tilt': [float(x) for x in (ch.get('tilt') or [])],
                'radius': [float(x) for x in (ch.get('radius') or [])],
                'modes': list(ch.get('modes') or []),
                'origin_ids': list(ch.get('origin_ids') or []),
                'influence': float(ch.get('influence') or 0.0),
                'point_influence': [float(x) for x in (ch.get('point_influence') or [])],
                'point_inf_falloff': list(ch.get('point_inf_falloff') or []),
                'bind_verts': [int(x) for x in (ch.get('bind_verts') or [])],
                'vg_name': ch.get('vg_name') or '',
            })
        mesh_rest = {}
        for k, v in (data.get('mesh_rest') or {}).items():
            mesh_rest[str(int(k))] = _vdh_vec3(v)
        payload = {'chains': chains_out, 'mesh_rest': mesh_rest}
        _vdh_write_mesh_json(obj.data, _VDH_BIND_REST_PROP, payload)
    except Exception:
        pass


def _vdh_load_bind_rest_from_mesh(obj):
    if obj is None or obj.type != 'MESH':
        return None
    raw = _vdh_read_mesh_json(obj.data, _VDH_BIND_REST_PROP)
    if not raw:
        return None
    try:
        chains = []
        for ch in (raw.get('chains') or []):
            chains.append({
                'chain_id': ch.get('chain_id'),
                'bez': _vdh_deserialize_bez(ch.get('bez')),
                'tilt': [float(x) for x in (ch.get('tilt') or [])],
                'radius': [float(x) for x in (ch.get('radius') or [])],
                'modes': list(ch.get('modes') or []),
                'origin_ids': list(ch.get('origin_ids') or []),
                'influence': float(ch.get('influence') or 0.0),
                'point_influence': [float(x) for x in (ch.get('point_influence') or [])],
                'point_inf_falloff': list(ch.get('point_inf_falloff') or []),
                'bind_verts': [int(x) for x in (ch.get('bind_verts') or [])],
                'vg_name': ch.get('vg_name') or '',
            })
        mesh_rest = {}
        for k, v in (raw.get('mesh_rest') or {}).items():
            vec = _vdh_from_vec3(v)
            if vec is not None:
                mesh_rest[int(k)] = vec
        return {'chains': chains, 'mesh_rest': mesh_rest}
    except Exception:
        return None


def _vdh_get_bind_rest(obj):
    """RAM cache, else load from mesh custom property."""
    global _vdh_spine_bind_rest
    if obj is None:
        return {'chains': [], 'mesh_rest': {}}
    key = _vdh_cache_key(obj)
    if key is not None:
        data = _vdh_spine_bind_rest.get(key)
        if data:
            return data
    data = _vdh_load_bind_rest_from_mesh(obj)
    if data and key is not None:
        _vdh_spine_bind_rest[key] = data
        return data
    return {'chains': [], 'mesh_rest': {}}


class MESH_OT_vdh_smooth_choice(Operator):
    """Choose smooth algorithm for Relax (R)"""
    bl_idname = "mesh.vdh_smooth_choice"
    bl_label = "VDH Smooth Type"
    bl_options = {'INTERNAL'}

    mode: StringProperty(default='LAPLACIAN')

    def execute(self, context):
        global VDH_SMOOTH_MODE
        VDH_SMOOTH_MODE = self.mode
        op = _active_vdh_op
        if op is not None:
            op.smooth_mode = self.mode
            if getattr(op, 'tool_mode', '') == 'SPINE_DEFORM':
                op._spine_smooth(context)
            else:
                op.relax_selection(context)
        self.report({'INFO'}, f"Smooth: {self.mode.title()}")
        return {'FINISHED'}



class MESH_OT_vdh_clear_cache(Operator):
    """Clear Blue Handles spine cache (recall + bind rest)"""
    bl_idname = "mesh.vdh_clear_cache"
    bl_label = "Clear Spine Cache"
    bl_options = {'INTERNAL', 'UNDO'}

    scope: StringProperty(default='THIS')  # THIS | ALL

    def execute(self, context):
        global _vdh_spine_recall, _vdh_spine_bind_rest, _active_vdh_op
        if self.scope == 'ALL':
            n_r = len(_vdh_spine_recall)
            n_b = len(_vdh_spine_bind_rest)
            _vdh_spine_recall.clear()
            _vdh_spine_bind_rest.clear()
            # Also strip mesh custom props on all meshes
            for me in bpy.data.meshes:
                for prop in (_VDH_RECALL_PROP, _VDH_BIND_REST_PROP):
                    if prop in me:
                        try:
                            del me[prop]
                        except Exception:
                            pass
            n_vg = 0
            for obj in bpy.data.objects:
                n_vg += _vdh_remove_spine_vertex_groups(obj)
            msg = f"Spine cache cleared (all: {max(n_r, n_b)}, groups: {n_vg})"
        else:
            obj = context.object
            if obj is None:
                self.report({'WARNING'}, "No active object")
                return {'CANCELLED'}
            key = _vdh_cache_key(obj)
            had = False
            if key is not None:
                had = key in _vdh_spine_recall or key in _vdh_spine_bind_rest
                if key in _vdh_spine_recall:
                    del _vdh_spine_recall[key]
                if key in _vdh_spine_bind_rest:
                    del _vdh_spine_bind_rest[key]
            if obj.type == 'MESH' and obj.data:
                for prop in (_VDH_RECALL_PROP, _VDH_BIND_REST_PROP):
                    if prop in obj.data:
                        try:
                            del obj.data[prop]
                            had = True
                        except Exception:
                            pass
            n_vg = _vdh_remove_spine_vertex_groups(obj)
            if n_vg:
                had = True
            msg = f"Spine cache cleared: {obj.name}" if had else f"No cache for: {obj.name}"
            if n_vg:
                msg += f"  |  removed {n_vg} vertex group(s)"

        try:
            obj = context.object
            if obj is not None and obj.type == 'MESH' and obj.mode == 'EDIT':
                bm = bmesh.from_edit_mesh(obj.data)
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                obj.data.update()
        except Exception:
            pass

        op = _active_vdh_op
        if op is not None and getattr(op, 'tool_mode', '') == 'SPINE_PLACE':
            try:
                op._spine_session_full_reset(context)
                msg += " | chains wiped — start from scratch"
            except Exception as e:
                msg += f" | session reset failed: {e}"
        self.report({'INFO'}, msg)
        return {'FINISHED'}



class MESH_OT_vdh_mirror_chain(Operator):
    """Mirror active spine chain curve (axis + space)"""
    bl_idname = "mesh.vdh_mirror_chain"
    bl_label = "Mirror Spine Chain"
    bl_options = {'INTERNAL'}

    axis: StringProperty(default='X')
    space: StringProperty(default='LOCAL')  # LOCAL | CURSOR | WORLD
    duplicate: BoolProperty(default=False)  # True = Ctrl+Shift+M style

    def execute(self, context):
        op = _active_vdh_op
        if op is None:
            self.report({'WARNING'}, "Tool not active")
            return {'CANCELLED'}
        ok = op._spine_mirror_active_chain(
            context, axis=self.axis, space=self.space, duplicate=bool(self.duplicate),
            event=getattr(op, '_last_modal_event', None),
        )
        return {'FINISHED'} if ok else {'CANCELLED'}


class MESH_OT_vdh_handle_type(Operator):
    """Set handle type for selected controllers (like Curve V menu)"""
    bl_idname = "mesh.vdh_handle_type"
    bl_label = "VDH Handle Type"
    bl_options = {'INTERNAL'}

    mode: StringProperty(default='AUTO')

    def execute(self, context):
        op = _active_vdh_op
        if op is None:
            return {'CANCELLED'}

        tm = getattr(op, 'tool_mode', 'VERTEX')
        chains = getattr(op, 'spine_chains', None) or []

        # Spine-only: operate directly on the stored chain data.  Do NOT
        # temporarily swap op.bez/point_modes while iterating chains; that
        # caused multi-chain selections to collapse back onto the active chain
        # in some sessions.  The selection is the source of truth here.
        if tm in ('SPINE_DEFORM', 'SPINE_PLACE') and chains:
            try:
                raw_selected = list(getattr(op, 'selected', None) or set())
            except Exception:
                raw_selected = []

            targets = set()
            ac = int(getattr(op, 'active_chain', 0) or 0)
            for k in raw_selected:
                try:
                    if len(k) == 3:
                        ci, i, part = int(k[0]), int(k[1]), k[2]
                    elif len(k) == 2:
                        # Legacy single-chain selection keys.
                        ci, i, part = ac, int(k[0]), k[1]
                    else:
                        continue
                    if part == 'co':
                        targets.add((ci, i))
                except Exception:
                    continue

            if not targets and getattr(op, 'active_handle', None) is not None:
                targets.add((ac, int(op.active_handle)))

            if not targets:
                self.report({'INFO'}, "Select a controller first")
                return {'CANCELLED'}

            changed = 0
            for ci, i in sorted(targets):
                if not (0 <= ci < len(chains)):
                    continue
                ch = chains[ci]
                bez = ch.get('bez') or []
                n = len(bez)
                if not (0 <= i < n):
                    continue

                modes = list(ch.get('modes') or ['AUTO'] * n)
                if len(modes) != n:
                    modes = (modes + ['AUTO'] * n)[:n]

                modes[i] = self.mode
                ch['modes'] = modes

                # AUTO uses the existing Blender-style automatic tangent
                # builder.  ALIGNED keeps each handle's existing lengths but
                # forces both sides onto one tangent line.  FREE leaves the
                # current handle positions untouched.
                if self.mode == 'AUTO':
                    old_bez = getattr(op, 'bez', None)
                    old_modes = getattr(op, 'point_modes', None)
                    try:
                        op.bez = bez
                        op.point_modes = modes
                        op.rebuild_auto_handles()
                        ch['bez'] = op.bez
                    finally:
                        op.bez = old_bez
                        op.point_modes = old_modes
                elif self.mode == 'ALIGNED' and 0 < i < n - 1:
                    bp = bez[i]
                    left = bp['co'] - bp['hl']
                    right = bp['hr'] - bp['co']
                    if right.length > 1e-8:
                        tdir = right.normalized()
                    elif left.length > 1e-8:
                        tdir = left.normalized()
                    else:
                        tdir = None
                    if tdir is not None:
                        L_l = left.length if left.length > 1e-8 else right.length
                        L_r = right.length if right.length > 1e-8 else left.length
                        bp['hl'] = bp['co'] - tdir * L_l
                        bp['hr'] = bp['co'] + tdir * L_r

                changed += 1

            # Refresh the active-chain runtime references only once, after all
            # selected chains have been modified.
            op.spine_chains = chains
            try:
                op._spine_load_active_chain()
            except Exception:
                pass
            try:
                op._spine_store_active_chain()
            except Exception:
                pass
            if tm == 'SPINE_DEFORM':
                try:
                    op._spine_apply(context)
                except Exception:
                    pass
            context.area.tag_redraw()
            self.report({'INFO'}, f"Handle type: {self.mode} ({changed} point(s), multi-chain)")
            return {'FINISHED'}

        # Vertex Mode stores handle types in point_modes (not in the
        # bez-point dictionaries).  Keep this path independent from Spine.
        if tm == 'VERTEX':
            pts = getattr(op, 'bez', None) or []
            try:
                sel = set(op.selected_point_indices()) if hasattr(op, 'selected_point_indices') else set()
            except Exception:
                sel = set()
            if not sel and getattr(op, 'active_handle', None) is not None:
                sel = {int(op.active_handle)}
            if not sel:
                self.report({'INFO'}, "Select a controller first")
                return {'CANCELLED'}

            modes = list(getattr(op, 'point_modes', None) or [])
            if len(modes) != len(pts):
                modes = (modes + ['AUTO'] * len(pts))[:len(pts)]
            if len(modes) < len(pts):
                modes += ['AUTO'] * (len(pts) - len(modes))

            for i in sel:
                if not (0 <= i < len(pts)):
                    continue
                modes[i] = self.mode
                if self.mode == 'ALIGNED':
                    bp = pts[i]
                    left = bp['co'] - bp['hl']
                    right = bp['hr'] - bp['co']
                    if right.length > 1e-8:
                        tdir = right.normalized()
                    elif left.length > 1e-8:
                        tdir = left.normalized()
                    else:
                        tdir = None
                    if tdir is not None:
                        L_l = left.length if left.length > 1e-8 else right.length
                        L_r = right.length if right.length > 1e-8 else left.length
                        bp['hl'] = bp['co'] - tdir * L_l
                        bp['hr'] = bp['co'] + tdir * L_r

            op.point_modes = modes
            if self.mode == 'AUTO' and hasattr(op, 'rebuild_auto_handles'):
                try:
                    op.rebuild_auto_handles()
                    # Handle-type changes in Vertex Mode must immediately
                    # evaluate the mesh from the new AUTO handle positions.
                    # Without this, the mesh waits for the next controller drag.
                    op.apply_deform(context)
                    op._vdh_refresh_edit_normals(context)
                except Exception:
                    pass
            context.area.tag_redraw()
            self.report({'INFO'}, f"Handle type: {self.mode}")
            return {'FINISHED'}

        return {'CANCELLED'}


class MESH_OT_vdh_influence_falloff(Operator):
    """Set influence falloff type for selected controllers"""
    bl_idname = "mesh.vdh_influence_falloff"
    bl_label = "VDH Influence Falloff"
    bl_options = {'INTERNAL'}

    mode: StringProperty(default='CONSTANT')

    def execute(self, context):
        op = _active_vdh_op
        if op is None:
            self.report({'WARNING'}, "Tool not active")
            return {'CANCELLED'}
        ok = op._spine_set_influence_falloff(context, self.mode)
        return {'FINISHED'} if ok else {'CANCELLED'}


class MESH_OT_vdh_straighten_tube(Operator):
    """Straighten Tube along chosen axis"""
    bl_idname = "mesh.vdh_straighten_tube"
    bl_label = "Straighten Tube"
    bl_options = {'INTERNAL'}

    axis: StringProperty(default='FREE')  # FREE | X | Y | Z
    active_only: BoolProperty(default=True)
    circularize: BoolProperty(default=True)
    even_spacing: BoolProperty(default=True)

    def execute(self, context):
        op = _active_vdh_op
        if op is None:
            self.report({'WARNING'}, "Tool not active")
            return {'CANCELLED'}
        if getattr(op, 'tool_mode', '') != 'SPINE_DEFORM':
            self.report({'INFO'}, "Straighten Tube: use in Spine Deform")
            return {'CANCELLED'}
        ok = op._spine_straighten_tube(
            context,
            circularize=bool(self.circularize),
            even_spacing=bool(self.even_spacing),
            active_only=bool(self.active_only),
            axis=str(self.axis or 'FREE').upper(),
        )
        return {'FINISHED'} if ok else {'CANCELLED'}



class MESH_OT_vdh_set_initial(Operator):
    """Set active chain current pose as Ctrl+R initial state"""
    bl_idname = "mesh.vdh_set_initial"
    bl_label = "Set as Initial State"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        op = _active_vdh_op
        if op is None:
            self.report({'WARNING'}, "Tool not active")
            return {'CANCELLED'}
        ok = op._spine_set_as_initial(context)
        return {'FINISHED'} if ok else {'CANCELLED'}


class VDH_Preferences(AddonPreferences):
    bl_idname = __name__

    prefs_tab: EnumProperty(
        name="Tab",
        items=(
            ('ABOUT', "About", "Overview"),
            ('KEYMAPS', "Keymaps", "All shortcuts"),
        ),
        default='ABOUT',
    )

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.prop(self, "prefs_tab", expand=True)
        if self.prefs_tab == 'KEYMAPS':
            self._draw_keymaps_tab(layout)
        else:
            self._draw_about_tab(layout)

    def _km(self, col, key, action):
        row = col.row(align=True)
        split = row.split(factor=0.44)
        split.label(text=key)
        split.label(text=action)

    def _draw_about_tab(self, layout):
        box = layout.box()
        box.label(text="Blue Handles", icon='CURVE_BEZCURVE')
        col = box.column(align=True)
        col.label(text="Mesh deformation tool. Start with Ctrl+Shift+Alt+D or the BH button.")

        box = layout.box()
        box.label(text="Vertex Mode", icon='VERTEXSEL')
        col = box.column(align=True)
        col.label(text="Select 2 or more vertices, then start the tool.")
        col.label(text="A temporary Bezier curve controls the selected mesh.")

        box = layout.box()
        box.label(text="Spine Mode", icon='CURVE_PATH')
        col = box.column(align=True)
        col.label(text="Start with nothing selected. Click to place controllers, Enter to bind.")
        col.label(text="Each chain uses a BH_Spine vertex group for Weight Paint.")
        col.label(text="Reopening the tool restores the last spine on this mesh.")

        box = layout.box()
        box.label(text="Cache", icon='FILE_REFRESH')
        col = box.column(align=True)
        col.operator("mesh.vdh_clear_cache", text="Clear Cache (This Object)", icon='X').scope = 'THIS'
        col.operator("mesh.vdh_clear_cache", text="Clear Cache (All Meshes)", icon='TRASH').scope = 'ALL'

    def _draw_keymaps_tab(self, layout):
        box = layout.box()
        box.label(text="Common", icon='EVENT_D')
        col = box.column(align=True)
        self._km(col, "Ctrl + Shift + Alt + D", "Start tool")
        self._km(col, "BH button", "Start tool (Tool Header)")
        self._km(col, "Enter", "Confirm")
        self._km(col, "Esc", "Cancel")
        self._km(col, "[ / ]", "Resize controllers")

        box = layout.box()
        box.label(text="Vertex Mode", icon='VERTEXSEL')
        col = box.column(align=True)
        self._km(col, "Shift + Middle Mouse", "Add controller on curve")
        self._km(col, "Alt + Shift + Middle Mouse", "Remove controller")
        self._km(col, "V", "Handle type")
        self._km(col, "Shift + LMB", "Multi-select handles")
        self._km(col, "G / R / S", "Move / Rotate / Scale")
        self._km(col, "S / R / L / F", "Space / Relax / Line / Set Flow")
        self._km(col, "Shift + R", "Smooth type")
        self._km(col, "Scroll while dragging", "Proportional size")
        self._km(col, "Shift + Scroll", "Align selection to curve")
        self._km(col, "Alt + X + Mouse Direction", "Align handle to the mouse direction")

        box = layout.box()
        box.label(text="Spine — Place / Edit Place", icon='CURVE_PATH')
        col = box.column(align=True)
        self._km(col, "Click / Alt + Click", "Add / Remove controller")
        self._km(col, "Alt + Click handle tip", "Set handle type to Free")
        self._km(col, "Drag AUTO handle tip", "Convert AUTO to Aligned")
        self._km(col, "V", "Handle type")
        self._km(col, "Shift + Middle Mouse", "Insert controller on curve")
        self._km(col, "Shift + Enter", "Close chain, start next")
        self._km(col, "Enter", "Bind / Rebind all")
        self._km(col, "Alt + Enter", "Edit Place (from Deform)")
        self._km(col, "P", "Place in Volume")
        self._km(col, "N / Shift + N", "Toggle In Front / all chains")
        self._km(col, "Box select", "Select controllers")
        self._km(col, "Ctrl + Shift + Box", "Select handle tips")
        self._km(col, "Shift + D", "Duplicate chain")
        self._km(col, "Ctrl + M", "Mirror")
        self._km(col, "Ctrl + Shift + M", "Dup + Mirror")
        self._km(col, "M", "Set current pose as initial")
        self._km(col, "Ctrl + Alt + C", "Clear cache")

        box = layout.box()
        box.label(text="Spine — Deform", icon='MOD_CURVE')
        col = box.column(align=True)
        self._km(col, "G / R / S", "Move / Rotate / Scale")
        self._km(col, "Ctrl + T / Alt + T", "Tilt / Remove Tilt")
        self._km(col, "Ctrl + A / Ctrl + Alt + A", "Shrink / Inflate / Reset")
        self._km(col, "[ / ] during Tilt or Shrink", "Cycle falloff: Smooth / Sphere / Linear / Sharp / Constant")
        self._km(col, "Shift + [ / ] / Wheel", "Adjust influence radius")
        self._km(col, "Alt + [ / ] / Scroll", "Adjust influence marker size")
        self._km(col, "Shift + F", "Choose influence falloff")
        self._km(col, "I", "Show / hide weight overlay")
        self._km(col, "N / Shift + N", "Toggle In Front / all chains")
        self._km(col, "V", "Change handle type")
        self._km(col, "Ctrl + R", "Reset active chain")
        self._km(col, "Ctrl + Alt + R", "Reset all chains")
        self._km(col, "Alt + R", "Reset selected influence radius")
        self._km(col, "Shift + L", "Straighten tube")
        self._km(col, "W", "Align longitudinal edge loops to tube")
        self._km(col, "Q", "Circularize edge rings")
        self._km(col, "E", "Uniform tube thickness (average)")
        self._km(col, "Alt + X + Mouse Direction", "Align handle to the mouse direction")
        self._km(col, "Ctrl + Shift + L", "Straighten all chains")
        self._km(col, "Ctrl + M", "Mirror")
        self._km(col, "Alt + Enter", "Enter Edit Place")
        self._km(col, "Enter", "Confirm")
        self._km(col, "Esc", "Cancel")

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_selected_vert_indices(bm):
    return [v.index for v in bm.verts if v.select]


def order_verts_along_chain(bm, indices):
    """Order selected verts along an edge chain when possible."""
    if len(indices) < 2:
        return list(indices)

    idx_set = set(indices)
    neighbors = {i: [] for i in indices}
    for e in bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        if a in idx_set and b in idx_set:
            neighbors[a].append(b)
            neighbors[b].append(a)

    endpoints = [i for i in indices if len(neighbors[i]) <= 1]
    start = endpoints[0] if endpoints else indices[0]

    ordered = []
    visited = set()
    cur = start
    prev = None
    while cur is not None and cur not in visited:
        visited.add(cur)
        ordered.append(cur)
        nxts = [n for n in neighbors[cur] if n != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]

    # leftovers
    for i in indices:
        if i not in visited:
            ordered.append(i)
    return ordered


def sample_polyline(points, count):
    """Evenly sample count points along polyline (list of Vector)."""
    if count <= 1:
        return [points[0].copy()] if points else []
    if len(points) == 1:
        return [points[0].copy() for _ in range(count)]

    lengths = [0.0]
    for i in range(1, len(points)):
        lengths.append(lengths[-1] + (points[i] - points[i - 1]).length)
    total = lengths[-1]
    if total < 1e-12:
        return [points[0].copy() for _ in range(count)]

    result = []
    for s in range(count):
        t = 0.0 if count == 1 else s / (count - 1)
        target = t * total
        j = 0
        while j < len(lengths) - 1 and lengths[j + 1] < target:
            j += 1
        seg = lengths[j + 1] - lengths[j]
        u = 0.0 if seg < 1e-12 else (target - lengths[j]) / seg
        result.append(points[j].lerp(points[j + 1], u))
    return result


def polyline_at(points, t):
    """Point at normalized chord-length parameter t in [0,1] on polyline."""
    if not points:
        return Vector()
    if len(points) == 1 or t <= 0.0:
        return points[0].copy()
    if t >= 1.0:
        return points[-1].copy()
    lengths = [0.0]
    for i in range(1, len(points)):
        lengths.append(lengths[-1] + (points[i] - points[i - 1]).length)
    total = lengths[-1]
    if total < 1e-12:
        return points[0].copy()
    target = t * total
    j = 0
    while j < len(lengths) - 1 and lengths[j + 1] < target:
        j += 1
    seg = lengths[j + 1] - lengths[j]
    u = 0.0 if seg < 1e-12 else (target - lengths[j]) / seg
    return points[j].lerp(points[j + 1], u)


def deformed_curve_point(rest_poly, rest_handles, handles, t):
    """Selection shape + handle delta so blue line matches selection at rest."""
    base = polyline_at(rest_poly, t)
    delta = eval_curve(handles, t) - eval_curve(rest_handles, t)
    return base + delta





def bezier_segment_derivative(p0, h1, h2, p3, t):
    """Derivative of cubic Bezier at t (tangent direction, not unit)."""
    u = 1.0 - t
    return (
        3.0 * u * u * (h1 - p0)
        + 6.0 * u * t * (h2 - h1)
        + 3.0 * t * t * (p3 - h2)
    )


def bezier_chain_tangent(pts, t):
    """Exact unit tangent on Bezier point chain at global t in [0,1]."""
    n = len(pts)
    if n < 2:
        return Vector((1, 0, 0))
    if n == 2:
        d = bezier_segment_derivative(pts[0]['co'], pts[0]['hr'], pts[1]['hl'], pts[1]['co'], t)
        return d.normalized() if d.length > 1e-12 else Vector((1, 0, 0))
    seg_f = t * (n - 1)
    i = int(math.floor(seg_f))
    if i >= n - 1:
        i = n - 2
        local_t = 1.0
    else:
        local_t = seg_f - i
    d = bezier_segment_derivative(
        pts[i]['co'], pts[i]['hr'], pts[i + 1]['hl'], pts[i + 1]['co'], local_t
    )
    if d.length < 1e-12:
        # fallback finite difference
        return curve_tangent_at(pts, t)
    return d.normalized()

def curve_tangent_at(pts, t, eps=0.002):
    """Unit tangent of Bezier chain at parameter t."""
    t0 = max(0.0, t - eps)
    t1 = min(1.0, t + eps)
    d = eval_bezier_points(pts, t1) - eval_bezier_points(pts, t0)
    if d.length < 1e-12:
        # fallback larger step
        t0 = max(0.0, t - 0.02)
        t1 = min(1.0, t + 0.02)
        d = eval_bezier_points(pts, t1) - eval_bezier_points(pts, t0)
    if d.length < 1e-12:
        return Vector((1, 0, 0))
    return d.normalized()


def curve_curvature_length(pts, t, neighbor_dist):
    """Handle length from local curvature / neighbor spacing."""
    eps = 0.02
    t0 = max(0.0, t - eps)
    t1 = min(1.0, t + eps)
    p0 = eval_bezier_points(pts, t0)
    p1 = eval_bezier_points(pts, t)
    p2 = eval_bezier_points(pts, t1)
    v0 = p1 - p0
    v1 = p2 - p1
    if v0.length < 1e-12 or v1.length < 1e-12:
        return max(neighbor_dist * 0.3, 1e-4)
    v0n, v1n = v0.normalized(), v1.normalized()
    # turning: 0 straight, 2 sharp
    turn = 1.0 - max(-1.0, min(1.0, v0n.dot(v1n)))
    # sharper bend -> slightly shorter handles; straight -> longer
    factor = 0.33 - 0.12 * turn
    factor = max(0.15, min(0.4, factor))
    return max(neighbor_dist * factor, neighbor_dist * 0.12)


def nearest_param_on_polyline(points, point):
    """Chord-length parameter in [0,1] of closest location on polyline."""
    if not points:
        return 0.0
    if len(points) == 1:
        return 0.0
    lengths = [0.0]
    for i in range(1, len(points)):
        lengths.append(lengths[-1] + (points[i] - points[i - 1]).length)
    total = lengths[-1]
    if total < 1e-12:
        return 0.0

    best_d, best_u = 1e18, 0.0
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        ab = b - a
        lab2 = ab.length_squared
        if lab2 < 1e-16:
            u_seg, proj = 0.0, a
        else:
            u_seg = max(0.0, min(1.0, (point - a).dot(ab) / lab2))
            proj = a + ab * u_seg
        d = (point - proj).length_squared
        if d < best_d:
            best_d = d
            seg_len = lengths[i + 1] - lengths[i]
            best_u = (lengths[i] + seg_len * u_seg) / total
    return best_u


def deform_weights(u, handle_params, power=1.5):
    """Inverse-distance weights along the curve parameter.
    power~1.5 → smooth automatic falloff that adjusts as controllers move.
    """
    n = len(handle_params)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    raw = []
    for hp in handle_params:
        d = abs(u - hp)
        raw.append(1.0 / (pow(d, power) + 1e-8))
    s = sum(raw)
    if s < 1e-18:
        return [1.0 / n] * n
    return [r / s for r in raw]


def weighted_delta(u, handles, rest_handles, handle_params):
    """Normalized blend of handle translations at parameter u.
    handles/rest may be Bezier point dicts with 'co' or plain Vectors.
    """
    w = deform_weights(u, handle_params)
    delta = Vector((0, 0, 0))
    for j, wj in enumerate(w):
        hj = handles[j]['co'] if isinstance(handles[j], dict) else handles[j]
        rj = rest_handles[j]['co'] if isinstance(rest_handles[j], dict) else rest_handles[j]
        delta += (hj - rj) * wj
    return delta

def _perp_dist(point, a, b):
    """Distance from point to segment ab."""
    ab = b - a
    lab2 = ab.length_squared
    if lab2 < 1e-16:
        return (point - a).length
    t = max(0.0, min(1.0, (point - a).dot(ab) / lab2))
    return (point - (a + ab * t)).length


def douglas_peucker(points, epsilon):
    """Simplify polyline keeping shape within epsilon."""
    if len(points) < 3:
        return [p.copy() for p in points]

    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _perp_dist(points[i], points[0], points[-1])
        if d > dmax:
            dmax, idx = d, i

    if dmax > epsilon:
        left = douglas_peucker(points[: idx + 1], epsilon)
        right = douglas_peucker(points[idx:], epsilon)
        return left[:-1] + right
    return [points[0].copy(), points[-1].copy()]


def curvature_scores(points):
    """Turning-angle score per interior vertex (0..n-1)."""
    n = len(points)
    scores = [0.0] * n
    if n < 3:
        return scores
    for i in range(1, n - 1):
        v0 = points[i] - points[i - 1]
        v1 = points[i + 1] - points[i]
        if v0.length < 1e-12 or v1.length < 1e-12:
            continue
        v0.normalize()
        v1.normalize()
        # 0 = straight, 1 = sharp bend
        scores[i] = 1.0 - max(-1.0, min(1.0, v0.dot(v1)))
    return scores



def fit_circle_3d(points):
    """Fit a circle to 3D points (plane + 2D circle). Returns center, radius, rms_error, normal."""
    npts = len(points)
    if npts < 3:
        return None, 0.0, 1e9, None

    # centroid
    c = Vector((0, 0, 0))
    for p in points:
        c += p
    c /= float(npts)

    # covariance for plane normal (PCA)
    xx = xy = xz = yy = yz = zz = 0.0
    for p in points:
        d = p - c
        xx += d.x * d.x
        xy += d.x * d.y
        xz += d.x * d.z
        yy += d.y * d.y
        yz += d.y * d.z
        zz += d.z * d.z
    # normal ~ eigenvector of smallest eigenvalue (simple analytic for 3x3)
    # use cross of two edges as fallback normal if PCA-like fails
    cov = [
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz],
    ]
    # power iteration on adjugate-ish: find min eigenvector via cross products of rows
    r0 = Vector(cov[0])
    r1 = Vector(cov[1])
    r2 = Vector(cov[2])
    nrm = r0.cross(r1)
    if nrm.length < 1e-12:
        nrm = r0.cross(r2)
    if nrm.length < 1e-12:
        nrm = r1.cross(r2)
    if nrm.length < 1e-12:
        nrm = Vector((0, 0, 1))
    else:
        nrm.normalize()

    # build in-plane axes
    tmp = Vector((0, 0, 1)) if abs(nrm.z) < 0.9 else Vector((1, 0, 0))
    xu = nrm.cross(tmp)
    if xu.length < 1e-12:
        xu = Vector((1, 0, 0))
    xu.normalize()
    yu = nrm.cross(xu)

    # project to 2D
    pts2 = []
    for p in points:
        d = p - c
        pts2.append((d.dot(xu), d.dot(yu)))

    # algebraic circle fit: x^2 + y^2 + D x + E y + F = 0
    A = []
    b = []
    for x, y in pts2:
        A.append([x, y, 1.0])
        b.append(-(x * x + y * y))
    # solve least squares 3x3
    def mat_mul(M, v):
        return [
            M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
        ]
    # AtA and Atb
    AtA = [[0.0]*3 for _ in range(3)]
    Atb = [0.0]*3
    for row, bi in zip(A, b):
        for i in range(3):
            Atb[i] += row[i] * bi
            for j in range(3):
                AtA[i][j] += row[i] * row[j]
    # Cramer's rule / inverse 3x3
    def det3(m):
        return (
            m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
            - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
            + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
        )
    D = det3(AtA)
    if abs(D) < 1e-14:
        return None, 0.0, 1e9, None

    def replace_col(m, col, vec):
        out = [list(r) for r in m]
        for i in range(3):
            out[i][col] = vec[i]
        return out

    Dx = det3(replace_col(AtA, 0, Atb))
    Dy = det3(replace_col(AtA, 1, Atb))
    Dz = det3(replace_col(AtA, 2, Atb))
    Dco, Eco, Fco = Dx/D, Dy/D, Dz/D
    cx2 = -0.5 * Dco
    cy2 = -0.5 * Eco
    rad2 = cx2*cx2 + cy2*cy2 - Fco
    if rad2 <= 1e-12:
        return None, 0.0, 1e9, None
    radius = math.sqrt(rad2)
    center = c + xu * cx2 + yu * cy2

    # RMS radial error
    err = 0.0
    for p in points:
        err += abs((p - center).length - radius)
    err /= float(npts)
    return center, radius, err, nrm


def detect_circular_arc(points):
    """True if points lie on a circular arc; returns (ok, center, radius, normal)."""
    center, radius, err, nrm = fit_circle_3d(points)
    if center is None or radius < 1e-8:
        return False, None, 0.0, None
    if err > radius * 0.06:
        return False, None, 0.0, None
    # reject near-full ambiguous cases only if needed — semicircle is fine
    return True, center, radius, nrm


def arc_handles_3(points, center, normal):
    """Place start, angular-mid, end on the fitted arc."""
    nrm = normal.normalized()
    a0 = points[0] - center
    a1 = points[-1] - center
    # mid by half angle in plane
    # orthonormal basis in plane
    x_axis = a0.normalized()
    y_axis = nrm.cross(x_axis)
    if y_axis.length < 1e-8:
        y_axis = Vector((0, 1, 0))
    y_axis.normalize()

    def ang(v):
        return math.atan2(v.dot(y_axis), v.dot(x_axis))

    t0 = ang(a0)
    t1 = ang(a1)
    # choose shorter arc direction matching polyline mid
    mid_p = points[len(points)//2]
    tm = ang(mid_p - center)
    # unwrap t1 near path through tm
    def unwrap(a, base):
        while a - base > math.pi:
            a -= 2*math.pi
        while a - base < -math.pi:
            a += 2*math.pi
        return a
    tm = unwrap(tm, t0)
    t1 = unwrap(t1, tm)

    t_mid = 0.5 * (t0 + t1)
    r0 = a0.length
    r1 = (points[-1] - center).length
    rm = (r0 + r1) * 0.5

    def on_arc(t, r):
        return center + x_axis * (math.cos(t) * r) + y_axis * (math.sin(t) * r)

    return [points[0].copy(), on_arc(t_mid, rm), points[-1].copy()]


def auto_fit_handles(points, max_count=5, min_count=2):
    """
    Place controllers so the blue Bezier hugs the selection curvature.
    Strategy:
      1. Circular arc -> exactly 3 handles.
      2. Douglas-Peucker shape simplification (primary shape points).
      3. Curvature peaks (extra points on sharp bends).
      4. Arc-length fill if still under target.
    Controllers always sit on actual selection vertices.
    """
    if not points:
        return []
    if len(points) <= 2:
        return [p.copy() for p in points]

    ok, center, radius, nrm = detect_circular_arc(points)
    if ok:
        return arc_handles_3(points, center, nrm)

    bb_min = Vector(points[0])
    bb_max = Vector(points[0])
    for p in points:
        bb_min = Vector((min(bb_min.x, p.x), min(bb_min.y, p.y), min(bb_min.z, p.z)))
        bb_max = Vector((max(bb_max.x, p.x), max(bb_max.y, p.y), max(bb_max.z, p.z)))
    diag = (bb_max - bb_min).length
    if diag < 1e-12:
        return [points[0].copy(), points[-1].copy()]

    n = len(points)
    lengths = [0.0]
    for i in range(1, n):
        lengths.append(lengths[-1] + (points[i] - points[i - 1]).length)
    total_len = lengths[-1] if lengths[-1] > 1e-12 else 1.0

    scores = curvature_scores(points)
    total_turn = sum(scores)

    if total_turn < 0.12:
        target = 2
    elif total_turn < 0.30:
        target = 3
    elif total_turn < 0.55:
        target = 4
    elif total_turn < 0.90:
        target = 5
    elif total_turn < 1.40:
        target = 6
    elif total_turn < 2.00:
        target = 7
    else:
        target = min(10, max_count)
    target = max(int(min_count), min(int(max_count), target))

    avg_step = total_len / max(n - 1, 1)
    eps = max(diag * 0.012, avg_step * 0.35)
    simplified = douglas_peucker(points, eps)
    chosen_idx = {0, n - 1}
    for sp in simplified:
        best_j, best_d = 0, 1e18
        for j, p in enumerate(points):
            d = (p - sp).length_squared
            if d < best_d:
                best_d, best_j = d, j
        chosen_idx.add(best_j)

    if len(chosen_idx) > target:
        scored = [(scores[i], i) for i in chosen_idx if i not in (0, n - 1)]
        scored.sort(key=lambda x: -x[0])
        keep = {0, n - 1}
        for _sc, i in scored:
            if len(keep) >= target:
                break
            keep.add(i)
        chosen_idx = keep

    peaks = []
    for i in range(1, n - 1):
        if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1] and scores[i] > 0.025:
            peaks.append((scores[i], i))
    peaks.sort(key=lambda x: -x[0])

    min_sep = 0.055
    for _sc, i in peaks:
        if len(chosen_idx) >= target:
            break
        t_i = lengths[i] / total_len
        if any(abs(t_i - lengths[j] / total_len) < min_sep for j in chosen_idx):
            continue
        chosen_idx.add(i)

    if len(chosen_idx) < target:
        for k in range(target * 3):
            t = (k + 1) / (target * 3 + 1)
            best_j, best_d = 0, 1e18
            for j, L in enumerate(lengths):
                d = abs(L / total_len - t)
                if d < best_d:
                    best_d, best_j = d, j
            if any(abs(lengths[j] / total_len - lengths[best_j] / total_len) < min_sep for j in chosen_idx):
                continue
            chosen_idx.add(best_j)
            if len(chosen_idx) >= target:
                break

    ordered = sorted(chosen_idx)
    best = [points[i].copy() for i in ordered]
    best[0] = points[0].copy()
    best[-1] = points[-1].copy()

    cleaned = [best[0]]
    for p in best[1:]:
        if (p - cleaned[-1]).length > diag * 1e-4:
            cleaned.append(p)
    if len(cleaned) < 2:
        return sample_polyline(points, min(3, max_count))
    return cleaned

def catmull_rom(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )




def cubic_bezier(p0, h1, h2, p3, t):
    u = 1.0 - t
    return (u * u * u) * p0 + (3.0 * u * u * t) * h1 + (3.0 * u * t * t) * h2 + (t * t * t) * p3


def polyline_tangents(points):
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [Vector((1, 0, 0))]
    out = []
    for i in range(n):
        if i == 0:
            d = points[1] - points[0]
        elif i == n - 1:
            d = points[n - 1] - points[n - 2]
        else:
            d = points[i + 1] - points[i - 1]
        out.append(d.normalized() if d.length > 1e-12 else Vector((1, 0, 0)))
    return out


def arc_tangent_at(point, center, normal, toward):
    r = point - center
    if r.length < 1e-12:
        return toward.normalized() if toward.length > 1e-12 else Vector((1, 0, 0))
    t = normal.normalized().cross(r)
    if t.length < 1e-12:
        t = Vector((1, 0, 0))
    else:
        t.normalize()
    if toward.length > 1e-12 and t.dot(toward) < 0:
        t = -t
    return t


def segment_kappa_length(p0, p3, t0, t3):
    """Bezier handle length approximating circular arc between p0 and p3."""
    chord_v = p3 - p0
    chord = chord_v.length
    if chord < 1e-12:
        return 0.0
    cdir = chord_v.normalized()
    a0 = max(-1.0, min(1.0, t0.dot(cdir)))
    a1 = max(-1.0, min(1.0, t3.dot(cdir)))
    alpha = math.acos(a0) + math.acos(a1)
    alpha = max(1e-4, min(alpha, math.pi * 0.95))
    sin_half = math.sin(alpha * 0.5)
    if abs(sin_half) < 1e-8:
        return chord / 3.0
    R = chord / (2.0 * sin_half)
    L = (4.0 / 3.0) * math.tan(alpha * 0.25) * R
    return max(chord * 0.05, min(L, chord * 1.25))


def make_bezier_points(cos, poly=None, center=None, normal=None):
    """
    Build Blender-like Bezier points from on-curve positions.
    Each item: {'co', 'hl', 'hr'} in local space.
    Ends only get OUT handle toward the connected point (hr on start, hl on end).
    """
    n = len(cos)
    pts = [{'co': c.copy(), 'hl': c.copy(), 'hr': c.copy()} for c in cos]
    if n == 0:
        return pts
    if n == 1:
        return pts

    # unit tangents at each co
    tans = []
    for i, c in enumerate(cos):
        if center is not None and normal is not None:
            if i < n - 1:
                toward = cos[i + 1] - c
            else:
                toward = c - cos[i - 1]
            tans.append(arc_tangent_at(c, center, normal, toward))
        elif poly is not None:
            best_i, best_d = 0, 1e18
            for j, p in enumerate(poly):
                d = (p - c).length_squared
                if d < best_d:
                    best_d, best_i = d, j
            i0 = max(0, best_i - 2)
            i1 = min(len(poly) - 1, best_i + 2)
            d = poly[i1] - poly[i0]
            if d.length < 1e-12:
                pt = polyline_tangents(poly)
                d = pt[best_i]
            tans.append(d.normalized() if d.length > 1e-12 else Vector((1, 0, 0)))
        else:
            if i == 0:
                d = cos[1] - cos[0]
            elif i == n - 1:
                d = cos[n - 1] - cos[n - 2]
            else:
                d = cos[i + 1] - cos[i - 1]
            tans.append(d.normalized() if d.length > 1e-12 else Vector((1, 0, 0)))

    # force end outs along neighbor direction oriented with tangent
    d0 = cos[1] - cos[0]
    if d0.length > 1e-12:
        tans[0] = tans[0] if tans[0].dot(d0) >= 0 else -tans[0]
    d1 = cos[-1] - cos[-2]
    if d1.length > 1e-12:
        tans[-1] = tans[-1] if tans[-1].dot(d1) >= 0 else -tans[-1]

    for i in range(n - 1):
        p0, p3 = cos[i], cos[i + 1]
        t0, t3 = tans[i], tans[i + 1]
        # orient into segment
        cdir = p3 - p0
        if cdir.length > 1e-12:
            cdir.normalize()
            if t0.dot(cdir) < 0:
                t0 = -t0
            if t3.dot(cdir) < 0:
                t3 = -t3
        L = segment_kappa_length(p0, p3, t0, t3)
        # OUT from i toward i+1
        pts[i]['hr'] = p0 + t0 * L
        # OUT from i+1 toward i (left handle)
        pts[i + 1]['hl'] = p3 - t3 * L

    # Ends: clear unused side (start has no left, end has no right)
    pts[0]['hl'] = pts[0]['co'].copy()
    pts[-1]['hr'] = pts[-1]['co'].copy()
    return pts


def copy_bezier_points(pts):
    return [
        {'co': p['co'].copy(), 'hl': p['hl'].copy(), 'hr': p['hr'].copy()}
        for p in pts
    ]


def eval_bezier_points(pts, t):
    """Evaluate smooth cubic Bezier chain; ends use only their OUT handles."""
    n = len(pts)
    if n == 0:
        return Vector()
    if n == 1:
        return pts[0]['co'].copy()
    if n == 2:
        # single segment: start.hr and end.hl
        return cubic_bezier(pts[0]['co'], pts[0]['hr'], pts[1]['hl'], pts[1]['co'], t)

    seg_f = t * (n - 1)
    i = int(math.floor(seg_f))
    if i >= n - 1:
        return pts[-1]['co'].copy()
    local_t = seg_f - i
    p0 = pts[i]['co']
    h1 = pts[i]['hr']
    h2 = pts[i + 1]['hl']
    p3 = pts[i + 1]['co']
    return cubic_bezier(p0, h1, h2, p3, local_t)



def curve_tangent_at(pts, t, eps=0.002):
    """Unit tangent of Bezier chain at parameter t."""
    t0 = max(0.0, t - eps)
    t1 = min(1.0, t + eps)
    d = eval_bezier_points(pts, t1) - eval_bezier_points(pts, t0)
    if d.length < 1e-12:
        # fallback larger step
        t0 = max(0.0, t - 0.02)
        t1 = min(1.0, t + 0.02)
        d = eval_bezier_points(pts, t1) - eval_bezier_points(pts, t0)
    if d.length < 1e-12:
        return Vector((1, 0, 0))
    return d.normalized()


def curve_curvature_length(pts, t, neighbor_dist):
    """Handle length from local curvature / neighbor spacing."""
    eps = 0.02
    t0 = max(0.0, t - eps)
    t1 = min(1.0, t + eps)
    p0 = eval_bezier_points(pts, t0)
    p1 = eval_bezier_points(pts, t)
    p2 = eval_bezier_points(pts, t1)
    v0 = p1 - p0
    v1 = p2 - p1
    if v0.length < 1e-12 or v1.length < 1e-12:
        return max(neighbor_dist * 0.3, 1e-4)
    v0n, v1n = v0.normalized(), v1.normalized()
    # turning: 0 straight, 2 sharp
    turn = 1.0 - max(-1.0, min(1.0, v0n.dot(v1n)))
    # sharper bend -> slightly shorter handles; straight -> longer
    factor = 0.33 - 0.12 * turn
    factor = max(0.15, min(0.4, factor))
    return max(neighbor_dist * factor, neighbor_dist * 0.12)


def nearest_param_on_polyline(points, point):
    if not points:
        return 0.0
    if len(points) == 1:
        return 0.0
    lengths = [0.0]
    for i in range(1, len(points)):
        lengths.append(lengths[-1] + (points[i] - points[i - 1]).length)
    total = lengths[-1]
    if total < 1e-12:
        return 0.0
    best_d, best_u = 1e18, 0.0
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        ab = b - a
        lab2 = ab.length_squared
        if lab2 < 1e-16:
            u_seg, proj = 0.0, a
        else:
            u_seg = max(0.0, min(1.0, (point - a).dot(ab) / lab2))
            proj = a + ab * u_seg
        d = (point - proj).length_squared
        if d < best_d:
            best_d = d
            seg_len = lengths[i + 1] - lengths[i]
            best_u = (lengths[i] + seg_len * u_seg) / total
    return best_u


def prop_falloff_weight(t, falloff='SMOOTH'):
    """Blender-compatible proportional falloff. t = dist/radius in [0,1]."""
    t = max(0.0, min(1.0, float(t)))
    # enum may come as string name
    fo = str(falloff).upper().replace(' ', '_')
    if fo == 'CONSTANT':
        return 1.0
    if fo == 'LINEAR':
        return 1.0 - t
    if fo == 'SHARP':
        u = 1.0 - t
        return u * u
    if fo == 'ROOT':
        return math.sqrt(max(0.0, 1.0 - t))
    if fo == 'SPHERE':
        return max(0.0, 1.0 - t * t)
    if fo == 'INVERSE_SQUARE':
        tt = t * t
        u = 1.0 - t
        return (u * u) / (u * u + tt + 1e-8)
    # SMOOTH (default): 1 - (3t^2 - 2t^3)
    return 1.0 - (t * t * (3.0 - 2.0 * t))



def ensure_point_inf_falloff(n, existing=None, default='CONSTANT'):
    """Per-controller influence falloff type list length n."""
    out = list(existing) if existing else []
    while len(out) < n:
        out.append(default)
    return out[:n]


def ensure_point_influence(n, existing=None, default=0.1):
    """Per-controller influence radius list length n (always remembered)."""
    out = []
    src = list(existing) if existing else []
    d = max(float(default), 1e-4)
    for i in range(n):
        if i < len(src):
            try:
                out.append(max(float(src[i]), 1e-4))
            except Exception:
                out.append(d)
        else:
            out.append(d)
    return out


def spine_influence_weight(dist, influence, falloff='SMOOTH'):
    """Soft radial weight for spine deform. 1 at curve, 0 at/beyond influence radius."""
    r = max(float(influence), 1e-8)
    t = float(dist) / r
    if t >= 1.0:
        return 0.0
    if t <= 0.0:
        return 1.0
    return float(prop_falloff_weight(t, falloff))


_INFLUENCE_FALLOFF_ORDER = (
    'SMOOTH', 'SPHERE', 'ROOT', 'INVERSE_SQUARE', 'SHARP', 'LINEAR', 'CONSTANT',
)


def _vdh_spine_vg_name(index, stored=None):
    """Stable vertex-group name for a spine chain."""
    if stored:
        name = str(stored).strip()
        if name:
            return name
    i = int(index or 0)
    if i <= 0:
        return "BH_Spine"
    return "BH_Spine.{:03d}".format(i)


def _vdh_ensure_vertex_group(obj, name):
    if obj is None or getattr(obj, 'type', None) != 'MESH' or not name:
        return None
    vg = obj.vertex_groups.get(name)
    if vg is None:
        try:
            vg = obj.vertex_groups.new(name=name)
        except Exception:
            return None
    return vg


def _vdh_is_spine_vg_name(name):
    n = str(name or "")
    return n == "BH_Spine" or n.startswith("BH_Spine.")


def _vdh_remove_spine_vertex_group(obj, name):
    """Delete one specific spine vertex group by its stored chain name."""
    if obj is None or getattr(obj, 'type', None) != 'MESH' or not name:
        return False
    try:
        vg = obj.vertex_groups.get(str(name))
        if vg is None:
            return False
        obj.vertex_groups.remove(vg)
        return True
    except Exception:
        return False


def _vdh_remove_spine_vertex_groups(obj):
    """Delete BH_Spine* groups on a mesh object. Returns how many were removed."""
    if obj is None or getattr(obj, 'type', None) != 'MESH':
        return 0
    names = [vg.name for vg in obj.vertex_groups if _vdh_is_spine_vg_name(vg.name)]
    n = 0
    for name in names:
        vg = obj.vertex_groups.get(name)
        if vg is None:
            continue
        try:
            obj.vertex_groups.remove(vg)
            n += 1
        except Exception:
            pass
    return n


class MESH_OT_vertex_deform_handles(Operator):
    bl_idname = "mesh.vertex_deform_handles"
    bl_label = "Blue Handles"
    bl_options = {'REGISTER', 'UNDO'}

    handle_count: IntProperty(
        name="Max Handles",
        description="Maximum number of auto-placed handles based on selection curvature",
        default=7,
        min=2,
        max=10,
    )

    def invoke(self, context, event):
        global _active_vdh_op
        # Prevent overlapping modals (re-running tool while already active)
        if _active_vdh_op is not None and _active_vdh_op is not self:
            try:
                _active_vdh_op.finish(context, cancel=False)
            except Exception:
                try:
                    if getattr(_active_vdh_op, '_draw_handle', None) is not None:
                        bpy.types.SpaceView3D.draw_handler_remove(_active_vdh_op._draw_handle, 'WINDOW')
                        _active_vdh_op._draw_handle = None
                    if getattr(_active_vdh_op, '_draw_text_handle', None) is not None:
                        bpy.types.SpaceView3D.draw_handler_remove(_active_vdh_op._draw_text_handle, 'WINDOW')
                        _active_vdh_op._draw_text_handle = None
                except Exception:
                    pass
            _active_vdh_op = None

        obj = context.object
        if obj is None or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
            self.report({'WARNING'}, "Edit Mode mesh required")
            return {'CANCELLED'}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        sel = get_selected_vert_indices(bm)

        # No (or insufficient) selection → Spine Deform mode
        if len(sel) < 2:
            return self._invoke_spine(context, event)

        ordered = order_verts_along_chain(bm, sel)
        self.tool_mode = 'VERTEX'  # vs SPINE_PLACE / SPINE_DEFORM

        # Store LOCAL rest positions (object space) - survives object move/rotate/scale
        self.vert_indices = list(ordered)
        self.rest_local = [bm.verts[i].co.copy() for i in ordered]
        self.params = []
        if len(self.rest_local) == 1:
            self.params = [0.0]
        else:
            # parameterize by chord length
            lengths = [0.0]
            for i in range(1, len(self.rest_local)):
                lengths.append(lengths[-1] + (self.rest_local[i] - self.rest_local[i - 1]).length)
            total = lengths[-1] if lengths[-1] > 1e-12 else 1.0
            self.params = [L / total for L in lengths]

        # Auto-place on-curve points, then build real Bezier handles (like Curve)
        cos = auto_fit_handles(
            self.rest_local,
            max_count=self.handle_count,
            min_count=2,
        )
        if len(cos) < 2:
            cos = sample_polyline(self.rest_local, min(3, len(self.rest_local)))

        ok, center, radius, nrm = detect_circular_arc(self.rest_local)
        self._arc_center = center if ok else None
        self._arc_normal = nrm if ok else None

        self.bez = make_bezier_points(
            cos, poly=self.rest_local, center=self._arc_center, normal=self._arc_normal
        )
        self.rest_bez = copy_bezier_points(self.bez)

        self.handle_params = [nearest_param_on_polyline(self.rest_local, p['co']) for p in self.bez]
        self.selected = set()
        # All controllers AUTO by default (V menu or drag tip → Aligned/Free)
        n_bez = len(self.bez)
        self.point_modes = ['AUTO'] * n_bez
        if self.bez:
            self.active_handle = 0
            self.active_bez_part = 'co'
            self.selected.add((0, 'co'))
        # apply look-at auto ends once (does not overwrite FREE interiors)
        self.rebuild_auto_handles()
        self.rest_bez = copy_bezier_points(self.bez)
        if self.handle_params:
            self.handle_params[0] = 0.0
            self.handle_params[-1] = 1.0
            for i in range(1, len(self.handle_params)):
                if self.handle_params[i] <= self.handle_params[i - 1]:
                    self.handle_params[i] = min(1.0, self.handle_params[i - 1] + 1e-4)
        # Map each vert to nearest point on rest curve → tiny offsets, clean deform weights
        self._reparam_verts_on_curve()

        self.active_handle = None
        self.active_bez_part = 'co'
        self.selected = set()  # set of (index, part)
        self.handle_mode = 'AUTO'  # AUTO | FREE | ALIGNED
        self.dragging = False
        self.drag_start_mouse = None
        self.drag_start_handle = None
        self.drag_plane_point = None
        self.drag_plane_normal = None

        # Handle-local undo/redo stacks (mesh + handles stay in sync)
        self.undo_stack = []
        self.redo_stack = []

        self._obj_name = obj.name_full
        self._draw_handle = None
        self._draw_text_handle = None
        self._lock_selection = list(ordered)

        # Proportional - always snapshot so O / scroll can be used during modal
        self.prop_size = context.tool_settings.proportional_size
        self.prop_falloff = context.tool_settings.proportional_edit_falloff
        self.display_scale = _vdh_get_vertex_display_scale(obj, 1.0)
        self.all_rest = {v.index: v.co.copy() for v in bm.verts}
        # Immutable originals for proper Cancel (ESC)
        self.initial_rest_local = [p.copy() for p in self.rest_local]
        self.initial_all_rest = {k: v.copy() for k, v in self.all_rest.items()}
        self._ctrl_snap = False  # Ctrl held -> temporary snap
        self._prop_kdtree = None          # of selected rest_local
        self._all_kdtree = None           # of all verts for fast prop queries
        self._all_kdtree_dirty = False
        self._pending_undo_prop_before = None
        self._history_prop_snap = {}
        self._history_fast_undo = False
        self._prop_last_affected = set()  # verts touched by prop last frame
        self._prop_was_on = bool(context.tool_settings.use_proportional_edit)
        self._prop_circle_mouse = None
        # Vertex Mode mesh mirror: OFF / AUTO / X / Y / Z.  This mirrors only
        # mesh deformation; Bezier controllers/handles on the opposite side are
        # never moved by this feature.
        self._vertex_mirror_axis = 'AUTO'
        self._vertex_mirror_pairs = None
        self._vertex_mirror_pairs_key = None
        self._vertex_mirror_last_targets = set()
        # Per-drag mirror baseline. This preserves any deformation that already
        # exists on the mirrored side when the user switches from Handle Drag
        # to Controller Drag (or starts another drag).
        self._vertex_mirror_drag_source_base = {}
        self._vertex_mirror_drag_target_base = {}
        self._vertex_mirror_drag_pairs = None
        self.smooth_mode = VDH_SMOOTH_MODE
        self._rebuild_prop_kdtree()
        self._rebuild_all_kdtree()

        # Box select state
        self.box_selecting = False
        self.box_start = None
        self.box_end = None
        self.box_handles_only = False

        # Hide transform gizmos while tool is active
        self._gizmo_backup = {}
        try:
            sp = context.space_data
            for attr in (
                'show_gizmo', 'show_gizmo_context', 'show_gizmo_tool',
                'show_gizmo_object_translate', 'show_gizmo_object_rotate',
                'show_gizmo_object_scale',
            ):
                if hasattr(sp, attr):
                    self._gizmo_backup[attr] = getattr(sp, attr)
                    setattr(sp, attr, False)
        except Exception:
            pass

        _active_vdh_op = self
        context.window_manager.modal_handler_add(self)
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        self._draw_text_handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_text_callback, (context,), 'WINDOW', 'POST_PIXEL'
        )
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def get_obj_bm(self, context):
        obj = context.object
        if obj is None or obj.name_full != self._obj_name:
            obj = bpy.data.objects.get(self._obj_name.split(".")[0])
            # fallback by name_full search
            for o in bpy.data.objects:
                if o.name_full == self._obj_name:
                    obj = o
                    break
        if obj is None or obj.type != 'MESH':
            return None, None
        if context.mode != 'EDIT_MESH':
            return obj, None
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        return obj, bm

    def _rebuild_prop_kdtree(self):
        """Build KDTree of rest_local (selected) for nearest lookup."""
        n = len(self.rest_local)
        if n == 0:
            self._prop_kdtree = None
            return
        kd = KDTree(n)
        for i, co in enumerate(self.rest_local):
            kd.insert(co, i)
        kd.balance()
        self._prop_kdtree = kd

    def _rebuild_all_kdtree(self):
        """Build KDTree of ALL verts (from all_rest) so prop only touches nearby verts."""
        self._prop_nearest = None
        self._prop_nearest_key = None
        self._prop_curve_field = None
        if not self.all_rest:
            self._all_kdtree = None
            return
        items = list(self.all_rest.items())  # (index, co)
        kd = KDTree(len(items))
        self._all_kdtree_indices = []
        for i, (vidx, co) in enumerate(items):
            kd.insert(co, i)
            self._all_kdtree_indices.append(vidx)
        kd.balance()
        self._all_kdtree = kd
        self._all_kdtree_dirty = False

    def _vertex_mirror_invalidate_state(self, rebuild=False):
        """Invalidate Vertex-Mirror caches after controller topology/state changes.

        Adding/removing Blue Handles controllers can change the active curve/rest
        representation without changing the mesh vertex count. Any cached mirror
        pairing or per-drag target state must therefore be discarded so the next
        operation rebuilds from the current rest coordinates.
        """
        self._vertex_mirror_pairs = None
        self._vertex_mirror_pairs_key = None
        self._vertex_mirror_drag_pairs = None
        self._vertex_mirror_drag_axis = 'OFF'
        self._vertex_mirror_drag_axis_items = ()
        self._vertex_mirror_drag_source_sides = {}
        self._vertex_mirror_drag_source_base = {}
        self._vertex_mirror_drag_target_base = {}
        self._vertex_mirror_last_targets = set()
        self._prop_curve_field = None
        self._prop_nearest = None
        self._prop_nearest_key = None
        if rebuild:
            try:
                self._all_kdtree_dirty = True
            except Exception:
                pass

    def _vertex_mirror_axis_resolved(self, obj):
        """Resolve Vertex Mode mirror axes directly from Blender's mesh symmetry flags.

        Blue Handles intentionally has no independent Vertex-Mirror switch anymore:
        the Blender Tool Header/Object mirror flags are the single source of truth.
        Multiple enabled axes are supported together (e.g. XY).
        """
        if obj is None:
            return 'OFF'
        axes = []
        for ax, attr in (('X', 'use_mesh_mirror_x'),
                         ('Y', 'use_mesh_mirror_y'),
                         ('Z', 'use_mesh_mirror_z')):
            try:
                if bool(getattr(obj, attr, False)):
                    axes.append(ax)
            except Exception:
                pass
        return ''.join(axes) if axes else 'OFF'

    def _vertex_mirror_build_pairs(self, obj):
        """Build stable one-to-one rest-space mirror pairs.

        Pairing is deliberately bijective. A loose nearest-neighbor lookup can
        make several source vertices share one target, which is especially
        visible when proportional editing is active. That produces uneven
        mirrored falloff. We instead collect valid symmetric candidates and
        greedily assign each target only once, preferring the smallest error.
        """
        axis = self._vertex_mirror_axis_resolved(obj)
        if axis == 'OFF' or not self.all_rest:
            self._vertex_mirror_pairs = {}
            self._vertex_mirror_pairs_key = None
            return {}
        key = (axis, id(self.all_rest), len(self.all_rest),
               tuple(sorted(self.all_rest.keys()))[:8])
        if getattr(self, '_vertex_mirror_pairs_key', None) == key:
            return getattr(self, '_vertex_mirror_pairs', {}) or {}

        items = list(self.all_rest.items())
        if not items:
            self._vertex_mirror_pairs = {}
            self._vertex_mirror_pairs_key = key
            return {}

        kd = KDTree(len(items))
        coords = []
        ids = []
        mn = Vector((1e30, 1e30, 1e30))
        mx = Vector((-1e30, -1e30, -1e30))
        for ti, (vid, co) in enumerate(items):
            kd.insert(co, ti)
            coords.append(co)
            ids.append(int(vid))
            mn.x = min(mn.x, co.x); mn.y = min(mn.y, co.y); mn.z = min(mn.z, co.z)
            mx.x = max(mx.x, co.x); mx.y = max(mx.y, co.y); mx.z = max(mx.z, co.z)
        kd.balance()
        diag = (mx - mn).length

        spacing = 0.0
        try:
            sample_n = min(len(items), 256)
            step = max(1, len(items) // sample_n)
            vals = []
            for ii in range(0, len(items), step):
                hits = kd.find_n(coords[ii], 2)
                if hits and len(hits) > 1:
                    vals.append(float(hits[1][2]))
            if vals:
                vals.sort()
                spacing = vals[len(vals) // 2]
        except Exception:
            pass

        tol = max(diag * 1e-3, spacing * 0.35, 1e-5)
        # Candidate records: (distance, source_id, target_id). Use all nearby
        # candidates so the final assignment is one-to-one rather than relying
        # on KDTree.find()'s arbitrary tie choice.
        candidates = []
        for vid, co in items:
            mco = co.copy()
            if 'X' in axis: mco.x = -mco.x
            if 'Y' in axis: mco.y = -mco.y
            if 'Z' in axis: mco.z = -mco.z
            try:
                hits = kd.find_range(mco, tol)
            except Exception:
                hits = []
            for _hco, hi, dist in hits:
                target = ids[hi]
                if target == int(vid):
                    # Center-plane vertices are valid self-pairs, but do not
                    # compete with an actual opposite-side vertex.  For multi-axis
                    # mirror (e.g. XY), keep a self-pair only when the vertex is
                    # actually on every enabled mirror plane.
                    on_all_planes = True
                    for _ax in axis:
                        _ci = {'X': 0, 'Y': 1, 'Z': 2}[_ax]
                        if abs(float(co[_ci])) > tol:
                            on_all_planes = False
                            break
                    if not on_all_planes:
                        continue
                candidates.append((float(dist), int(vid), target))

        candidates.sort(key=lambda x: x[0])
        used_src = set()
        used_tgt = set()
        pairs = {}
        for dist, src, tgt in candidates:
            if src in used_src or tgt in used_tgt:
                continue
            pairs[src] = tgt
            pairs[tgt] = src
            used_src.add(src)
            used_tgt.add(tgt)

        self._vertex_mirror_pairs = pairs
        self._vertex_mirror_pairs_key = key
        return pairs

    def _vertex_mirror_apply_mesh(self, context, obj, bm, source_indices):
        """Mirror deformation deltas from one side of the mesh to its paired side."""
        axis = getattr(self, '_vertex_mirror_drag_axis', 'OFF')
        if axis == 'OFF':
            axis = self._vertex_mirror_axis_resolved(obj)
        if axis == 'OFF' or not source_indices:
            # Mirror is driven solely by Blender's mesh-mirror flags. Turning
            # those flags off must not erase an already existing deformation;
            # simply stop generating new mirrored updates.
            self._vertex_mirror_last_targets = set()
            return

        pairs = getattr(self, '_vertex_mirror_drag_pairs', None) or self._vertex_mirror_build_pairs(obj)
        if not pairs:
            self._vertex_mirror_last_targets = set()
            return

        # During a drag, use the mesh state at DRAG START as the baseline.
        # This is essential when a Handle was edited first: the opposite side
        # already contains a valid mirrored deformation and must not be rebuilt
        # from all_rest (which would erase it).
        drag_src_base = getattr(self, '_vertex_mirror_drag_source_base', None) or {}
        drag_tgt_base = getattr(self, '_vertex_mirror_drag_target_base', None) or {}
        use_drag_base = bool(drag_src_base and drag_tgt_base)

        # Determine the source quadrant from the actual selected/bound rest verts.
        # Blender can enable X/Y/Z independently, so each enabled axis gets its
        # own source sign. The selected side is authoritative for that axis.
        source_sides = getattr(self, '_vertex_mirror_drag_source_sides', None) or {}
        if not source_sides:
            selected_rest = [p for p in (self.rest_local or []) if p is not None]
            if not selected_rest:
                return
            side_eps = 1e-6
            source_sides = {}
            for ax in axis:
                coord_i = {'X': 0, 'Y': 1, 'Z': 2}[ax]
                pos = sum(1 for co in selected_rest if float(co[coord_i]) > side_eps)
                neg = sum(1 for co in selected_rest if float(co[coord_i]) < -side_eps)
                if pos > neg:
                    source_sides[ax] = 1
                elif neg > pos:
                    source_sides[ax] = -1
                else:
                    avg = sum(float(co[coord_i]) for co in selected_rest) / float(len(selected_rest))
                    source_sides[ax] = 1 if avg >= 0.0 else -1

        source_set = source_indices if isinstance(source_indices, set) else set(source_indices)
        axis_items = getattr(self, '_vertex_mirror_drag_axis_items', None)
        if not axis_items:
            axis_items = tuple((ax, {'X': 0, 'Y': 1, 'Z': 2}[ax]) for ax in axis)

        # The proportional curve field is authored only on the selected/source
        # quadrant. Clear opposite quadrants before applying the exact mirrored
        # delta, so stale independent proportional results cannot survive.
        field_data = getattr(self, '_prop_curve_field', None) or {}
        field_data = field_data.get('data', {}) if isinstance(field_data, dict) else {}
        for vidx in field_data.keys():
            try:
                vidx = int(vidx)
                if vidx >= len(bm.verts):
                    continue
                rest = self.all_rest.get(vidx)
                if rest is None:
                    continue
                opposite = False
                for ax, sign in source_sides.items():
                    coord_i = dict(axis_items)[ax]
                    sval = float(rest[coord_i])
                    if ((sval > 1e-7 and sign < 0) or (sval < -1e-7 and sign > 0)):
                        opposite = True
                        break
                if opposite:
                    bm.verts[vidx].co = rest.copy()
            except Exception:
                continue

        new_targets = set()
        for vidx in source_set:
            rest = self.all_rest.get(vidx)
            if rest is None or vidx >= len(bm.verts):
                continue
            # Only the selected/source quadrant is authoritative.
            valid_source = True
            for ax, sign in source_sides.items():
                coord_i = {'X': 0, 'Y': 1, 'Z': 2}[ax]
                sval = float(rest[coord_i])
                if ((sval < -1e-7 and sign > 0) or (sval > 1e-7 and sign < 0)):
                    valid_source = False
                    break
            if not valid_source:
                continue
            target = pairs.get(vidx)
            if target is None or target == vidx or target in source_set:
                continue
            if target >= len(bm.verts):
                continue
            # The source vertex has already received its complete deformation
            # field (including proportional falloff). Mirror only the CHANGE
            # produced during this drag. If a previous Handle deformation already
            # existed, it remains in the target drag baseline instead of being
            # replaced by all_rest. Do NOT run proportional falloff a second time.
            if use_drag_base and vidx in drag_src_base and target in drag_tgt_base:
                delta = bm.verts[vidx].co - drag_src_base[vidx]
                target_base = drag_tgt_base[target]
            else:
                delta = bm.verts[vidx].co - rest
                target_base = self.all_rest.get(target)
                if target_base is None:
                    continue
            md = delta.copy()
            for ax, coord_i in axis_items:
                md[coord_i] = -md[coord_i]
            bm.verts[target].co = target_base + md
            new_targets.add(target)

        # Restore mirror targets that have left the current proportional field.
        old_targets = getattr(self, '_vertex_mirror_last_targets', set()) or set()
        for vidx in (old_targets - new_targets):
            if vidx in source_set or vidx >= len(bm.verts):
                continue
            if use_drag_base and vidx in drag_tgt_base:
                bm.verts[vidx].co = drag_tgt_base[vidx].copy()
            else:
                rest = self.all_rest.get(vidx)
                if rest is not None:
                    bm.verts[vidx].co = rest.copy()
        self._vertex_mirror_last_targets = new_targets

    def _vertex_mirror_operation_begin(self, context):
        """Capture the pre-operation mesh state for an exact operation-level mirror.

        R/L/S/F/Shift+Scroll are not drags: they rebuild rest/curve state and may
        move a proportional neighborhood.  Capture the complete mesh before the
        operation, then mirror the final absolute source positions afterward.
        This keeps the existing drag mirror engine completely separate.
        """
        if getattr(self, 'tool_mode', 'VERTEX') != 'VERTEX':
            return None
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return None
        axis = self._vertex_mirror_axis_resolved(obj)
        if axis == 'OFF':
            return None
        pairs = self._vertex_mirror_build_pairs(obj)
        if not pairs:
            return None

        selected_rest = [p.copy() for p in (getattr(self, 'rest_local', None) or []) if p is not None]
        if not selected_rest:
            return None

        # The selected/source quadrant is authoritative, matching the existing
        # drag mirror behavior.  Zero coordinates are allowed on an individual
        # mirror plane; they only become ambiguous if every selected point is on it.
        source_sides = {}
        eps = 1e-6
        for ax in axis:
            ci = {'X': 0, 'Y': 1, 'Z': 2}[ax]
            pos = sum(1 for co in selected_rest if float(co[ci]) > eps)
            neg = sum(1 for co in selected_rest if float(co[ci]) < -eps)
            if pos > neg:
                source_sides[ax] = 1
            elif neg > pos:
                source_sides[ax] = -1
            else:
                avg = sum(float(co[ci]) for co in selected_rest) / float(len(selected_rest))
                source_sides[ax] = 1 if avg >= 0.0 else -1

        before = {int(v.index): v.co.copy() for v in bm.verts}
        return {
            'obj': obj,
            'bm': bm,
            'axis': axis,
            'pairs': dict(pairs),
            'source_sides': source_sides,
            'before': before,
        }

    def _vertex_mirror_operation_end(self, context, state):
        """Apply the exact final source result to mirrored vertices.

        The operation itself remains untouched.  We compare pre/post positions on
        the source side, then reflect the *final absolute position* (not a second
        proportional falloff).  This is the key difference from trying to call the
        drag mirror routine from R/L/S/F/Align.
        """
        if not state:
            return
        try:
            obj = state['obj']
            bm = state['bm']
            axis = state['axis']
            pairs = state['pairs']
            source_sides = state['source_sides']
            before = state['before']
            bm.verts.ensure_lookup_table()

            def is_source(co):
                for ax, sign in source_sides.items():
                    ci = {'X': 0, 'Y': 1, 'Z': 2}[ax]
                    val = float(co[ci])
                    if (val < -1e-7 and sign > 0) or (val > 1e-7 and sign < 0):
                        return False
                return True

            mirrored = set()
            eps2 = 1e-12
            # Each pair is visited once from the source side.
            for src, tgt in pairs.items():
                src = int(src); tgt = int(tgt)
                if src == tgt or src not in before or tgt >= len(bm.verts):
                    continue
                if not is_source(before[src]):
                    continue
                old = before[src]
                new = bm.verts[src].co.copy()
                if (new - old).length_squared <= eps2:
                    continue

                # Reflect the final source position across every active Blender
                # mirror plane.  This handles X/Y/Z and combinations such as XY.
                target_co = new.copy()
                for ax in axis:
                    ci = {'X': 0, 'Y': 1, 'Z': 2}[ax]
                    target_co[ci] = -target_co[ci]
                bm.verts[tgt].co = target_co
                mirrored.add(tgt)

            if mirrored:
                # The mirrored result is now the baked rest state for those target
                # vertices, just like the source operation's own rest updates.
                for vidx in mirrored:
                    self.all_rest[vidx] = bm.verts[vidx].co.copy()

                # Rest-space caches depend on these coordinates.  Invalidate them
                # rather than allowing a later proportional edit to use stale data.
                self._all_kdtree_dirty = True
                self._prop_curve_field = None
                self._prop_nearest = None
                self._prop_nearest_key = None
                self._vertex_mirror_last_targets = set(mirrored)

                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                try:
                    bm.normal_update()
                    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
                except Exception:
                    pass
                try:
                    obj.data.update()
                except Exception:
                    pass
                try:
                    context.area.tag_redraw()
                except Exception:
                    pass
        except Exception:
            # Mirror must never break the underlying modeling operation.
            return

    def apply_deform(self, context):
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return

        if getattr(self, '_all_kdtree_dirty', False):
            self._rebuild_all_kdtree()

        # Cache curve samples for unique parameters (selection often shares u)
        u_cache = {}

        def _curve_at(u):
            q = int(round(float(u) * 512.0))
            hit = u_cache.get(q)
            if hit is not None:
                return hit
            rest_on = eval_bezier_points(self.rest_bez, u)
            now_on = eval_bezier_points(self.bez, u)
            hit = (rest_on, now_on)
            u_cache[q] = hit
            return hit

        # Follow full Bezier (control points + handles). Keep per-vert offset
        # from rest curve so mesh reacts when hl/hr are edited too.
        for idx, v_i in enumerate(self.vert_indices):
            if v_i >= len(bm.verts):
                continue
            u = self.params[idx]
            rest_on, now_on = _curve_at(u)
            offset = self.rest_local[idx] - rest_on
            bm.verts[v_i].co = now_on + offset

        # proportional – smooth influence around the whole Bezier curve.
        #
        # The old field used the nearest selected vertex as the deformation
        # source. On a tube this creates visible hand-off zones: adjacent mesh
        # vertices can suddenly use different controller deltas. Instead, locate
        # each affected vertex against the continuous REST Bezier curve and use
        # the curve displacement at that longitudinal position. Radial distance
        # from the curve controls the proportional falloff. This gives a single,
        # continuous deformation field around the whole tube.
        prop_on = context.tool_settings.use_proportional_edit
        if prop_on and self.all_rest is not None and self._all_kdtree is not None:
            radius = max(float(context.tool_settings.proportional_size), 1e-6)
            self.prop_size = radius
            falloff = getattr(
                context.tool_settings, 'proportional_edit_falloff', 'SMOOTH'
            )
            self.prop_falloff = falloff
            sel_set = set(self.vert_indices)

            # Cache the rest-space influence field. The moving Bezier does not
            # invalidate this: only rest geometry, radius and topology matter.
            curve_key = (
                round(radius, 6),
                id(self.rest_bez),
                len(self.rest_bez or []),
                id(self._all_kdtree),
                id(self.all_rest),
            )
            field = getattr(self, '_prop_curve_field', None)
            if field is None or field.get('key') != curve_key:
                rest_bez = self.rest_bez or []
                field_data = {}
                if rest_bez:
                    samples_n = max(256, min(1024, max(2, len(rest_bez) * 64)))
                    samples = []
                    sample_us = []
                    for si in range(samples_n):
                        u = si / float(samples_n - 1)
                        samples.append(eval_bezier_points(rest_bez, u))
                        sample_us.append(u)
                    kd_curve = KDTree(len(samples))
                    for si, co in enumerate(samples):
                        kd_curve.insert(co, si)
                    kd_curve.balance()

                    for vidx, rest_co in self.all_rest.items():
                        if vidx in sel_set:
                            continue
                        nearest = kd_curve.find(rest_co)
                        if nearest is None:
                            continue
                        _pnt, si, dist = nearest
                        if dist > radius:
                            continue

                        # Refine the nearest parameter using the two neighbouring
                        # sampled segments, so the field does not step with the
                        # KDTree samples.
                        best_d2 = float(dist) * float(dist)
                        best_u = sample_us[si]
                        for sj in (si - 1, si):
                            if sj < 0 or sj >= samples_n - 1:
                                continue
                            a = samples[sj]
                            b = samples[sj + 1]
                            ab = b - a
                            lab2 = ab.length_squared
                            if lab2 <= 1e-16:
                                seg_t = 0.0
                                q = a
                            else:
                                seg_t = max(0.0, min(1.0, (rest_co - a).dot(ab) / lab2))
                                q = a + ab * seg_t
                            d2 = (rest_co - q).length_squared
                            if d2 < best_d2:
                                best_d2 = d2
                                best_u = sample_us[sj] + (sample_us[sj + 1] - sample_us[sj]) * seg_t
                        field_data[int(vidx)] = (
                            float(math.sqrt(max(best_d2, 0.0))),
                            float(best_u),
                        )

                field = {'key': curve_key, 'data': field_data}
                self._prop_curve_field = field

            current = field.get('data', {})
            # In Vertex Mirror mode the proportional field is authored only on
            # the selected/source side.  The opposite side must NOT receive a
            # second independent curve-field evaluation; it will be populated
            # later by the exact mirrored delta.  This is critical on dense
            # meshes where both sides can fall inside the proportional radius.
            mirror_axis = self._vertex_mirror_axis_resolved(obj) if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX' else 'OFF'
            mirror_coord = {'X': 0, 'Y': 1, 'Z': 2}.get(mirror_axis)
            mirror_source_side = 0
            if mirror_coord is not None:
                _pos = sum(1 for co in self.rest_local if float(co[mirror_coord]) > 1e-6)
                _neg = sum(1 for co in self.rest_local if float(co[mirror_coord]) < -1e-6)
                if _pos > _neg:
                    mirror_source_side = 1
                elif _neg > _pos:
                    mirror_source_side = -1
                else:
                    _avg = sum(float(co[mirror_coord]) for co in self.rest_local) / max(1, len(self.rest_local))
                    mirror_source_side = 1 if _avg >= 0.0 else -1
            last = getattr(self, '_prop_last_affected', None) or set()
            for vidx in (set(last) - set(current.keys())):
                rest_co = self.all_rest.get(vidx)
                if rest_co is not None and vidx < len(bm.verts):
                    bm.verts[vidx].co = rest_co

            affected = set()
            for vidx, (dist, u) in current.items():
                if vidx >= len(bm.verts):
                    continue
                rest_co = self.all_rest.get(vidx)
                if mirror_coord is not None and mirror_source_side != 0:
                    _side = float(rest_co[mirror_coord])
                    # Keep center-plane vertices local to the source deformation;
                    # only the source half is evaluated here.
                    if ((_side > 1e-7 and mirror_source_side < 0) or
                            (_side < -1e-7 and mirror_source_side > 0)):
                        bm.verts[vidx].co = rest_co
                        continue
                if rest_co is None:
                    continue
                w = prop_falloff_weight(dist / radius, falloff)
                if w <= 1e-8:
                    bm.verts[vidx].co = rest_co
                    continue

                rest_on = eval_bezier_points(self.rest_bez, u)
                now_on = eval_bezier_points(self.bez, u)
                curve_delta = now_on - rest_on
                bm.verts[vidx].co = rest_co + curve_delta * w
                affected.add(vidx)

            self._prop_last_affected = affected
        else:
            # prop off: restore any previously affected verts once
            last = getattr(self, '_prop_last_affected', None)
            if last:
                for vidx in last:
                    rest_co = self.all_rest.get(vidx)
                    if rest_co is not None and vidx < len(bm.verts):
                        bm.verts[vidx].co = rest_co
                self._prop_last_affected = set()
            self._prop_nearest = None
            self._prop_nearest_key = None
            self._prop_curve_field = None
            self._prop_curve_field = None

        # Vertex Mode mesh mirror: mirror the complete deformation field
        # (selected + proportional vertices) without moving opposite-side handles.
        if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX':
            mirror_axis = getattr(self, '_vertex_mirror_drag_axis', 'OFF')
            if mirror_axis == 'OFF':
                mirror_axis = self._vertex_mirror_axis_resolved(obj)
            if mirror_axis != 'OFF':
                mirror_sources = set(self.vert_indices)
                mirror_sources |= set(getattr(self, '_prop_last_affected', set()) or set())
                self._vertex_mirror_apply_mesh(context, obj, bm, mirror_sources)

        # Live normal refresh for Vertex Mode deformation.
        # Controller and handle movement changes vertex positions through this
        # path, so keep viewport normals in sync without touching Spine mode.
        try:
            bm.normal_update()
        except Exception:
            pass
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


    # --------------------------------------------------
    # Spine Deform mode (no vertex selection)
    # --------------------------------------------------
    def _invoke_spine(self, context, event):
        """Enter Spine placement mode: click to add controllers (uses Blender snap)."""
        obj = context.object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        self.tool_mode = 'SPINE_PLACE'
        self.display_scale = getattr(self, 'display_scale', 1.0) or 1.0
        self._show_influence = False
        # Spine-only weight overlay marker size (screen pixels).  This is
        # intentionally separate from display_scale so controller/handle sizes
        # are never affected by the overlay-size shortcut.
        self._influence_overlay_size = 4.0
        self.influence_falloff = 'CONSTANT'  # default; per-controller overrides in point_inf_falloff
        # Spine Shrink/Inflate + Tilt interpolation falloff. This is persisted
        # per chain so reopening the tool keeps the exact falloff used last.
        self.spine_attr_interp = 'SMOOTH'
        self.point_inf_falloff = []
        self.point_influence = []  # per-controller influence radius
        self._obj_name = obj.name_full
        # Spine Mode always starts with In Front enabled.  This is an
        # intentional Blue Handles default on every tool entry; do not restore
        # the object's previous show_in_front state on exit.
        try:
            # Object In Front must not be changed; only Spine chain overlays use in_front.
            pass
        except Exception:
            pass
        # Blue Handles Spine Mode always starts with every recalled chain In Front.
        # This is intentionally NOT restored from the previous per-chain state.
        try:
            for _ch in (getattr(self, 'spine_chains', None) or []):
                _ch['in_front'] = True
        except Exception:
            pass
        self.spine_points = []          # current chain controllers (local Vectors)
        self.spine_chains_pts = []       # completed chains while placing [[pts], ...]
        self.spine_chains = []           # bound chain data after Enter [{bez,bind,...}, ...]
        self.active_chain = 0
        self._spine_edit_place = False
        self._spine_placing_new_chain = False
        self._spine_origin_ids = []
        self._spine_chains_origin_ids = []
        self._spine_last_add_idx = None  # index of last controller the user placed
        self._face_flip_parity = 0  # toggles on geometric mirror; tracked in undo
        self.spine_bind = []            # list of (vert_index, t, offset Vector)
        self.spine_rest_poly = []       # rest polyline after bind
        self.selected = set()
        self.active_handle = None
        self.active_bez_part = 'co'
        self.dragging = False
        self.drag_start_mouse = None
        self.drag_start_handle = None
        self.drag_plane_point = None
        self.drag_plane_normal = None
        self.drag_start_mouse_local = None
        self.constraint_axis = None
        self.undo_stack = []
        self.redo_stack = []
        self.point_modes = []
        self.bez = []
        self.rest_bez = []
        self.handle_params = []
        self.vert_indices = []
        self.rest_local = []
        self.params = []
        self._lock_selection = []
        self.prop_size = context.tool_settings.proportional_size
        self.prop_falloff = context.tool_settings.proportional_edit_falloff
        self.all_rest = {v.index: v.co.copy() for v in bm.verts}
        self.initial_all_rest = {k: v.copy() for k, v in self.all_rest.items()}
        self.initial_rest_local = []
        self._ctrl_snap = False
        self._prop_kdtree = None
        self._all_kdtree = None
        self._prop_last_affected = set()
        self._prop_was_on = bool(context.tool_settings.use_proportional_edit)
        self.smooth_mode = VDH_SMOOTH_MODE
        self.box_selecting = False
        self.box_start = None
        self.box_end = None
        self.box_handles_only = False
        # Place in Volume (P): toggle Blender snap → VOLUME
        ts = context.tool_settings
        elems = set(getattr(ts, 'snap_elements', set()) or set())
        self._snap_backup_use = bool(ts.use_snap)
        self._snap_backup_elements = elems.copy() if elems else set()
        self._place_in_volume = bool(ts.use_snap and 'VOLUME' in elems)
        self._gizmo_backup = {}
        try:
            sp = context.space_data
            for attr in (
                'show_gizmo', 'show_gizmo_context', 'show_gizmo_tool',
                'show_gizmo_object_translate', 'show_gizmo_object_rotate',
                'show_gizmo_object_scale',
            ):
                if hasattr(sp, attr):
                    self._gizmo_backup[attr] = getattr(sp, attr)
                    setattr(sp, attr, False)
        except Exception:
            pass

        global _active_vdh_op
        _active_vdh_op = self
        context.window_manager.modal_handler_add(self)
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback, (context,), 'WINDOW', 'POST_VIEW'
        )
        self._draw_text_handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_text_callback, (context,), 'WINDOW', 'POST_PIXEL'
        )
        # Lock mesh selection empty
        for v in bm.verts:
            v.select = False
        for e in bm.edges:
            e.select = False
        for f in bm.faces:
            f.select = False
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        self._lock_selection = []

        # Auto-resume previous spine on this mesh (controllers + weights, no rebind)
        if self._spine_try_auto_restore(context):
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        context.area.tag_redraw()
        self.report({'INFO'}, "Spine Deform: Click to place controllers  |  Enter: Bind & Deform  |  Esc: Cancel")
        return {'RUNNING_MODAL'}

    def _spine_ray_hits(self, context, ray_origin, view_vec, max_hits=12):
        """Collect successive ray_cast hits along a ray (for volume snap)."""
        hits = []
        depsgraph = context.evaluated_depsgraph_get()
        origin = ray_origin.copy()
        direction = view_vec.normalized()
        for _ in range(max_hits):
            try:
                hit, loc, normal, face_index, hit_obj, matrix = context.scene.ray_cast(
                    depsgraph, origin, direction
                )
            except Exception:
                break
            if not hit:
                break
            hits.append((loc.copy(), normal.copy() if normal else Vector((0, 0, 1)), hit_obj))
            # Push through surface to find next intersection
            origin = loc + direction * 1e-4
        return hits

    def _spine_mouse_local(self, context, event):
        """Project mouse to object-local 3D using Blender snap modes (Face / Volume / etc.)."""
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return None
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        if context.tool_settings.use_proportional_edit:
            self._prop_circle_mouse = coord
        view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        inv = obj.matrix_world.inverted()
        ts = context.tool_settings
        elements = set(getattr(ts, 'snap_elements', set()) or set())
        use_snap = bool(ts.use_snap) or bool(event.ctrl)
        # Honor Place-in-Volume toggle even if Blender snap flags drifted
        if getattr(self, '_place_in_volume', False):
            use_snap = True
            elements = set(elements) | {'VOLUME'}

        world = None
        hits = self._spine_ray_hits(context, ray_origin, view_vec)

        if use_snap and 'VOLUME' in elements and len(hits) >= 2:
            # Interior of volume: midpoint between first entry and exit along the ray
            world = (hits[0][0] + hits[1][0]) * 0.5
        elif use_snap and 'VOLUME' in elements and len(hits) == 1:
            # Single hit: place slightly inside along -normal
            loc, normal, _ = hits[0]
            world = loc - normal.normalized() * 0.01
        elif hits:
            # FACE / EDGE / default surface: first hit under cursor
            world = hits[0][0].copy()
        else:
            # Fallback: view plane through object origin
            plane_n = rv3d.view_rotation @ Vector((0, 0, 1))
            plane_p = obj.matrix_world.translation
            denom = view_vec.dot(plane_n)
            if abs(denom) < 1e-8:
                return None
            t = (plane_p - ray_origin).dot(plane_n) / denom
            world = ray_origin + view_vec * t

        local = inv @ world
        # Grid / vertex snap on top (when enabled).
        # Do NOT overwrite drag_start_handle — that breaks live relative drag (jitter).
        if use_snap:
            saved = getattr(self, 'drag_start_handle', None)
            try:
                # Provide a stable reference for INCREMENT snap only
                if saved is None:
                    self.drag_start_handle = local.copy()
                local = self.snap_local(context, obj, local, force=True)
            finally:
                if saved is not None:
                    self.drag_start_handle = saved
                elif hasattr(self, 'drag_start_handle') and not getattr(self, 'dragging', False):
                    pass
                elif saved is None and getattr(self, 'dragging', False):
                    # restore whatever start_drag set
                    pass
            if saved is not None:
                self.drag_start_handle = saved
        return local


    def _spine_activate_place_chain(self, key):
        """Activate a chain for place/edit.
        key: 'current' | int index into spine_chains_pts (new place chains)
             | ('chain', ci) for spine_chains when Edit Place
        """
        if key == 'current':
            return
        # Edit Place: switch among bound spine_chains
        if isinstance(key, tuple) and len(key) == 2 and key[0] == 'chain':
            ci = int(key[1])
            chains = getattr(self, 'spine_chains', None) or []
            if not (0 <= ci < len(chains)):
                return
            # Store current active into its chain (keep list identity)
            try:
                self._spine_store_active_chain()
            except Exception:
                pass
            # Sync cos into origin tracking for active
            if getattr(self, 'bez', None):
                self.spine_points = [bp['co'].copy() for bp in self.bez]
            self.active_chain = ci
            try:
                self._spine_load_active_chain()
            except Exception:
                ch = chains[ci]
                self.bez = ch.get('bez')
                self.point_modes = ch.get('modes') or ['AUTO'] * len(self.bez or [])
            self.spine_points = [bp['co'].copy() for bp in (self.bez or [])]
            ch = chains[ci]
            n = len(self.spine_points)
            self._spine_origin_ids = list(ch.get('origin_ids') or list(range(n)))
            self.selected = set()
            self.active_handle = None
            self.active_bez_part = 'co'
            self._spine_placing_new_chain = False
            return
        # Legacy: index into spine_chains_pts (new chains being placed)
        if key < 0 or key >= len(getattr(self, 'spine_chains_pts', []) or []):
            return
        # If leaving an existing deform chain that was active, store it first
        if getattr(self, '_spine_edit_place', False) and getattr(self, 'spine_chains', None):
            try:
                self._spine_store_active_chain()
            except Exception:
                pass
        old_pts = [p.copy() for p in (self.spine_points or [])]
        old_oids = list(getattr(self, '_spine_origin_ids', None) or [])
        pts = [p.copy() for p in self.spine_chains_pts[key]]
        oids_list = getattr(self, '_spine_chains_origin_ids', None) or []
        oids = list(oids_list[key]) if key < len(oids_list) else list(range(len(pts)))
        if len(oids) != len(pts):
            oids = list(range(len(pts)))
        new_completed = []
        new_oids = []
        for i, chp in enumerate(self.spine_chains_pts):
            if i == key:
                continue
            new_completed.append([p.copy() for p in chp])
            if i < len(oids_list):
                new_oids.append(list(oids_list[i]))
            else:
                new_oids.append(list(range(len(chp))))
        if len(old_pts) >= 2:
            new_completed.append(old_pts)
            if len(old_oids) != len(old_pts):
                old_oids = list(range(len(old_pts)))
            new_oids.append(old_oids)
        self.spine_chains_pts = new_completed
        self._spine_chains_origin_ids = new_oids
        self.spine_points = pts
        self._spine_origin_ids = oids
        self.selected = set()
        self.active_handle = None
        self._spine_last_add_idx = len(pts) - 1 if pts else None
        self._spine_placing_new_chain = True
        # New place chain has no deform bez yet
        self.bez = []
        self.point_modes = []


    def _spine_mirror_transform_orientation(self, context, obj):
        """Return Blender's current transform-orientation axes expressed in object-local space."""
        try:
            slots = context.scene.tool_settings.transform_orientation_slots
            typ = str(slots[0].type or 'GLOBAL') if slots else 'GLOBAL'
        except Exception:
            typ = 'GLOBAL'
        try:
            obj_inv_rot = obj.matrix_world.to_3x3().inverted()
        except Exception:
            obj_inv_rot = Matrix.Identity(3)
        if typ == 'LOCAL':
            return Matrix.Identity(3), typ
        if typ == 'GLOBAL':
            return obj_inv_rot, typ
        if typ == 'CURSOR':
            try:
                cmat = context.scene.cursor.rotation_euler.to_matrix()
                return obj_inv_rot @ cmat, typ
            except Exception:
                return Matrix.Identity(3), typ
        if typ == 'VIEW':
            try:
                rv3d = context.region_data
                if rv3d is not None:
                    return obj_inv_rot @ rv3d.view_rotation.to_matrix(), typ
            except Exception:
                pass
        # NORMAL / GIMBAL / custom orientations: the exact mesh-normal basis is
        # ambiguous for a chain pair, so local axes are the safest deterministic fallback.
        return Matrix.Identity(3), typ

    def _spine_mirror_pivot_local(self, context, obj, points):
        """Use Blender's current Transform Pivot Point, with paired-chain points as selection."""
        pts = [p.copy() for p in (points or [])]
        if not pts:
            return Vector((0, 0, 0))
        try:
            pivot_mode = str(context.scene.tool_settings.transform_pivot_point or 'MEDIAN_POINT')
        except Exception:
            pivot_mode = 'MEDIAN_POINT'
        if pivot_mode == 'CURSOR':
            try:
                return obj.matrix_world.inverted() @ context.scene.cursor.location
            except Exception:
                pass
        if pivot_mode == 'ACTIVE_ELEMENT':
            ac = int(getattr(self, 'active_chain', 0) or 0)
            ah = getattr(self, 'active_handle', None)
            chains = getattr(self, 'spine_chains', None) or []
            if ah is not None and 0 <= ac < len(chains):
                bez = chains[ac].get('bez') or []
                if 0 <= int(ah) < len(bez):
                    return bez[int(ah)]['co'].copy()
        if pivot_mode == 'BOUNDING_BOX_CENTER':
            return Vector((
                0.5 * (min(p.x for p in pts) + max(p.x for p in pts)),
                0.5 * (min(p.y for p in pts) + max(p.y for p in pts)),
                0.5 * (min(p.z for p in pts) + max(p.z for p in pts)),
            ))
        # Blender's median point for the effective paired selection.
        return sum(pts, Vector((0, 0, 0))) / float(len(pts))

    def _spine_blender_mesh_symmetry_axes(self, context):
        """Read Blender's LIVE Edit Mesh Mirror toggles from the active object.

        In current Blender versions the X/Y/Z Mesh Symmetry buttons shown in the
        3D View Tool Header are properties of the active *Object*, not
        ToolSettings.  Reading ToolSettings here was the reason the mirror became
        permanently inactive in the previous versions.
        """
        objs = []
        seen = set()
        try:
            ob = context.object
            if ob is not None:
                objs.append(ob)
        except Exception:
            pass
        try:
            import bpy
            ob = bpy.context.object
            if ob is not None and id(ob) not in seen:
                objs.append(ob)
        except Exception:
            pass

        for ob in objs:
            vals = []
            found = False
            for axis, attr in (('X', 'use_mesh_mirror_x'),
                                ('Y', 'use_mesh_mirror_y'),
                                ('Z', 'use_mesh_mirror_z')):
                try:
                    val = getattr(ob, attr)
                except Exception:
                    continue
                found = True
                if bool(val):
                    vals.append(axis)
            if found:
                return vals
        return []

    def _spine_find_mirrored_chain(self, context, active_ci):
        """Find a geometrically mirrored partner for the active chain.

        The test is intentionally conservative: same controller count and a low
        all-point reflection error are required.  This keeps unrelated nearby
        chains from becoming coupled accidentally.
        """
        # Do not run the add-on mirror unless Blender's own Edit Mesh symmetry is enabled.
        blender_axes = self._spine_blender_mesh_symmetry_axes(context)
        if not blender_axes:
            return None
        chains = getattr(self, 'spine_chains', None) or []
        if not (0 <= int(active_ci) < len(chains)):
            return None
        src = chains[int(active_ci)]
        src_bez = src.get('bez') or []
        if len(src_bez) < 2:
            return None
        obj, _bm = self.get_obj_bm(context)
        if obj is None:
            return None

        # Blender Mesh Symmetry X/Y/Z are object-local axes.  Use ONLY the
        # axes currently enabled in Blender's Tool Header.
        enabled = self._spine_blender_mesh_symmetry_axes(context)
        if not enabled:
            return None
        local_axes = {'X': Vector((1, 0, 0)), 'Y': Vector((0, 1, 0)), 'Z': Vector((0, 0, 1))}
        axes = [(name, local_axes[name]) for name in enabled]
        orient_name = 'BLENDER_MESH_SYMMETRY'

        src_pts = [bp['co'].copy() for bp in src_bez]
        best = None
        for ci, ch in enumerate(chains):
            if ci == int(active_ci):
                continue
            bez = ch.get('bez') or []
            if len(bez) != len(src_pts):
                continue
            dst_pts = [bp['co'].copy() for bp in bez]
            combined = src_pts + dst_pts
            pivot = self._spine_mirror_pivot_local(context, obj, combined)
            scale = max(1e-5, max((p - pivot).length for p in combined))
            # Reject candidates that are effectively on top of the source.
            for axis_idx, (axis_name, normal) in enumerate(axes):
                def refl(p):
                    d = p - pivot
                    return p - 2.0 * d.dot(normal) * normal
                for rev in (False, True):
                    err2 = 0.0
                    maxerr = 0.0
                    for i, p in enumerate(src_pts):
                        j = (len(dst_pts) - 1 - i) if rev else i
                        e = (refl(p) - dst_pts[j]).length
                        err2 += e * e
                        maxerr = max(maxerr, e)
                    rms = math.sqrt(err2 / float(len(src_pts)))
                    # A few percent of the chain's extent is tolerated for manually
                    # placed chains; require both RMS and worst-point agreement.
                    if rms <= scale * 0.035 and maxerr <= scale * 0.09:
                        if best is None or rms < best[0]:
                            best = (rms, ci, axis_idx, axis_name, normal.copy(), pivot.copy(), rev, orient_name)
        if best is None:
            return None
        _rms, ci, axis_idx, axis_name, normal, pivot, rev, orient_name = best
        return {
            'chain': int(ci),
            'axis_index': int(axis_idx),
            'axis': axis_name,
            'normal': normal,
            'pivot': pivot,
            'reverse': bool(rev),
            'orientation': orient_name,
        }

    def _spine_prepare_mirror_drag(self, context, active_ci):
        """Prepare a controller-level mirrored pair for the current drag."""
        self._mirror_drag = None
        pair = self._spine_find_mirrored_chain(context, active_ci)
        if pair is None:
            return
        chains = getattr(self, 'spine_chains', None) or []
        pci = int(pair.get('chain', -1))
        if not (0 <= pci < len(chains)):
            return
        pbez = chains[pci].get('bez') or []
        if len(pbez) < 2:
            return
        ac = int(active_ci)
        selected = getattr(self, 'selected', set()) or set()
        targets = []
        for key in selected:
            if isinstance(key, tuple) and len(key) == 3:
                ci, ti, tp = int(key[0]), int(key[1]), key[2]
                if ci == ac and tp in ('co', 'hl', 'hr'):
                    targets.append((ti, tp))
            elif isinstance(key, tuple) and len(key) == 2:
                targets.append((int(key[0]), key[1]))
        if not targets:
            targets = [(int(getattr(self, 'active_handle', 0) or 0),
                        getattr(self, 'active_bez_part', 'co'))]
        clean, seen = [], set()
        for ti, tp in targets:
            if 0 <= ti < len(pbez) and tp in ('co', 'hl', 'hr') and (ti, tp) not in seen:
                seen.add((ti, tp)); clean.append((ti, tp))
        if not clean:
            return
        starts = {}
        for ti, tp in clean:
            pi = len(pbez) - 1 - ti if pair.get('reverse') else ti
            bp = pbez[pi]
            starts[(ti, tp)] = {
                'pi': pi,
                'co': bp['co'].copy(),
                'hl': bp.get('hl', bp['co']).copy(),
                'hr': bp.get('hr', bp['co']).copy(),
            }
        pair['starts'] = starts
        pair['source_chain'] = ac
        pair['target_parts'] = clean
        pair['partner_selected'] = any(
            any(isinstance(k, tuple) and len(k) == 3 and int(k[0]) == pci and
                int(k[1]) == st['pi'] and k[2] == tp for k in selected)
            for (_ti, tp), st in starts.items()
        )
        self._mirror_drag = pair

    @staticmethod
    def _spine_reflect_delta(delta, normal):
        """Reflect a translation vector across the detected mirror plane."""
        d = delta.copy()
        return d - 2.0 * d.dot(normal) * normal

    def _spine_sync_mirror_chain_live(self, context):
        """Live-sync only the corresponding controller/handles on the mirrored chain.

        The source chain remains authoritative.  Every update reflects the source
        Bezier point (co + both handles), so the partner never lags one event behind.
        """
        info = getattr(self, '_mirror_drag', None)
        if not info or info.get('partner_selected'):
            return False
        chains = getattr(self, 'spine_chains', None) or []
        ac = int(info.get('source_chain', -1))
        pci = int(info.get('chain', -1))
        if not (0 <= ac < len(chains) and 0 <= pci < len(chains)):
            return False
        src_bez = chains[ac].get('bez') or []
        dst_bez = chains[pci].get('bez') or []
        if not src_bez or len(src_bez) != len(dst_bez):
            return False
        normal = info.get('normal')
        pivot = info.get('pivot')
        if normal is None or pivot is None:
            return False
        rev = bool(info.get('reverse'))
        changed = False
        targets = info.get('starts') or {}
        for (ti, _tp), _st in targets.items():
            si = int(ti)
            di = len(dst_bez) - 1 - si if rev else si
            if not (0 <= si < len(src_bez) and 0 <= di < len(dst_bez)):
                continue
            src = src_bez[si]
            dst = dst_bez[di]
            def refl(p):
                d = p - pivot
                return p - 2.0 * d.dot(normal) * normal
            dst['co'] = refl(src['co'])
            # Keep the source point mode on the paired point.  AUTO handles on
            # the partner must be rebuilt from the partner's mirrored controller
            # positions; copying the source handle tips here would freeze them.
            sm = list(chains[ac].get('modes') or ['AUTO'] * len(src_bez))
            dm = list(chains[pci].get('modes') or ['AUTO'] * len(dst_bez))
            src_mode = sm[si] if si < len(sm) else 'AUTO'
            if si < len(sm) and di < len(dm):
                dm[di] = src_mode
                chains[pci]['modes'] = dm
            if src_mode != 'AUTO':
                dst['hl'] = refl(src.get('hl', src['co']))
                dst['hr'] = refl(src.get('hr', src['co']))
            changed = True
        if changed:
            # Rebuild the mirrored chain's AUTO handles immediately from its
            # current mirrored controller positions.  This is intentionally done
            # inside the live mirror sync so Edit Place gets the same per-event
            # AUTO update as Deform.
            try:
                dst_ch = chains[pci]
                dst_modes = list(dst_ch.get('modes') or ['AUTO'] * len(dst_bez))
                while len(dst_modes) < len(dst_bez):
                    dst_modes.append('AUTO')
                old_bez, old_modes = self.bez, self.point_modes
                self.bez = dst_bez
                self.point_modes = dst_modes
                if len(self.bez) >= 2:
                    self.rebuild_auto_handles(interior=True)
                dst_ch['bez'] = self.bez
                dst_ch['modes'] = self.point_modes
                self.bez, self.point_modes = old_bez, old_modes
            except Exception:
                try:
                    self.bez, self.point_modes = old_bez, old_modes
                except Exception:
                    pass
            info['mesh_dirty'] = True
        return changed

    def _spine_apply_mirror_drag(self, context, delta, part='co'):
        """Mirror only the corresponding controller/handle, never the whole chain."""
        info = getattr(self, '_mirror_drag', None)
        if not info or info.get('partner_selected'):
            return
        chains = getattr(self, 'spine_chains', None) or []
        pci = int(info.get('chain', -1))
        if not (0 <= pci < len(chains)):
            return
        pbez = chains[pci].get('bez') or []
        starts = info.get('starts') or {}
        dmir = self._spine_reflect_delta(delta, info['normal'])
        for (_ti, tp), st in starts.items():
            if tp != part:
                continue
            pi = int(st['pi'])
            if not (0 <= pi < len(pbez)):
                continue
            bp = pbez[pi]
            if part in ('hl', 'hr'):
                bp[part] = st[part] + dmir
                modes = list(chains[pci].get('modes') or ['AUTO'] * len(pbez))
                if pi < len(modes) and modes[pi] == 'AUTO' and pi not in (0, len(pbez) - 1):
                    modes[pi] = 'ALIGNED'
                    chains[pci]['modes'] = modes
                if pi not in (0, len(pbez) - 1) and pi < len(modes) and modes[pi] == 'ALIGNED':
                    other = 'hl' if part == 'hr' else 'hr'
                    off = bp[part] - bp['co']
                    bp[other] = bp['co'] - off
            else:
                bp['co'] = st['co'] + dmir
                bp['hl'] = st['hl'] + dmir
                bp['hr'] = st['hr'] + dmir
        info['mesh_dirty'] = True

    def _spine_start_place_drag(self, context, event, index, part='co'):
        """Start dragging place-mode controllers / handle tips."""
        obj, _ = self.get_obj_bm(context)
        edit = bool(getattr(self, '_spine_edit_place', False))
        bez = getattr(self, 'bez', None)
        # Sync points from bez when editing
        if edit and bez and len(bez) >= 2:
            self.spine_points = [bp['co'].copy() for bp in bez]
        if obj is None or not self.spine_points or index < 0 or index >= len(self.spine_points):
            return
        self.active_handle = index
        self.active_bez_part = part
        self.dragging = True
        self.constraint_axis = None
        if edit and bez and 0 <= index < len(bez) and part in ('hl', 'hr', 'co'):
            # AUTO tip drag → ALIGNED (same as Deform)
            n = len(bez)
            if not hasattr(self, 'point_modes') or len(self.point_modes) != n:
                ch_modes = None
                chains = getattr(self, 'spine_chains', None) or []
                ac = int(getattr(self, 'active_chain', 0) or 0)
                if 0 <= ac < len(chains):
                    ch_modes = chains[ac].get('modes')
                self.point_modes = list(ch_modes) if ch_modes and len(ch_modes) == n else ['AUTO'] * n
            if part in ('hl', 'hr') and self.point_modes[index] == 'AUTO':
                self.point_modes[index] = 'ALIGNED'
            self.drag_start_handle = bez[index][part].copy()
        else:
            part = 'co'
            self.active_bez_part = 'co'
            self.drag_start_handle = self.spine_points[index].copy()
        self.drag_start_mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        # Automatically couple geometrically mirrored chains for Grab.
        # Detection uses Blender's current pivot/orientation settings.
        if edit and getattr(self, 'spine_chains', None):
            self._spine_prepare_mirror_drag(context, getattr(self, 'active_chain', 0) or 0)
        else:
            self._mirror_drag = None
        self.drag_start_sel = {}
        self.drag_start_bez_sel = {}
        self.drag_start_tip_sel = {}  # (ti, 'hl'|'hr') -> Vector
        keys = set(getattr(self, 'selected', set())) | {(index, part)}
        for key in keys:
            if len(key) == 3:
                ti, tp = key[1], key[2]
            else:
                ti, tp = key[0], key[1]
            if tp == 'co' and 0 <= ti < len(self.spine_points):
                self.drag_start_sel[ti] = self.spine_points[ti].copy()
                if edit and bez and 0 <= ti < len(bez):
                    bp = bez[ti]
                    self.drag_start_bez_sel[ti] = {
                        'co': bp['co'].copy(),
                        'hl': bp['hl'].copy(),
                        'hr': bp['hr'].copy(),
                    }
            elif tp in ('hl', 'hr') and edit and bez and 0 <= ti < len(bez):
                self.drag_start_tip_sel[(ti, tp)] = bez[ti][tp].copy()
        region = context.region
        rv3d = context.region_data
        wco = obj.matrix_world @ self.drag_start_handle
        self.drag_plane_point = wco.copy()
        self.drag_plane_normal = rv3d.view_rotation @ Vector((0, 0, 1))
        inv = obj.matrix_world.inverted()
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        denom = view_vec.dot(self.drag_plane_normal)
        if abs(denom) > 1e-8:
            t = (self.drag_plane_point - ray_origin).dot(self.drag_plane_normal) / denom
            self.drag_start_mouse_local = inv @ (ray_origin + view_vec * t)
        else:
            self.drag_start_mouse_local = self.drag_start_handle.copy()

    def _spine_update_place_drag(self, context, event):
        if not self.dragging or self.active_handle is None:
            return
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        inv = obj.matrix_world.inverted()
        nrm = getattr(self, 'drag_plane_normal', None)
        ppt = getattr(self, 'drag_plane_point', None)
        if nrm is None or ppt is None:
            return
        denom = view_vec.dot(nrm)
        if abs(denom) < 1e-8:
            return
        t = (ppt - ray_origin).dot(nrm) / denom
        local = inv @ (ray_origin + view_vec * t)
        start_mouse = getattr(self, 'drag_start_mouse_local', None)
        start = getattr(self, 'drag_start_handle', None)
        if start is None:
            return
        # Relative grab on view plane
        if start_mouse is None:
            candidate = local.copy()
        else:
            candidate = start + (local - start_mouse)

        part = getattr(self, 'active_bez_part', 'co')
        ts = context.tool_settings
        want_snap = (
            bool(getattr(ts, 'use_snap', False))
            or bool(getattr(event, 'ctrl', False))
            or bool(getattr(self, '_place_in_volume', False))
        )

        # Snap: absolute target for the active element, then delta for multi-select.
        # Handle tips use the same mouse-local snap path as controllers, including
        # Volume snapping, so they can be placed directly on/inside a mesh volume.
        if part == 'co' and want_snap:
            ml = None
            try:
                # Temporary: protect drag_start_handle
                saved_dsh = getattr(self, 'drag_start_handle', None)
                ml = self._spine_mouse_local(context, event)
                if saved_dsh is not None:
                    self.drag_start_handle = saved_dsh
            except Exception:
                ml = None
            if ml is not None:
                candidate = ml
            else:
                try:
                    saved_dsh = getattr(self, 'drag_start_handle', None)
                    candidate = self.snap_local(context, obj, candidate, force=True)
                    if saved_dsh is not None:
                        self.drag_start_handle = saved_dsh
                except Exception:
                    pass

        # Handle-tip snapping: use the exact same snap target computation as the
        # controller (including VOLUME), then convert it to a relative delta so
        # multi-selected handle tips preserve their offsets.
        if part in ('hl', 'hr') and want_snap:
            try:
                ml = self._spine_mouse_local(context, event)
                if ml is not None:
                    candidate = ml
            except Exception:
                pass

        dlt = candidate - start

        # axis lock (respects Blender transform orientation: Global/Local/Normal/View/…)
        ax = getattr(self, 'constraint_axis', None)
        if ax:
            candidate = self._apply_axis_constraint(context, obj, start, candidate, ax)
            dlt = candidate - start

        st = getattr(self, 'drag_start_sel', {}) or {}
        edit = bool(getattr(self, '_spine_edit_place', False))
        bez = getattr(self, 'bez', None) if edit else None

        if part in ('hl', 'hr') and edit and bez and 0 <= self.active_handle < len(bez):
            # Drag selected handle tips (multi-select from Ctrl+Shift box)
            if not hasattr(self, 'point_modes') or len(self.point_modes) != len(bez):
                self.point_modes = ['AUTO'] * len(bez)
            tip_starts = getattr(self, 'drag_start_tip_sel', None) or {}
            if not tip_starts:
                tip_starts = {(self.active_handle, part): start.copy()}
            for (ti, tp), tip0 in tip_starts.items():
                if not (0 <= ti < len(bez)) or tp not in ('hl', 'hr'):
                    continue
                bp = bez[ti]
                cur_mode = self.point_modes[ti] if ti < len(self.point_modes) else 'AUTO'
                if getattr(event, 'alt', False):
                    self.point_modes[ti] = 'FREE'
                elif cur_mode == 'AUTO':
                    self.point_modes[ti] = 'ALIGNED'
                mode = self.point_modes[ti]
                bp[tp] = tip0 + dlt
                if mode == 'ALIGNED' and ti not in (0, len(bez) - 1):
                    other = 'hl' if tp == 'hr' else 'hr'
                    off = bp[tp] - bp['co']
                    bp[other] = bp['co'] - off
                if ti == 0:
                    bp['hl'] = bp['co'].copy()
                if ti == len(bez) - 1:
                    bp['hr'] = bp['co'].copy()
                if 0 <= ti < len(self.spine_points):
                    self.spine_points[ti] = bp['co'].copy()
        else:
            # Drag controllers (co)
            if st:
                for hi, start_co in st.items():
                    if 0 <= hi < len(self.spine_points):
                        self.spine_points[hi] = start_co + dlt
                        if edit and bez and 0 <= hi < len(bez):
                            bp = bez[hi]
                            # translate handles with controller
                            prev = getattr(self, 'drag_start_bez_sel', {}).get(hi)
                            if prev:
                                bp['co'] = start_co + dlt
                                bp['hl'] = prev['hl'] + dlt
                                bp['hr'] = prev['hr'] + dlt
                            else:
                                off_l = bp['hl'] - bp['co']
                                off_r = bp['hr'] - bp['co']
                                bp['co'] = start_co + dlt
                                bp['hl'] = bp['co'] + off_l
                                bp['hr'] = bp['co'] + off_r
            elif 0 <= self.active_handle < len(self.spine_points):
                hi = self.active_handle
                self.spine_points[hi] = start + dlt
                if edit and bez and 0 <= hi < len(bez):
                    bp = bez[hi]
                    prev = getattr(self, 'drag_start_bez_sel', {}).get(hi)
                    if prev:
                        bp['co'] = start + dlt
                        bp['hl'] = prev['hl'] + dlt
                        bp['hr'] = prev['hr'] + dlt
                    else:
                        off_l = bp['hl'] - bp['co']
                        off_r = bp['hr'] - bp['co']
                        bp['co'] = start + dlt
                        bp['hl'] = bp['co'] + off_l
                        bp['hr'] = bp['co'] + off_r
        # Mirror paired chain in real time, if one was detected at drag start.
        if getattr(self, '_mirror_drag', None):
            # Source is authoritative: reflect its live Bezier point so controller +
            # both handles stay perfectly synchronized on every mouse event.
            self._spine_sync_mirror_chain_live(context)

        # Edit Place: AUTO handles follow controller positions live while dragging.
        # ALIGNED/FREE handles keep their user-edited positions.
        if edit and getattr(self, 'bez', None) and part == 'co':
            try:
                self.rebuild_auto_handles()
            except Exception:
                pass

        # Keep place points + chain data in sync while editing
        if edit and getattr(self, 'bez', None):
            self.spine_points = [bp['co'].copy() for bp in self.bez]
            try:
                # Ensure chain dict has current modes list (FREE/ALIGNED)
                chains = getattr(self, 'spine_chains', None) or []
                ac = int(getattr(self, 'active_chain', 0) or 0)
                if chains and 0 <= ac < len(chains) and getattr(self, 'point_modes', None):
                    chains[ac]['modes'] = self.point_modes
                self._spine_store_active_chain()
            except Exception:
                pass

    def _spine_pick_any_chain_controller(self, context, event, pixel_dist=22.0):
        """Pick controller. Returns (key, index) or None.
        key: 'current' | int (spine_chains_pts) | ('chain', ci) for spine_chains
        """
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return None
        region = context.region
        rv3d = context.region_data
        mx, my = event.mouse_region_x, event.mouse_region_y
        best, best_d = None, pixel_dist
        mw = obj.matrix_world

        def consider(co, key, i):
            nonlocal best, best_d
            sc = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ co)
            if sc is None:
                return
            dist = math.hypot(sc.x - mx, sc.y - my)
            if dist < best_d:
                best_d = dist
                best = (key, i)

        # Active place points
        for i, co in enumerate(getattr(self, 'spine_points', None) or []):
            consider(co, 'current', i)

        # New place chains (not yet bound)
        for ci, pts in enumerate(getattr(self, 'spine_chains_pts', None) or []):
            for i, co in enumerate(pts):
                consider(co, ci, i)

        # Edit Place: all deform spine_chains (full bez cos)
        if getattr(self, '_spine_edit_place', False):
            for ci, ch in enumerate(getattr(self, 'spine_chains', None) or []):
                bez = ch.get('bez') or []
                for i, bp in enumerate(bez):
                    consider(bp['co'], ('chain', ci), i)

        return best


    def _spine_pick_place_handle(self, context, event, pixel_dist=16.0):
        """Pick controller/handle tip in Edit Place across ALL spine_chains.
        Returns (key, idx, part) where key is ('chain', ci) or 'current'.
        """
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return None
        region = context.region
        rv3d = context.region_data
        mx, my = event.mouse_region_x, event.mouse_region_y
        best, best_d = None, float(pixel_dist)
        mw = obj.matrix_world

        def consider(co, key, i, part):
            nonlocal best, best_d
            sc = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ co)
            if sc is None:
                return
            dist = math.hypot(sc.x - mx, sc.y - my)
            if part != 'co':
                dist *= 0.85
            if dist < best_d:
                best_d = dist
                best = (key, i, part)

        def scan_bez(bez, key):
            if not bez:
                return
            n = len(bez)
            for i, bp in enumerate(bez):
                parts = ['co']
                if i == 0:
                    parts.append('hr')
                elif i == n - 1:
                    parts.append('hl')
                else:
                    parts.extend(['hl', 'hr'])
                for part in parts:
                    if part != 'co' and (bp[part] - bp['co']).length < 1e-8:
                        continue
                    consider(bp[part], key, i, part)

        # Active bez
        if getattr(self, 'bez', None) and len(self.bez) >= 2:
            ac = int(getattr(self, 'active_chain', 0) or 0)
            if getattr(self, '_spine_edit_place', False) and getattr(self, 'spine_chains', None):
                scan_bez(self.bez, ('chain', ac))
            else:
                scan_bez(self.bez, 'current')

        # Other deform chains
        if getattr(self, '_spine_edit_place', False):
            ac = int(getattr(self, 'active_chain', 0) or 0)
            for ci, ch in enumerate(getattr(self, 'spine_chains', None) or []):
                if ci == ac:
                    continue
                bez = ch.get('bez') or []
                if len(bez) >= 2:
                    scan_bez(bez, ('chain', ci))

        return best

    def _spine_pick_controller(self, context, event, pixel_dist=16.0):
        obj, _ = self.get_obj_bm(context)
        if obj is None or not self.spine_points:
            return None
        region = context.region
        rv3d = context.region_data
        mx, my = event.mouse_region_x, event.mouse_region_y
        best, best_d = None, pixel_dist
        for i, co in enumerate(self.spine_points):
            sc = view3d_utils.location_3d_to_region_2d(region, rv3d, obj.matrix_world @ co)
            if sc is None:
                continue
            dist = math.hypot(sc.x - mx, sc.y - my)
            if dist < best_d:
                best_d = dist
                best = i
        return best

    def _spine_pick_segment(self, context, event, pixel_dist=14.0):
        """Nearest green-chain segment under cursor. Returns (seg_index, local_point) or None.
        seg_index is the start controller index (insert at seg_index+1)."""
        obj, _ = self.get_obj_bm(context)
        pts = getattr(self, 'spine_points', None) or []
        if obj is None or len(pts) < 2:
            return None
        region = context.region
        rv3d = context.region_data
        mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
        mw = obj.matrix_world
        best_i, best_d, best_local = None, pixel_dist, None
        for i in range(len(pts) - 1):
            a = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ pts[i])
            b = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ pts[i + 1])
            if a is None or b is None:
                continue
            abx, aby = b.x - a.x, b.y - a.y
            lab2 = abx * abx + aby * aby
            if lab2 < 1e-8:
                u, px, py = 0.0, a.x, a.y
            else:
                u = max(0.0, min(1.0, ((mx - a.x) * abx + (my - a.y) * aby) / lab2))
                px = a.x + abx * u
                py = a.y + aby * u
            # Prefer interior of segment (not on the endpoints themselves)
            if u < 0.05 or u > 0.95:
                continue
            dist = math.hypot(mx - px, my - py)
            if dist < best_d:
                best_d = dist
                best_i = i
                best_local = pts[i].lerp(pts[i + 1], u)
        if best_i is None:
            return None
        return best_i, best_local


    def _spine_close_chain(self, context):
        """Shift+Enter: finalize current chain and start a new one (fresh place, like first open)."""
        # If Edit Place has a live bez, sync controller positions from it first
        try:
            if getattr(self, '_spine_edit_place', False) and getattr(self, 'bez', None) and len(self.bez) >= 2:
                self.spine_points = [bp['co'].copy() for bp in self.bez]
                # Persist active chain curve into spine_chains if still linked
                chains = getattr(self, 'spine_chains', None) or []
                ac = int(getattr(self, 'active_chain', 0) or 0)
                if chains and 0 <= ac < len(chains):
                    chains[ac]['bez'] = self.bez
                    if getattr(self, 'point_modes', None):
                        chains[ac]['modes'] = self.point_modes
        except Exception:
            pass
        if len(getattr(self, 'spine_points', []) or []) < 2:
            self.report({'WARNING'}, "Need at least 2 controllers to close a chain")
            return False
        self._spine_push_undo(context)
        edit = bool(getattr(self, '_spine_edit_place', False))
        was_new = bool(getattr(self, '_spine_placing_new_chain', False))
        # Existing deform chain already lives in spine_chains (with modes).
        # Only stash into spine_chains_pts when this was a brand-new place chain.
        if (not edit) or was_new:
            if not hasattr(self, 'spine_chains_pts') or self.spine_chains_pts is None:
                self.spine_chains_pts = []
            self.spine_chains_pts.append([p.copy() for p in self.spine_points])
            oids = list(getattr(self, '_spine_origin_ids', None) or [])
            if len(oids) != len(self.spine_points):
                oids = list(range(len(self.spine_points)))
            if not hasattr(self, '_spine_chains_origin_ids') or self._spine_chains_origin_ids is None:
                self._spine_chains_origin_ids = []
            self._spine_chains_origin_ids.append(oids)
        else:
            # Closing an existing deform chain after edit — already stored above
            pass
        # Fresh empty chain for next placement
        self.spine_points = []
        self._spine_origin_ids = []
        self._spine_last_add_idx = None
        self.selected = set()
        self.active_handle = None
        self.dragging = False
        self.box_selecting = False
        self._spine_placing_new_chain = True
        # Detach live bez without writing empty into spine_chains (store is guarded)
        self.bez = []
        self.rest_bez = []
        self.point_modes = []
        self.handle_params = []
        self.spine_tilt = []
        self.spine_radius = []
        n_existing = len(getattr(self, 'spine_chains', None) or [])
        n_new = len(getattr(self, 'spine_chains_pts', None) or [])
        self.report({'INFO'}, f"Saved — existing {n_existing}, new pending {n_new}  |  click to place next  |  Enter: Rebind")
        context.area.tag_redraw()
        return True

    def _spine_build_chain_from_pts(self, cos):
        """Build bez/rest/modes/params from controller positions."""
        # Controller positions can occasionally arrive from the modal placement
        # path as plain coordinate sequences instead of mathutils.Vector objects.
        # Normalize them here before any geometry helper (especially Vector.dot)
        # is allowed to consume them.
        _cos = []
        for _p in (cos or []):
            try:
                _cos.append(_p.copy() if isinstance(_p, Vector) else Vector(_p))
            except Exception:
                continue
        cos = _cos
        ok, center, radius, nrm = detect_circular_arc(cos)
        arc_center = center if ok else None
        arc_normal = nrm if ok else None
        bez = make_bezier_points(cos, poly=cos, center=arc_center, normal=arc_normal)
        old_bez = getattr(self, 'bez', None)
        old_modes = getattr(self, 'point_modes', None)
        self.bez = bez
        self.point_modes = ['AUTO'] * len(bez)
        self.rebuild_auto_handles()
        bez = copy_bezier_points(self.bez)
        modes = list(self.point_modes)
        self.bez = old_bez
        self.point_modes = old_modes if old_modes is not None else []
        rest_bez = copy_bezier_points(bez)
        cos2 = [bp['co'].copy() for bp in rest_bez]
        lengths = [0.0]
        for i in range(1, len(cos2)):
            lengths.append(lengths[-1] + (cos2[i] - cos2[i - 1]).length)
        total = lengths[-1] if lengths[-1] > 1e-12 else 1.0
        hparams = [L / total for L in lengths]
        if hparams:
            hparams[0], hparams[-1] = 0.0, 1.0
        return {
            'bez': bez,
            'rest_bez': rest_bez,
            'modes': modes,
            'tilt': [0.0] * len(bez),
            'radius': [1.0] * len(bez),
            'handle_params': hparams,
            'bind': [],
            'influence': 0.1,  # legacy max (compat)
            'point_influence': [0.1] * len(bez),
            'point_influence_default': [0.1] * len(bez),
            'point_inf_falloff': ['CONSTANT'] * len(bez),
            'inf_falloff': 'CONSTANT',
            'in_front': True,
        }

    def _spine_load_active_chain(self):
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return
        i = int(getattr(self, 'active_chain', 0) or 0) % len(chains)
        self.active_chain = i
        ch = chains[i]
        self.bez = ch['bez']
        self.rest_bez = ch['rest_bez']
        self.point_modes = ch['modes']
        self.spine_tilt = ch['tilt']
        self.spine_radius = ch['radius']
        self.handle_params = ch.get('handle_params') or []
        self.spine_bind = ch['bind']
        n = len(ch.get('bez') or [])
        # Per-controller radius + falloff (always pad/remember)
        legacy = float(ch.get('influence') or 0.1)
        self.point_influence = ensure_point_influence(n, ch.get('point_influence'), default=legacy)
        ch['point_influence'] = self.point_influence
        # Persistent reset baseline for per-controller influence radius.
        # Newer chains store the original bind-time radius; legacy chains fall
        # back to their current values rather than inventing a new scale.
        ch['point_influence_default'] = ensure_point_influence(
            n, ch.get('point_influence_default'), default=(ch['point_influence'][0] if ch.get('point_influence') else legacy)
        )
        self.spine_influence = max(self.point_influence) if self.point_influence else legacy
        ch['influence'] = self.spine_influence
        self.point_inf_falloff = ensure_point_inf_falloff(
            n, ch.get('point_inf_falloff'),
            default=ch.get('inf_falloff') or 'CONSTANT',
        )
        ch['point_inf_falloff'] = self.point_inf_falloff
        # Restore the actual Shrink/Inflate/Tilt interpolation falloff used by
        # this chain. Older sessions have no field and safely fall back to Smooth.
        attr_interp = str(ch.get('attr_interp') or 'SMOOTH').upper()
        if attr_interp not in _ATTR_INTERP_ORDER:
            attr_interp = 'SMOOTH'
        self.spine_attr_interp = attr_interp
        ch['attr_interp'] = attr_interp
        self.influence_falloff = 'CONSTANT'
        self.spine_in_front = bool(ch.get('in_front', True))
        if 'in_front' not in ch:
            ch['in_front'] = True
        self.spine_points = [bp['co'].copy() for bp in self.bez]
        if not (
            getattr(self, 'dragging', False)
            or getattr(self, '_xform_mode', None)
            or getattr(self, '_attr_mode', None)
        ):
            try:
                self._spine_highlight_active_vg()
            except Exception:
                pass

    def _spine_highlight_active_vg(self, context=None):
        """Select this chain's BH_Spine* group in the mesh Data panel."""
        ctx = context or bpy.context
        obj = getattr(ctx, 'object', None) if ctx else None
        if obj is None or getattr(obj, 'type', None) != 'MESH':
            return False
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return False
        i = int(getattr(self, 'active_chain', 0) or 0)
        if i < 0 or i >= len(chains):
            return False
        ch = chains[i]
        name = ch.get('vg_name') or _vdh_spine_vg_name(i)
        if not name:
            return False
        ch['vg_name'] = name
        vg = obj.vertex_groups.get(name)
        if vg is None:
            vg = _vdh_ensure_vertex_group(obj, name)
        if vg is None:
            return False
        if obj.vertex_groups.active_index != vg.index:
            obj.vertex_groups.active_index = vg.index
        return True

    def _spine_store_active_chain(self):
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return
        # While placing a brand-new chain, do not clobber an existing deform chain
        if getattr(self, '_spine_placing_new_chain', False):
            return
        i = int(getattr(self, 'active_chain', 0) or 0) % len(chains)
        ch = chains[i]
        # Keep list identity so edits during drag stay on the same objects.
        # Never write empty/invalid bez — that wipes handle types after Shift+Enter.
        bez = getattr(self, 'bez', None)
        if not bez or len(bez) < 2:
            return
        ch['bez'] = bez
        if self.rest_bez is not None and len(getattr(self, 'rest_bez', []) or []) >= 2:
            ch['rest_bez'] = self.rest_bez
        if self.point_modes is not None and len(self.point_modes) >= 2:
            ch['modes'] = self.point_modes
        if getattr(self, 'spine_tilt', None) is not None:
            ch['tilt'] = self.spine_tilt
        if getattr(self, 'spine_radius', None) is not None:
            ch['radius'] = self.spine_radius
        ch['handle_params'] = getattr(self, 'handle_params', None)
        # keep bind/influence if already present
        if getattr(self, 'spine_bind', None) is not None:
            ch['bind'] = self.spine_bind
        n = len(ch.get('bez') or getattr(self, 'bez', None) or [])
        if getattr(self, 'point_inf_falloff', None) is not None:
            ch['point_inf_falloff'] = ensure_point_inf_falloff(n, self.point_inf_falloff, default='CONSTANT')
        # Persist the falloff that controls Shrink/Inflate and Tilt between
        # tool activations and chain switches.
        attr_interp = str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper()
        if attr_interp not in _ATTR_INTERP_ORDER:
            attr_interp = 'SMOOTH'
        ch['attr_interp'] = attr_interp
        if getattr(self, 'point_influence', None) is not None:
            ch['point_influence'] = ensure_point_influence(n, self.point_influence, default=0.1)
            ch['point_influence_default'] = ensure_point_influence(
                n, ch.get('point_influence_default'),
                default=(ch['point_influence'][0] if ch['point_influence'] else 0.1),
            )
            ch['influence'] = max(ch['point_influence']) if ch['point_influence'] else 0.1
            self.spine_influence = ch['influence']
        elif hasattr(self, 'spine_influence'):
            ch['influence'] = self.spine_influence
        if hasattr(self, 'spine_in_front'):
            ch['in_front'] = bool(self.spine_in_front)
        if not ch.get('origin_ids') or len(ch.get('origin_ids') or []) != len(ch.get('bez') or []):
            n = len(ch.get('bez') or [])
            ch['origin_ids'] = list(range(n))


    def _spine_ensure_chain_ids(self):
        """Every chain gets a stable chain_id for reset matching."""
        for i, ch in enumerate(getattr(self, 'spine_chains', None) or []):
            if not ch.get('chain_id'):
                ch['chain_id'] = f"chain_{i}_{id(ch) & 0xFFFFFF:x}"

    def _spine_ensure_vg_names(self):
        """Give every chain a vertex-group name (BH_Spine, BH_Spine.001, …)."""
        for i, ch in enumerate(getattr(self, 'spine_chains', None) or []):
            ch['vg_name'] = _vdh_spine_vg_name(i, ch.get('vg_name'))

    def _spine_set_vg_weights(self, obj, bm, vg, vert_weights, default_others=None):
        """Write weights into a vertex group via bmesh deform layer.
        vert_weights: dict vidx -> weight. If default_others is a float, all
        other verts get that value (use 0.0 to clear).
        """
        if obj is None or bm is None or vg is None:
            return 0
        bm.verts.ensure_lookup_table()
        try:
            dl = bm.verts.layers.deform.verify()
        except Exception:
            return 0
        gi = vg.index
        n = 0
        for v in bm.verts:
            d = v[dl]
            if v.index in vert_weights:
                d[gi] = max(0.0, min(1.0, float(vert_weights[v.index])))
                n += 1
            elif default_others is not None:
                d[gi] = float(default_others)
        return n

    def _spine_item_radial_dist(self, item):
        """Distance used by influence: perpendicular to the bind tangent."""
        if item is None:
            return 0.0
        if len(item) > 4 and item[4] is not None:
            try:
                stored = float(item[4])
                if stored >= 0.0:
                    off = item[2] if len(item) > 2 else None
                    tan = item[3] if len(item) > 3 else None
                    if off is not None and tan is not None and getattr(tan, 'length', 0) > 1e-8:
                        tn = tan.normalized()
                        return (off - tn * off.dot(tn)).length
                    return stored
            except Exception:
                pass
        off = item[2] if len(item) > 2 else None
        if off is None:
            return 0.0
        tan = item[3] if len(item) > 3 else None
        if tan is not None and getattr(tan, 'length', 0) > 1e-8:
            tn = tan.normalized()
            return (off - tn * off.dot(tn)).length
        return off.length

    def _spine_envelope_list(self, ch, bm=None):
        """Per-bind-entry envelope (radius × falloff). Island-equalize only the envelope."""
        bind = ch.get('bind') or []
        bez = ch.get('bez') or []
        n_ctrl = len(bez)
        if not bind or n_ctrl < 1:
            return []
        legacy = float(ch.get('influence') or getattr(self, 'spine_influence', 0.1) or 0.1)
        pinf = ensure_point_influence(n_ctrl, ch.get('point_influence'), default=legacy)
        pfo = ensure_point_inf_falloff(
            n_ctrl, ch.get('point_inf_falloff'),
            default=ch.get('inf_falloff') or 'CONSTANT',
        )
        hparams = list(ch.get('handle_params') or [])
        if len(hparams) != n_ctrl:
            hparams = [i / max(1, n_ctrl - 1) for i in range(n_ctrl)]
            if hparams:
                hparams[0], hparams[-1] = 0.0, 1.0
        env = []
        for item in bind:
            dist = self._spine_item_radial_dist(item)
            cw = deform_weights(float(item[1]), hparams)
            acc = 0.0
            wsum = 0.0
            for j, wj in enumerate(cw):
                if wj <= 1e-8:
                    continue
                wsum += wj
                r_j = pinf[j] if j < len(pinf) else legacy
                fo_j = pfo[j] if j < len(pfo) else 'CONSTANT'
                acc += wj * spine_influence_weight(dist, r_j, fo_j)
            if wsum > 1e-8:
                acc /= wsum
            else:
                acc = spine_influence_weight(dist, pinf[0] if pinf else legacy, pfo[0] if pfo else 'CONSTANT')
            env.append(max(0.0, min(1.0, float(acc))))
        # Keep the true per-vertex radial falloff. Do not equalize connected
        # rings here: Weight Paint and Radius should combine smoothly per vertex.
        return env

    def _spine_group_list(self, obj, bm, ch):
        """Vertex-group multiplier per bind entry. Missing vert → 0."""
        bind = ch.get('bind') or []
        if not bind:
            return []
        gw = [1.0] * len(bind)
        vg_name = ch.get('vg_name') or ''
        vg = obj.vertex_groups.get(vg_name) if (obj is not None and vg_name) else None
        if vg is None or bm is None:
            return gw
        try:
            bm.verts.ensure_lookup_table()
            dl = bm.verts.layers.deform.verify()
            gi = vg.index
        except Exception:
            return [0.0] * len(bind)
        for i, item in enumerate(bind):
            vidx = int(item[0])
            w = 0.0
            if 0 <= vidx < len(bm.verts):
                dvert = bm.verts[vidx][dl]
                w = float(dvert[gi]) if gi in dvert else 0.0
            gw[i] = max(0.0, min(1.0, w))
        return gw

    def _spine_final_weight_list(self, obj, bm, ch):
        """Spine runtime weight = Blender Weight Paint × live Radius/Falloff envelope."""
        env = self._spine_envelope_list(ch, bm)
        gw = self._spine_group_list(obj, bm, ch)
        n = min(len(env), len(gw))
        return [max(0.0, min(1.0, float(gw[i]) * float(env[i]))) for i in range(n)]

    def _spine_absorb_vg_into_bind(self, obj, bm, ch, ci):
        """If the user added verts to this chain's group, bind them (don't steal other chains).
        New verts bump the nearest controller radius so group weight 1 can actually move."""
        if obj is None or bm is None or ch is None:
            return 0
        vg_name = ch.get('vg_name') or ''
        vg = obj.vertex_groups.get(vg_name) if vg_name else None
        rest = ch.get('rest_bez') or ch.get('bez')
        if vg is None or not rest or len(rest) < 2:
            return 0
        existing = {int(it[0]) for it in (ch.get('bind') or [])}
        # Do not permanently block a vertex just because an older bind says
        # that another chain owns it. Weight Paint is authoritative here: if
        # the active Spine group now has weight and the other Spine groups have
        # been reduced to zero, ownership must be allowed to migrate.
        other_chains = [
            (j, och) for j, och in enumerate(getattr(self, 'spine_chains', None) or [])
            if j != ci
        ]
        try:
            bm.verts.ensure_lookup_table()
            dl = bm.verts.layers.deform.verify()
            gi = vg.index
        except Exception:
            return 0
        samples = max(32, len(rest) * 16)
        pts = [eval_bezier_points(rest, s / samples) for s in range(samples + 1)]
        n_ctrl = len(ch.get('bez') or [])
        if n_ctrl < 1:
            return 0
        legacy = float(ch.get('influence') or 0.1) or 0.1
        pinf = ensure_point_influence(n_ctrl, ch.get('point_influence'), default=legacy)
        added = 0
        bind = list(ch.get('bind') or [])
        for v in bm.verts:
            vidx = v.index
            if vidx in existing:
                continue
            if gi not in v[dl]:
                continue
            try:
                gw = float(v[dl][gi])
            except Exception:
                continue
            if gw <= 1e-6:
                continue

            # Weight Paint is authoritative for Spine influence. Keep a bind
            # entry for every Spine chain that has a non-zero painted weight.
            # The runtime blends overlapping chains using their final weights,
            # so partial weights remain meaningful.
            rest_co = self.all_rest.get(vidx)
            if rest_co is None:
                rest_co = v.co.copy()
                self.all_rest[vidx] = rest_co.copy()
            best_t, best_d = 0.0, 1e18
            for s, p in enumerate(pts):
                d = (p - rest_co).length_squared
                if d < best_d:
                    best_d = d
                    best_t = s / samples
            best_d = math.sqrt(best_d)
            on = eval_bezier_points(rest, best_t)
            tan = bezier_chain_tangent(rest, best_t)
            offset = rest_co - on
            rad = self._spine_item_radial_dist((vidx, best_t, offset, tan, 0.0))
            bind.append((vidx, best_t, offset.copy(), tan.copy(), float(rad)))
            existing.add(vidx)
            added += 1
        if added:
            ch['bind'] = bind
            ch['point_influence'] = pinf
            ch['influence'] = max(pinf) if pinf else legacy
            if ci == int(getattr(self, 'active_chain', 0) or 0):
                self.point_influence = list(pinf)
                self.spine_influence = ch['influence']
        return added

    def _spine_chain_envelope_map(self, ch, bm=None):
        """Envelope weights keyed by vert index."""
        bind = ch.get('bind') or []
        env = self._spine_envelope_list(ch, bm)
        out = {}
        for item, w in zip(bind, env):
            out[int(item[0])] = w
        return out

    def _spine_sync_weight_groups(self, context, mode='SYNC'):
        """Create/update BH_Spine* vertex groups.

        SYNC: bound verts get 1.0 only if they have no weight yet (keep paint).
        RESET: bound verts → 1.0, others in that group → 0.
        FILL_ENV: bound verts → current influence envelope.
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return False
        self._spine_ensure_chain_ids()
        self._spine_ensure_vg_names()
        bm.verts.ensure_lookup_table()
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return False
        wrote = 0
        for i, ch in enumerate(chains):
            name = _vdh_spine_vg_name(i, ch.get('vg_name'))
            ch['vg_name'] = name
            vg = _vdh_ensure_vertex_group(obj, name)
            if vg is None:
                continue
            bound = {}
            if mode == 'FILL_ENV':
                bound = self._spine_chain_envelope_map(ch)
            else:
                for item in (ch.get('bind') or []):
                    bound[int(item[0])] = 1.0
            if mode == 'SYNC':
                # Seed 1.0 only when this group is still empty (first bind).
                # If the user already removed verts from the group, keep them out.
                has_any = False
                try:
                    dl = bm.verts.layers.deform.verify()
                    gi = vg.index
                    for v in bm.verts:
                        if gi in v[dl]:
                            has_any = True
                            break
                except Exception:
                    has_any = False
                if not has_any:
                    wrote += self._spine_set_vg_weights(obj, bm, vg, bound, default_others=None)
            else:
                wrote += self._spine_set_vg_weights(obj, bm, vg, bound, default_others=0.0)
            ch['_sw_key'] = None
            ch['_soft_w'] = None
            ch['_fw'] = None
        try:
            ac = int(getattr(self, 'active_chain', 0) or 0)
            if 0 <= ac < len(chains):
                vg = obj.vertex_groups.get(chains[ac].get('vg_name') or '')
                if vg is not None:
                    obj.vertex_groups.active_index = vg.index
        except Exception:
            pass
        try:
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        except Exception:
            pass
        return True

    def _spine_connected_bind_clusters(self, bm, bind, idxs):
        """Split bind-indices in idxs into topology-connected islands."""
        if not idxs:
            return []
        try:
            bm.verts.ensure_lookup_table()
        except Exception:
            return [list(idxs)]
        vset = {}
        for i in idxs:
            if 0 <= i < len(bind):
                vset[int(bind[i][0])] = i
        if len(vset) <= 1:
            return [list(idxs)]
        seen = set()
        clusters = []
        for start_v, start_i in vset.items():
            if start_v in seen or start_v >= len(bm.verts):
                continue
            stack = [start_v]
            seen.add(start_v)
            cluster = [start_i]
            while stack:
                vidx = stack.pop()
                v = bm.verts[vidx]
                for e in v.link_edges:
                    o = e.other_vert(v)
                    if o is None:
                        continue
                    oj = o.index
                    if oj in vset and oj not in seen:
                        seen.add(oj)
                        stack.append(oj)
                        cluster.append(vset[oj])
            clusters.append(cluster)
        return clusters or [list(idxs)]

    def _spine_merge_bind_rest(self, context, force_update_ids=None, overwrite_mesh=False):
        """Update first-bind rest store.

        - By default mesh_rest[vidx] is write-once (keeps pre-deform positions)
        - force_update_ids: rewrite controller snapshot for those chain_ids
        - overwrite_mesh=True: also refresh mesh_rest for those chains' verts
          (used after Edit Place confirm so Ctrl+R returns to this new baseline)
        """
        global _vdh_spine_bind_rest
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        if not getattr(self, 'spine_chains', None):
            return
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        self._spine_ensure_chain_ids()
        bm.verts.ensure_lookup_table()
        force_update_ids = set(force_update_ids or [])

        prev = _vdh_get_bind_rest(obj)
        prev_by_id = {}
        for s in (prev.get('chains') or []):
            cid = s.get('chain_id')
            if cid:
                prev_by_id[cid] = s
        # write-once mesh rest
        mesh_rest = dict(prev.get('mesh_rest') or {})

        new_saved = []
        seen_ids = set()
        for ch in self.spine_chains:
            cid = ch.get('chain_id')
            if not cid:
                continue
            bind_verts = [int(item[0]) for item in (ch.get('bind') or [])]
            is_new = cid not in prev_by_id
            should_write = is_new or (cid in force_update_ids)
            if should_write:
                entry = {
                    'chain_id': cid,
                    'bez': copy_bezier_points(ch['bez']),
                    'tilt': list(ch.get('tilt') or [0.0] * len(ch['bez'])),
                    'radius': list(ch.get('radius') or [1.0] * len(ch['bez'])),
                    'modes': list(ch.get('modes') or ['AUTO'] * len(ch['bez'])),
                    'origin_ids': list(ch.get('origin_ids') or list(range(len(ch['bez'])))),
                    'influence': float(ch.get('influence') or 0.0),
                    'point_influence': ensure_point_influence(
                        len(ch['bez']), ch.get('point_influence'),
                        default=float(ch.get('influence') or 0.1),
                    ),
                    'point_inf_falloff': ensure_point_inf_falloff(
                        len(ch['bez']), ch.get('point_inf_falloff'), default='CONSTANT'
                    ),
                    'bind_verts': bind_verts,
                    'vg_name': ch.get('vg_name') or '',
                }
                new_saved.append(entry)
                for vidx in bind_verts:
                    if vidx >= len(bm.verts):
                        continue
                    # write-once unless overwrite_mesh explicitly requested
                    if overwrite_mesh or vidx not in mesh_rest:
                        mesh_rest[vidx] = bm.verts[vidx].co.copy()
            else:
                entry = dict(prev_by_id[cid])
                # refresh bind_verts list for matching, but keep bez/tilt/radius/mesh
                entry['bind_verts'] = bind_verts or list(entry.get('bind_verts') or [])
                entry['chain_id'] = cid
                if ch.get('vg_name'):
                    entry['vg_name'] = ch.get('vg_name')
                new_saved.append(entry)
                for vidx in bind_verts:
                    if vidx not in mesh_rest and vidx < len(bm.verts):
                        mesh_rest[vidx] = bm.verts[vidx].co.copy()
            seen_ids.add(cid)

        # Keep any saved chains not currently loaded (shouldn't drop history)
        for cid, s in prev_by_id.items():
            if cid not in seen_ids:
                new_saved.append(s)

        entry = {
            'chains': new_saved,
            'mesh_rest': mesh_rest,
        }
        _vdh_set_bind_rest(obj, entry)
        try:
            _vdh_persist_bind_rest_to_mesh(obj, entry)
        except Exception:
            pass

    def _spine_match_saved_for_chain(self, ch, saved_list, saved_by_id):
        """Find first-bind snapshot for a live chain: chain_id, then vert overlap."""
        cid = ch.get('chain_id')
        if cid and cid in saved_by_id:
            return saved_by_id[cid]
        live = set(int(item[0]) for item in (ch.get('bind') or []))
        if not live:
            return None
        best, best_ov = None, 0
        for s in saved_list:
            stored = set(int(v) for v in (s.get('bind_verts') or []))
            if not stored:
                continue
            ov = len(live & stored)
            if ov > best_ov:
                best_ov = ov
                best = s
        # require meaningful overlap
        if best is not None and best_ov >= max(1, min(3, len(live) // 10)):
            return best
        return None


    def _spine_unify_bind_rings(self, bm, ch):
        """Force verts on the same cross-section loop to share one t and one influence dist.

        Root cause of irregular rings during deform: nearest-point bind gives each
        ring vert a slightly different t (and different radial dist → different
        falloff weight). Different frames shear the loop.
        """
        bind = list(ch.get('bind') or [])
        rest_bez = ch.get('rest_bez') or ch.get('bez')
        if not bind or not rest_bez or len(rest_bez) < 2:
            return
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        v2i = {}
        for i, item in enumerate(bind):
            v2i[int(item[0])] = i
        bound = set(v2i.keys())
        if len(bound) < 4:
            return

        # Ring-edge graph: edges mostly perpendicular to spine, similar t
        adj = {i: [] for i in bound}
        for e in bm.edges:
            i0, i1 = e.verts[0].index, e.verts[1].index
            if i0 not in bound or i1 not in bound:
                continue
            t0 = float(bind[v2i[i0]][1])
            t1 = float(bind[v2i[i1]][1])
            if abs(t0 - t1) > 0.1:
                continue
            t_mid = 0.5 * (t0 + t1)
            tan = bezier_chain_tangent(rest_bez, t_mid)
            if tan.length < 1e-12:
                continue
            d = e.verts[1].co - e.verts[0].co
            if d.length < 1e-12:
                continue
            if abs(d.normalized().dot(tan.normalized())) > 0.5:
                continue
            adj[i0].append(i1)
            adj[i1].append(i0)

        # Prefer 2 ring-neighbors (most perpendicular)
        clean = {}
        for i, nbs in adj.items():
            if len(nbs) <= 2:
                clean[i] = list(dict.fromkeys(nbs))
                continue
            t = float(bind[v2i[i]][1])
            tan = bezier_chain_tangent(rest_bez, t)
            if tan.length < 1e-12:
                clean[i] = nbs[:2]
                continue
            tan = tan.normalized()
            scored = []
            for j in nbs:
                d = bm.verts[j].co - bm.verts[i].co
                if d.length < 1e-12:
                    continue
                scored.append((1.0 - abs(d.normalized().dot(tan)), j))
            scored.sort(key=lambda x: -x[0])
            clean[i] = [j for _, j in scored[:2]]

        visited = set()
        for start in bound:
            if start in visited:
                continue
            if len(clean.get(start, [])) != 2:
                continue
            cycle = [start]
            prev, cur = start, clean[start][0]
            ok = True
            for _ in range(len(bound) + 2):
                if cur == start:
                    break
                if cur in visited:
                    ok = False
                    break
                cycle.append(cur)
                nbs = clean.get(cur, [])
                nxts = [n for n in nbs if n != prev]
                if len(nxts) != 1:
                    ok = False
                    break
                prev, cur = cur, nxts[0]
            else:
                ok = False
            if not (ok and cur == start and len(cycle) >= 4):
                continue
            for v in cycle:
                visited.add(v)
            ts = [float(bind[v2i[v]][1]) for v in cycle]
            if max(ts) - min(ts) > 0.15:
                continue  # not a cross-section
            # Shared parameter: median t (robust)
            ts_sorted = sorted(ts)
            t_uni = ts_sorted[len(ts_sorted) // 2]
            # Shared influence distance: max in ring so falloff weight matches
            dists = []
            for v in cycle:
                item = bind[v2i[v]]
                dists.append(float(item[4]) if len(item) > 4 else 0.0)
            dist_uni = max(dists) if dists else 0.0
            on = eval_bezier_points(rest_bez, t_uni)
            tan = bezier_chain_tangent(rest_bez, t_uni)
            for v in cycle:
                bi = v2i[v]
                if v >= len(bm.verts):
                    continue
                off = bm.verts[v].co - on
                bind[bi] = (v, t_uni, off.copy(), tan.copy(), dist_uni)
        ch['bind'] = bind

    def _spine_bind(self, context):
        """Bind mesh verts to all spine chains (nearest-chain assignment)."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return False
        chains_pts = [list(c) for c in (getattr(self, 'spine_chains_pts', None) or [])]
        if len(getattr(self, 'spine_points', []) or []) >= 2:
            chains_pts.append([p.copy() for p in self.spine_points])
        if not chains_pts:
            self.report({'WARNING'}, "Place at least 2 controllers")
            return False
        for i, c in enumerate(chains_pts):
            if len(c) < 2:
                self.report({'WARNING'}, f"Chain {i + 1} needs at least 2 controllers")
                return False
        bm.verts.ensure_lookup_table()
        built = [self._spine_build_chain_from_pts([p.copy() for p in c]) for c in chains_pts]

        kd_samples = []
        for ci, ch in enumerate(built):
            samples = max(48, len(ch['rest_bez']) * 24)
            pts = [eval_bezier_points(ch['rest_bez'], s / samples) for s in range(samples + 1)]
            kd_samples.append((ci, samples, pts))

        edge_lens = [e.calc_length() for e in bm.edges]
        avg_edge = (sum(edge_lens) / len(edge_lens)) if edge_lens else 0.01
        kd = KDTree(len(bm.verts))
        for v in bm.verts:
            kd.insert(v.co, v.index)
        kd.balance()
        for ch in built:
            influence = 0.0
            for bp in ch['rest_bez']:
                for _co, _idx, dist in kd.find_n(bp['co'], min(12, len(bm.verts))):
                    influence = max(influence, dist)
            inf = max(influence * 1.35, avg_edge * 4.0)
            ch['influence'] = inf
            n_ctrl = len(ch.get('bez') or [])
            ch['point_influence'] = [inf] * n_ctrl
            ch['point_influence_default'] = [inf] * n_ctrl
            ch['point_inf_falloff'] = ensure_point_inf_falloff(
                n_ctrl, ch.get('point_inf_falloff'), default='CONSTANT'
            )

        bound_total = 0
        for v in bm.verts:
            best_ci, best_t, best_d = -1, 0.0, 1e18
            for ci, samples, pts in kd_samples:
                for s, p in enumerate(pts):
                    d = (p - v.co).length_squared
                    if d < best_d:
                        best_d = d
                        best_t = s / samples
                        best_ci = ci
            best_d = math.sqrt(best_d)
            if best_ci < 0:
                continue
            ch = built[best_ci]
            if best_d > max(ch['influence'] * 1.5, avg_edge * 8.0):
                continue
            on = eval_bezier_points(ch['rest_bez'], best_t)
            tan = bezier_chain_tangent(ch['rest_bez'], best_t)
            offset = v.co - on
            rad = offset.length
            if tan.length > 1e-8:
                tn = tan.normalized()
                # Both operands are explicitly mathutils.Vector here.  Keep
                # this conversion local so malformed modal data cannot make
                # Enter -> Bind abort with Vector.dot(other).
                try:
                    offset_v = offset if isinstance(offset, Vector) else Vector(offset)
                    tn_v = tn if isinstance(tn, Vector) else Vector(tn)
                    rad = (offset_v - tn_v * offset_v.dot(tn_v)).length
                except Exception:
                    rad = offset.length
            ch['bind'].append((v.index, best_t, offset.copy(), tan.copy(), float(rad)))
            bound_total += 1

        for ch in built:
            for i, item in enumerate(list(ch['bind'])):
                vidx, t = item[0], float(item[1])
                if vidx >= len(bm.verts):
                    continue
                on = eval_bezier_points(ch['rest_bez'], t)
                tan = bezier_chain_tangent(ch['rest_bez'], t)
                offset = bm.verts[vidx].co - on
                ch['bind'][i] = (vidx, t, offset.copy(), tan.copy(), self._spine_item_radial_dist((vidx, t, offset, tan, 0.0)))
            # Same t + same influence dist per topological ring → no shear
            try:
                self._spine_unify_bind_rings(bm, ch)
            except Exception:
                pass

        # Attach origin_ids matching chains_pts order
        done_oids = list(getattr(self, '_spine_chains_origin_ids', None) or [])
        cur_oids = list(getattr(self, '_spine_origin_ids', None) or [])
        all_oids = list(done_oids)
        if len(getattr(self, 'spine_points', []) or []) >= 2:
            if len(cur_oids) != len(self.spine_points):
                cur_oids = list(range(len(self.spine_points)))
            all_oids.append(cur_oids)
        for i, ch in enumerate(built):
            n = len(ch['bez'])
            if i < len(all_oids) and len(all_oids[i]) == n:
                ch['origin_ids'] = list(all_oids[i])
            else:
                ch['origin_ids'] = list(range(n))

        # Remember whether this bind is finishing Edit Place.  In that case
        # returning to Deform must NOT auto-select the first controller.
        # Normal initial Place keeps its existing selection behaviour.
        _was_edit_place = bool(getattr(self, '_spine_edit_place', False))
        self.spine_chains = built
        self.spine_chains_pts = []  # clear place lists in deform
        self._spine_chains_origin_ids = []
        self._spine_origin_ids = []
        self._spine_edit_place = False
        self._spine_placing_new_chain = False
        self.active_chain = 0
        self._spine_load_active_chain()

        for v in bm.verts:
            v.select = False
        for e in bm.edges:
            e.select = False
        for f in bm.faces:
            f.select = False
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        self._lock_selection = []
        self.all_rest = {v.index: v.co.copy() for v in bm.verts}
        self._attr_mode = None
        self._attr_start_mouse = None
        self._attr_start_values = None
        self.tool_mode = 'SPINE_DEFORM'
        self.selected = set()
        # Keep the internal active handle for W/Spine operations, but when
        # leaving Edit Place do not select it.  Selection itself is what the
        # deform modal uses to start an automatic grab.
        self.active_handle = 0 if self.bez else None
        self.active_bez_part = 'co'
        if self.bez and not _was_edit_place:
            self.selected.add((0, 0, 'co'))  # (chain, idx, part)

        # Origin ids: which controllers come from first bind (None = inserted later)
        for ch in self.spine_chains:
            n = len(ch.get('bez') or [])
            if not ch.get('origin_ids') or len(ch['origin_ids']) != n:
                ch['origin_ids'] = list(range(n))
        for i, ch in enumerate(self.spine_chains):
            if not ch.get('chain_id'):
                ch['chain_id'] = f"chain_{i}_{id(ch) & 0xFFFFFF:x}"
        try:
            self._spine_sync_bound_vertex_group_ownership(obj, self.spine_chains)
        except Exception:
            pass
        try:
            self._spine_sync_weight_groups(context, mode='SYNC')
        except Exception:
            pass
        global _vdh_spine_bind_rest
        # Preserve any existing mesh_rest (write-once); only fill new verts
        prev = _vdh_get_bind_rest(obj)
        mesh_rest = dict(prev.get('mesh_rest') or {})
        for v in bm.verts:
            if v.index not in mesh_rest:
                mesh_rest[v.index] = v.co.copy()
        entry_br = {
            'chains': [
                {
                    'chain_id': ch.get('chain_id'),
                    'bez': copy_bezier_points(ch['bez']),
                    'tilt': list(ch['tilt']),
                    'radius': list(ch['radius']),
                    'modes': list(ch['modes']),
                    'origin_ids': list(ch.get('origin_ids') or list(range(len(ch['bez'])))),
                    'influence': float(ch.get('influence') or 0.0),
                    'point_influence': ensure_point_influence(
                        len(ch['bez']), ch.get('point_influence'),
                        default=float(ch.get('influence') or 0.1),
                    ),
                    'point_inf_falloff': ensure_point_inf_falloff(
                        len(ch['bez']), ch.get('point_inf_falloff'), default='CONSTANT',
                    ),
                    'bind_verts': [int(item[0]) for item in (ch.get('bind') or [])],
                    'vg_name': ch.get('vg_name') or '',
                }
                for ch in self.spine_chains
            ],
            'mesh_rest': mesh_rest,
        }
        _vdh_set_bind_rest(obj, entry_br)
        try:
            _vdh_persist_bind_rest_to_mesh(obj, entry_br)
        except Exception:
            pass
        self._spine_session_rest_bez = copy_bezier_points(self.bez)
        self._spine_session_rest_tilt = list(self.spine_tilt)
        self._spine_session_rest_radius = list(self.spine_radius)
        self._spine_session_rest_modes = list(self.point_modes)

        try:
            self._spine_save_recall(context)
        except Exception:
            pass
        # Bind/recall capture can flip dragging on. Enter Deform idle so the
        # first controller click-drag is a real grab, not "confirm leftover drag".
        self.dragging = False
        self._pending_click_drag = False
        self._pending_place_drag = False
        self._xform_mode = None
        self._xform_start = None
        self._xform_keys = None
        self.constraint_axis = None
        self._attr_mode = None
        self.report({'INFO'}, f"Spine bound: {len(self.spine_chains)} chain(s), {bound_total} verts")
        context.area.tag_redraw()
        return True

    def _spine_tangent(self, poly, t, eps=0.02):
        t0 = max(0.0, t - eps)
        t1 = min(1.0, t + eps)
        d = polyline_at(poly, t1) - polyline_at(poly, t0)
        if d.length < 1e-12:
            return Vector((0, 0, 1))
        return d.normalized()

    def _spine_rotation_between(self, a, b):
        """Return Matrix rotating unit vector a onto unit vector b."""
        a = a.normalized()
        b = b.normalized()
        dot = max(-1.0, min(1.0, a.dot(b)))
        if dot > 0.99999:
            return Matrix.Identity(3)
        if dot < -0.99999:
            # 180° — pick an orthogonal axis
            tmp = Vector((1, 0, 0)) if abs(a.x) < 0.9 else Vector((0, 1, 0))
            axis = a.cross(tmp).normalized()
            return Matrix.Rotation(math.pi, 3, axis)
        axis = a.cross(b).normalized()
        angle = math.acos(dot)
        return Matrix.Rotation(angle, 3, axis)

    def _spine_soften_handle_weights(self):
        """Lengthen Bezier handles so deformation blends more softly between controllers."""
        n = len(getattr(self, 'bez', []) or [])
        if n < 2:
            return
        for i, bp in enumerate(self.bez):
            co = bp['co']
            for part in ('hl', 'hr'):
                off = bp[part] - co
                L = off.length
                if L < 1e-8:
                    continue
                scale = 1.65 if 0 < i < n - 1 else 1.35
                bp[part] = co + off.normalized() * (L * scale)
        if getattr(self, 'rest_bez', None) and len(self.rest_bez) == n:
            for i, bp in enumerate(self.rest_bez):
                co = bp['co']
                for part in ('hl', 'hr'):
                    off = bp[part] - co
                    L = off.length
                    if L < 1e-8:
                        continue
                    scale = 1.65 if 0 < i < n - 1 else 1.35
                    bp[part] = co + off.normalized() * (L * scale)

    def _spine_is_deformed(self, eps=1e-7):
        """True if controllers/handles/tilt/radius differ from rest bind state."""
        if not getattr(self, 'bez', None) or not getattr(self, 'rest_bez', None):
            return False
        if len(self.bez) != len(self.rest_bez):
            return True
        for a, b in zip(self.bez, self.rest_bez):
            if (a['co'] - b['co']).length_squared > eps:
                return True
            if (a['hl'] - b['hl']).length_squared > eps:
                return True
            if (a['hr'] - b['hr']).length_squared > eps:
                return True
        tilts = getattr(self, 'spine_tilt', None) or []
        radii = getattr(self, 'spine_radius', None) or []
        if any(abs(t) > 1e-8 for t in tilts):
            return True
        if any(abs(r - 1.0) > 1e-8 for r in radii):
            return True
        return False

    def _spine_attr_blend(self, u):
        """Segment blend 0..1 for tilt/radius. CONSTANT holds the left key."""
        mode = str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper()
        u = max(0.0, min(1.0, float(u)))
        if mode == 'LINEAR':
            return u
        if mode == 'CONSTANT':
            return 0.0 if u < 1.0 - 1e-9 else 1.0
        if mode == 'SPHERE':
            # Symmetric spherical-style interpolation.  The previous
            # implementation used 1-sqrt(1-u^2), which is directional: on
            # the left side of a selected controller it behaved differently
            # from the right side.  Use the normalized integral of a sphere
            # profile so f(1-u) == 1-f(u).
            x = 2.0 * u - 1.0
            x = max(-1.0, min(1.0, x))
            area = x * math.sqrt(max(0.0, 1.0 - x * x)) + math.asin(x)
            return 0.5 + area / math.pi
        if mode == 'SHARP':
            # Symmetric sharp interpolation.  This concentrates the change
            # toward the controller while remaining mirrored on both sides.
            a = u * u * u
            b = (1.0 - u) * (1.0 - u) * (1.0 - u)
            return a / (a + b) if (a + b) > 1e-12 else u
        return u * u * (3.0 - 2.0 * u)

    def _spine_cycle_attr_interp(self, context, step=1):
        order = _ATTR_INTERP_ORDER
        cur = str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper()
        if cur not in order:
            cur = 'SMOOTH'
        idx = (order.index(cur) + int(step)) % len(order)
        self.spine_attr_interp = order[idx]
        # Store it on the active chain immediately; this is the falloff that
        # must survive Enter -> exit -> re-enter.
        chains = getattr(self, 'spine_chains', None) or []
        ac = int(getattr(self, 'active_chain', 0) or 0)
        if 0 <= ac < len(chains):
            chains[ac]['attr_interp'] = self.spine_attr_interp
        try:
            self._spine_apply(context, auto_soft=False)
        except Exception:
            pass
        if context and getattr(context, 'area', None):
            context.area.tag_redraw()
        self.report({'INFO'}, f"Tilt/Radius falloff: {self.spine_attr_interp.title()}")
        return True

    def _spine_attr_at(self, t):
        """Interpolate tilt/radius only between neighboring controllers."""
        n = len(self.bez) if self.bez else 0
        if n == 0:
            return 0.0, 1.0
        tilts = getattr(self, 'spine_tilt', None) or [0.0] * n
        radii = getattr(self, 'spine_radius', None) or [1.0] * n
        if len(tilts) != n:
            tilts = [0.0] * n
            self.spine_tilt = tilts
        if len(radii) != n:
            radii = [1.0] * n
            self.spine_radius = radii
        if n == 1:
            return float(tilts[0]), max(float(radii[0]), 1e-4)
        hparams = list(getattr(self, 'handle_params', []) or [])
        if len(hparams) != n:
            hparams = [i / max(1, n - 1) for i in range(n)]
            if hparams:
                hparams[0], hparams[-1] = 0.0, 1.0
        keys = sorted(
            ((float(hparams[i]), float(tilts[i]), float(radii[i])) for i in range(n)),
            key=lambda x: x[0],
        )
        t = float(t)
        mode = str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper()
        if mode == 'CONSTANT':
            # Constant: each controller owns the nearest half of the
            # neighboring intervals, so the step is centered on each key.
            nearest = min(keys, key=lambda item: abs(float(item[0]) - t))
            return nearest[1], max(nearest[2], 1e-4)
        if t <= keys[0][0]:
            return keys[0][1], max(keys[0][2], 1e-4)
        if t >= keys[-1][0]:
            return keys[-1][1], max(keys[-1][2], 1e-4)
        for i in range(len(keys) - 1):
            t0, a0, r0 = keys[i]
            t1, a1, r1 = keys[i + 1]
            if t <= t1 or i == len(keys) - 2:
                span = t1 - t0
                u = 0.0 if abs(span) < 1e-12 else (t - t0) / span
                b = self._spine_attr_blend(u)
                tilt = a0 * (1.0 - b) + a1 * b
                rad = r0 * (1.0 - b) + r1 * b
                return tilt, max(rad, 1e-4)
        return keys[-1][1], max(keys[-1][2], 1e-4)

    def _spine_frame(self, pts, t, prev_x=None):
        """Orthonormal frame (tan, x, y) along curve — parallel-transport style."""
        tan = bezier_chain_tangent(pts, t)
        if prev_x is not None and prev_x.length > 1e-8:
            # keep x as parallel as possible to previous
            x = prev_x - tan * prev_x.dot(tan)
            if x.length < 1e-8:
                tmp = Vector((0, 0, 1)) if abs(tan.z) < 0.9 else Vector((1, 0, 0))
                x = tan.cross(tmp)
            x.normalize()
        else:
            tmp = Vector((0, 0, 1)) if abs(tan.z) < 0.9 else Vector((1, 0, 0))
            x = tan.cross(tmp)
            if x.length < 1e-8:
                x = Vector((1, 0, 0))
            x.normalize()
        y = tan.cross(x).normalized()
        return tan, x, y

    def _spine_build_frame_table(self, pts, samples=None, seed_x=None):
        """Dense parallel-transport frames along a Bezier chain.
        Returns list of (t, tan, x, y). Keeps cross-sections stable under bend.
        seed_x: optional initial side vector (e.g. rest frame X) so live curve
        starts with the same roll as rest — reduces end twist when editing tips.
        """
        if not pts or len(pts) < 2:
            return []
        if samples is None:
            samples = max(48, len(pts) * 20)
        table = []
        prev_x = seed_x.copy() if seed_x is not None and getattr(seed_x, 'length', 0) > 1e-12 else None
        for s in range(samples + 1):
            t = s / float(samples)
            tan, x, y = self._spine_frame(pts, t, prev_x)
            table.append((t, tan.copy(), x.copy(), y.copy()))
            prev_x = x
        return table

    def _spine_lookup_frame(self, table, t):
        """Interpolate frame at parameter t from a frame table."""
        if not table:
            return Vector((0, 0, 1)), Vector((1, 0, 0)), Vector((0, 1, 0))
        if t <= table[0][0]:
            return table[0][1], table[0][2], table[0][3]
        if t >= table[-1][0]:
            return table[-1][1], table[-1][2], table[-1][3]
        # Binary search segment
        lo, hi = 0, len(table) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if table[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        t0, T0, X0, Y0 = table[lo]
        t1, T1, X1, Y1 = table[hi]
        u = 0.0 if abs(t1 - t0) < 1e-12 else (t - t0) / (t1 - t0)
        # Nlerp-ish: normalize blended axes
        def _blend(a, b, u):
            v = a * (1.0 - u) + b * u
            if v.length < 1e-12:
                return a.copy()
            return v.normalized()
        T = _blend(T0, T1, u)
        X = _blend(X0, X1, u)
        # Re-orthonormalize: project X off T, rebuild Y
        X = X - T * X.dot(T)
        if X.length < 1e-12:
            X = X0.copy() if X0.length > 1e-12 else Vector((1, 0, 0))
            X = X - T * X.dot(T)
            if X.length < 1e-12:
                tmp = Vector((0, 0, 1)) if abs(T.z) < 0.9 else Vector((1, 0, 0))
                X = T.cross(tmp)
        X.normalize()
        Y = T.cross(X).normalized()
        return T, X, Y

    def _spine_apply(self, context, auto_soft=True):
        """Apply Spine with intuitive Weight Paint × Radius × Falloff blending."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        # Keep active chain edits in the list
        live_drag = bool(
            getattr(self, 'dragging', False)
            or getattr(self, '_xform_mode', None)
            or getattr(self, '_attr_mode', None)
        )
        if getattr(self, 'spine_chains', None):
            if not live_drag:
                self._spine_store_active_chain()
            chains = self.spine_chains
        else:
            if not self.spine_bind or not getattr(self, 'bez', None) or len(self.bez) < 2:
                return
            chains = [{
                'bez': self.bez,
                'rest_bez': self.rest_bez,
                'bind': self.spine_bind,
                'tilt': getattr(self, 'spine_tilt', []),
                'radius': getattr(self, 'spine_radius', []),
                'handle_params': getattr(self, 'handle_params', []),
                'modes': getattr(self, 'point_modes', []),
            }]
        bm.verts.ensure_lookup_table()

        # A Weight Paint stroke can add/migrate vertices after the previous bind
        # was built.  Before any live controller transform, reconcile bind
        # ownership from the current Blender weights.  This is intentionally
        # gated by the dirty flag so normal interactive transforms remain fast.
        weightpaint_dirty_global = False
        try:
            weightpaint_dirty_global = bool(obj.data.get('_bh_spine_weightpaint_dirty', False))
        except Exception:
            weightpaint_dirty_global = False
        if weightpaint_dirty_global and getattr(self, 'spine_chains', None):
            for _ci, _ch in enumerate(chains):
                try:
                    self._spine_absorb_vg_into_bind(obj, bm, _ch, _ci)
                except Exception:
                    pass

        # backup active attrs
        bak = (getattr(self, 'spine_tilt', None), getattr(self, 'spine_radius', None),
               getattr(self, 'handle_params', None), getattr(self, 'bez', None))

        # All chains participate in a live drag because Weight Paint may
        # intentionally overlap chains. Contributions are accumulated and
        # committed once from the shared rest pose.
        dirty = None
        accum_target = {}
        accum_weight = {}

        for ci, ch in enumerate(chains):
            if dirty is not None and ci not in dirty:
                continue
            if not getattr(self, 'dragging', False):
                try:
                    self._spine_absorb_vg_into_bind(obj, bm, ch, ci)
                except Exception:
                    pass
            bez = ch.get('bez')
            rest_bez = ch.get('rest_bez')
            bind = ch.get('bind') or []
            if not bez or len(bez) < 2 or not bind:
                continue
            self.spine_tilt = ch.get('tilt') or [0.0] * len(bez)
            self.spine_radius = ch.get('radius') or [1.0] * len(bez)
            self.handle_params = ch.get('handle_params') or []
            self.bez = bez
            # Bishop frames on rest AND live, seeded from the same side vector.
            # rotation_difference(T0, T1) is free to roll around the tangent
            # (especially on a U-bend / axis change) and twists the tube.
            rest_src = rest_bez if rest_bez and len(rest_bez) >= 2 else bez
            rf_key = id(rest_src)
            seed_x = ch.get('_seed_x')
            if seed_x is None or getattr(seed_x, 'length', 0) < 1e-12:
                t0 = bezier_chain_tangent(rest_src, 0.0)
                if t0.length < 1e-12:
                    t0 = Vector((0.0, 0.0, 1.0))
                else:
                    t0 = t0.normalized()
                axis = Vector((0.0, 0.0, 1.0))
                if abs(t0.z) > 0.9:
                    axis = Vector((1.0, 0.0, 0.0)) if abs(t0.x) < abs(t0.y) else Vector((0.0, 1.0, 0.0))
                seed_x = t0.cross(axis)
                if seed_x.length < 1e-12:
                    seed_x = Vector((1.0, 0.0, 0.0))
                else:
                    seed_x.normalize()
                ch['_seed_x'] = seed_x.copy()
            if ch.get('_rf_key') != rf_key or not ch.get('_rf_table'):
                ch['_rf_table'] = self._spine_build_frame_table(rest_src, seed_x=seed_x)
                ch['_rf_key'] = rf_key
            rest_frames = ch['_rf_table']
            live_frames = self._spine_build_frame_table(bez, seed_x=seed_x)

            n_ctrl = len(bez)
            legacy_inf = float(ch.get('influence') or getattr(self, 'spine_influence', 0.1) or 0.1)
            pinf = ensure_point_influence(
                n_ctrl, ch.get('point_influence'), default=legacy_inf,
            )
            ch['point_influence'] = pinf
            pfo = ensure_point_inf_falloff(
                n_ctrl,
                ch.get('point_inf_falloff'),
                default=ch.get('inf_falloff') or getattr(self, 'influence_falloff', 'CONSTANT') or 'CONSTANT',
            )
            ch['point_inf_falloff'] = pfo
            hparams = list(ch.get('handle_params') or [])
            if len(hparams) != n_ctrl:
                hparams = [i / max(1, n_ctrl - 1) for i in range(n_ctrl)]
                if hparams:
                    hparams[0], hparams[-1] = 0.0, 1.0

            # Weight Paint is the source of truth for Spine deformation. A
            # Weight Transfer stroke marks the mesh dirty; never reuse the
            # previous cached _fw after such a stroke.
            weightpaint_dirty = False
            try:
                weightpaint_dirty = bool(obj.data.get('_bh_spine_weightpaint_dirty', False))
            except Exception:
                weightpaint_dirty = False
            if weightpaint_dirty:
                ch['_fw'] = None

            if live_drag and ch.get('_fw') and len(ch['_fw']) == len(bind):
                soft_w = ch['_fw']
            else:
                soft_w = self._spine_final_weight_list(obj, bm, ch)
                ch['_fw'] = soft_w

            # Precompute curve samples for unique quantized t (big win on dense rings)
            t_cache = {}  # q -> (on, T1, T0, X0, Y0, tilt_a, rad_s)

            def _sample_at(t):
                q = int(round(t * 256.0))
                hit = t_cache.get(q)
                if hit is not None:
                    return hit
                on = eval_bezier_points(bez, t)
                T0, X0, Y0 = self._spine_lookup_frame(rest_frames, t)
                T1, X1, Y1 = self._spine_lookup_frame(live_frames, t)
                tilt_a, rad_s = self._spine_attr_at(t)
                hit = (on, T1, T0, X0, Y0, X1, Y1, tilt_a, rad_s)
                t_cache[q] = hit
                return hit

            for _bi, item in enumerate(bind):
                vidx = item[0]
                t = float(item[1])
                offset = item[2]
                if vidx >= len(bm.verts):
                    continue
                rest_pos = self.all_rest.get(vidx)
                if rest_pos is None:
                    continue
                on, T1, T0, X0, Y0, X1, Y1, tilt_a, rad_s = _sample_at(t)
                w = soft_w[_bi] if _bi < len(soft_w) else 1.0
                if w <= 1e-6:
                    continue
                if offset is None:
                    try:
                        target = rest_pos + (on - eval_bezier_points(rest_src, t))
                    except Exception:
                        target = rest_pos.copy()
                else:
                    lx = offset.dot(X0); ly = offset.dot(Y0); lz = offset.dot(T0)
                    lx_s = lx * rad_s; ly_s = ly * rad_s
                    if abs(tilt_a) > 1e-8:
                        ca = math.cos(tilt_a); sa = math.sin(tilt_a)
                        lx_s, ly_s = lx_s * ca - ly_s * sa, lx_s * sa + ly_s * ca
                    target = on + X1 * lx_s + Y1 * ly_s + T1 * lz
                accum_target[vidx] = accum_target.get(vidx, Vector((0.0, 0.0, 0.0))) + target * w
                accum_weight[vidx] = accum_weight.get(vidx, 0.0) + w

        # Blend every influenced vertex from the shared rest pose. With normal
        # normalized Spine weights (sum <= 1), the remaining fraction stays at
        # rest. Legacy sums > 1 are normalized to avoid overshoot.
        for vidx, total_w in accum_weight.items():
            if vidx < 0 or vidx >= len(bm.verts):
                continue
            rest_pos = self.all_rest.get(vidx)
            if rest_pos is None or total_w <= 1e-8:
                continue
            if total_w <= 1.0:
                bm.verts[vidx].co = rest_pos * (1.0 - total_w) + accum_target[vidx]
            else:
                bm.verts[vidx].co = accum_target[vidx] / total_w


        # restore active
        if bak[0] is not None:
            self.spine_tilt = bak[0]
        if bak[1] is not None:
            self.spine_radius = bak[1]
        if bak[2] is not None:
            self.handle_params = bak[2]
        if bak[3] is not None:
            self.bez = bak[3]
        if getattr(self, 'spine_chains', None) and not live_drag:
            self._spine_load_active_chain()
            self.spine_points = [bp['co'].copy() for bp in self.bez]

        try:
            bm.normal_update()
        except Exception:
            try:
                for f in bm.faces:
                    f.normal_update()
                for v in bm.verts:
                    v.normal_update()
            except Exception:
                pass
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        try:
            obj.data.update()
        except Exception:
            pass
        try:
            obj.data['_bh_spine_weightpaint_dirty'] = False
        except Exception:
            pass

    def _spine_auto_soft(self, bm, iterations=4, factor=0.4):
        """Light tangential relax that keeps each vert's radius to the live curve.
        Gives the 'everything moves smoothly' Curve feel without collapsing volume.
        """
        if not self.spine_bind:
            return
        # Precompute live curve attachment
        data = []
        for item in self.spine_bind:
            vidx = item[0]
            t = float(item[1])
            if vidx >= len(bm.verts):
                continue
            center = eval_bezier_points(self.bez, t)
            tan = bezier_chain_tangent(self.bez, t)
            rel = bm.verts[vidx].co - center
            rel = rel - tan * rel.dot(tan)
            radius = max(rel.length, 1e-8)
            data.append((vidx, t, center, tan, radius))

        bound_set = {d[0] for d in data}
        for _ in range(iterations):
            new_co = {}
            for vidx, t, center, tan, radius in data:
                v = bm.verts[vidx]
                linked = [e.other_vert(v) for e in v.link_edges if e.other_vert(v).index in bound_set]
                if not linked:
                    continue
                avg = Vector((0, 0, 0))
                for ov in linked:
                    avg += ov.co
                avg /= float(len(linked))
                delta = avg - v.co
                # tangential only
                delta = delta - tan * delta.dot(tan)
                # also remove normal-ish by staying in plane of ring: already removed tan
                trial = v.co + delta * factor
                # restore radius to curve
                rel = trial - center
                rel = rel - tan * rel.dot(tan)
                if rel.length > 1e-10:
                    rel = rel.normalized() * radius
                new_co[vidx] = center + rel
            for vidx, co in new_co.items():
                bm.verts[vidx].co = co

    def _spine_snapshot(self, context=None):
        """Full spine state: controllers, bind, and bound mesh positions."""
        if getattr(self, 'spine_chains', None):
            try:
                self._spine_store_active_chain()
            except Exception:
                pass
        mesh = {}
        # Collect verts from ALL chains (not only active).  For Spine history we
        # must also capture every vertex that currently has a BH_Spine* weight.
        # Straighten/Rebind/Weight-Paint reconciliation can change bind ownership
        # during an operation, so a bind-only snapshot is not sufficient: a vertex
        # that becomes bound/affected during the operation could otherwise be
        # missing from Undo and remain distorted after the curve itself is restored.
        bind_items = []
        for ch in (getattr(self, 'spine_chains', None) or []):
            bind_items.extend(ch.get('bind') or [])
        if not bind_items:
            bind_items = list(getattr(self, 'spine_bind', []) or [])
        if context is not None:
            obj, bm = self.get_obj_bm(context)
            if bm is not None:
                bm.verts.ensure_lookup_table()
                affected_ids = {int(item[0]) for item in bind_items if item}
                try:
                    dl = bm.verts.layers.deform.verify()
                    spine_gis = set()
                    for vg in obj.vertex_groups:
                        if str(vg.name).startswith('BH_Spine'):
                            spine_gis.add(int(vg.index))
                    if spine_gis:
                        for v in bm.verts:
                            d = v[dl]
                            if any(gi in d and float(d[gi]) > 1e-8 for gi in spine_gis):
                                affected_ids.add(int(v.index))
                except Exception:
                    pass
                for vidx in affected_ids:
                    if 0 <= vidx < len(bm.verts):
                        mesh[vidx] = bm.verts[vidx].co.copy()
        if not mesh:
            for item in bind_items:
                vidx = item[0]
                if vidx in (getattr(self, 'all_rest', {}) or {}):
                    mesh[vidx] = self.all_rest[vidx].copy()
        chains_snap = []
        for ch in (getattr(self, 'spine_chains', None) or []):
            chains_snap.append({
                'bez': copy_bezier_points(ch['bez']) if ch.get('bez') else None,
                'rest_bez': copy_bezier_points(ch['rest_bez']) if ch.get('rest_bez') else None,
                'modes': list(ch.get('modes') or []),
                'tilt': list(ch.get('tilt') or []),
                'radius': list(ch.get('radius') or []),
                'handle_params': list(ch.get('handle_params') or []),
                'bind': [
                    (
                        item[0], float(item[1]),
                        item[2].copy() if hasattr(item[2], 'copy') else item[2],
                        item[3].copy() if len(item) > 3 and hasattr(item[3], 'copy') else (item[3] if len(item) > 3 else None),
                        float(item[4]) if len(item) > 4 else 0.0,
                    )
                    for item in (ch.get('bind') or [])
                ],
                'influence': float(ch.get('influence', 0.1) or 0.1),
                'point_influence': ensure_point_influence(
                    len(ch.get('bez') or []), ch.get('point_influence'),
                    default=float(ch.get('influence', 0.1) or 0.1),
                ),
                # Keep the persistent Alt+R reset baseline in Undo/Redo.
                # Without this, restoring a snapshot after Radius Reset could
                # lose the original bind-time defaults and make Ctrl+Z appear
                # to work for the curve while Radius stayed reset.
                'point_influence_default': ensure_point_influence(
                    len(ch.get('bez') or []), ch.get('point_influence_default'),
                    default=(float(ch.get('influence', 0.1) or 0.1)),
                ),
                'point_inf_falloff': ensure_point_inf_falloff(
                    len(ch.get('bez') or []), ch.get('point_inf_falloff'), default='CONSTANT'
                ),
                'attr_interp': (str(ch.get('attr_interp') or getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper()
                                if str(ch.get('attr_interp') or getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper() in _ATTR_INTERP_ORDER
                                else 'SMOOTH'),
                'origin_ids': list(ch.get('origin_ids') or []),
                'chain_id': ch.get('chain_id'),
                'in_front': bool(ch.get('in_front', True)),
                'vg_name': ch.get('vg_name') or '',
            })
        return {
            'points': [p.copy() for p in self.spine_points],
            'chains_pts': [[p.copy() for p in c] for c in (getattr(self, 'spine_chains_pts', None) or [])],
            'chains': chains_snap,
            'face_flip_parity': int(getattr(self, '_face_flip_parity', 0) or 0),
            'active_chain': int(getattr(self, 'active_chain', 0) or 0),
            'bez': copy_bezier_points(self.bez) if getattr(self, 'bez', None) else None,
            'rest_bez': copy_bezier_points(self.rest_bez) if getattr(self, 'rest_bez', None) else None,
            'modes': list(getattr(self, 'point_modes', [])),
            'tilt': list(getattr(self, 'spine_tilt', []) or []),
            'radius': list(getattr(self, 'spine_radius', []) or []),
            'influence': float(getattr(self, 'spine_influence', 0.1) or 0.1),
            'point_influence': ensure_point_influence(
                len(getattr(self, 'bez', None) or []),
                getattr(self, 'point_influence', None),
                default=float(getattr(self, 'spine_influence', 0.1) or 0.1),
            ),
            'point_inf_falloff': ensure_point_inf_falloff(
                len(getattr(self, 'bez', None) or []),
                getattr(self, 'point_inf_falloff', None),
                default='CONSTANT',
            ),
            'attr_interp': (str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper()
                            if str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper() in _ATTR_INTERP_ORDER
                            else 'SMOOTH'),
            'bind': [
                (
                    item[0],
                    float(item[1]),
                    item[2].copy() if hasattr(item[2], 'copy') else item[2],
                    item[3].copy() if len(item) > 3 and hasattr(item[3], 'copy') else (item[3] if len(item) > 3 else None),
                    float(item[4]) if len(item) > 4 else 0.0,
                )
                for item in (getattr(self, 'spine_bind', []) or [])
            ],
            'mesh': mesh,
            'vg_weights': self._spine_capture_vg_state(context),
        }

    def _spine_capture_vg_state(self, context=None):
        """Sparse snapshot of BH_Spine* group assignments (missing vert = not in group)."""
        out = {}
        if context is None:
            return out
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return out
        try:
            bm.verts.ensure_lookup_table()
            dl = bm.verts.layers.deform.verify()
        except Exception:
            return out
        names = []
        for i, ch in enumerate(getattr(self, 'spine_chains', None) or []):
            name = ch.get('vg_name') or _vdh_spine_vg_name(i)
            if name and name not in names:
                names.append(name)
        for name in names:
            vg = obj.vertex_groups.get(name)
            if vg is None:
                out[name] = None
                continue
            gi = vg.index
            wmap = {}
            for v in bm.verts:
                d = v[dl]
                if gi in d:
                    try:
                        wmap[int(v.index)] = float(d[gi])
                    except Exception:
                        continue
            out[name] = wmap
        return out

    def _spine_restore_vg_state(self, context, vg_data):
        """Replace BH group assignments with a snapshot (exact membership)."""
        if not vg_data:
            return
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        bm.verts.ensure_lookup_table()
        try:
            dl = bm.verts.layers.deform.verify()
        except Exception:
            return
        for name, wmap in vg_data.items():
            if not name:
                continue
            if wmap is None:
                vg = obj.vertex_groups.get(name)
                if vg is not None:
                    try:
                        obj.vertex_groups.remove(vg)
                    except Exception:
                        pass
                continue
            vg = _vdh_ensure_vertex_group(obj, name)
            if vg is None:
                continue
            gi = vg.index
            assigned = set(int(k) for k in wmap.keys())
            for v in bm.verts:
                d = v[dl]
                if v.index in assigned:
                    d[gi] = max(0.0, min(1.0, float(wmap[v.index])))
                elif gi in d:
                    try:
                        del d[gi]
                    except Exception:
                        try:
                            d[gi] = 0.0
                            del d[gi]
                        except Exception:
                            pass
        try:
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        except Exception:
            pass

    def _spine_push_undo(self, context=None):
        self.undo_stack.append(self._spine_snapshot(context))
        # Cap stack so memory stays bounded (does not change undo behavior)
        if len(self.undo_stack) > 64:
            self.undo_stack = self.undo_stack[-64:]
        self.redo_stack.clear()

    def _spine_restore_entry(self, entry, context=None):
        if isinstance(entry, list):
            self.spine_points = [p.copy() for p in entry]
            return
        self.spine_points = [p.copy() for p in entry.get('points', [])]
        if 'chains_pts' in entry:
            self.spine_chains_pts = [[p.copy() for p in c] for c in (entry.get('chains_pts') or [])]
        if entry.get('chains') is not None:
            restored = []
            for ch in entry['chains']:
                nbez = len(ch.get('bez') or [])
                restored.append({
                    'bez': copy_bezier_points(ch['bez']) if ch.get('bez') else None,
                    'rest_bez': copy_bezier_points(ch['rest_bez']) if ch.get('rest_bez') else None,
                    'modes': list(ch.get('modes') or []),
                    'tilt': list(ch.get('tilt') or []),
                    'radius': list(ch.get('radius') or []),
                    'handle_params': list(ch.get('handle_params') or []),
                    'bind': [tuple(item) for item in (ch.get('bind') or [])],
                    'influence': float(ch.get('influence', 0.1) or 0.1),
                    'point_influence': ensure_point_influence(
                        nbez, ch.get('point_influence'),
                        default=float(ch.get('influence', 0.1) or 0.1),
                    ),
                    'point_influence_default': ensure_point_influence(
                        nbez, ch.get('point_influence_default'),
                        default=(float(ch.get('influence', 0.1) or 0.1)),
                    ),
                    'point_inf_falloff': ensure_point_inf_falloff(
                        nbez, ch.get('point_inf_falloff'), default='CONSTANT'
                    ),
                    'attr_interp': (str(ch.get('attr_interp') or 'SMOOTH').upper()
                                    if str(ch.get('attr_interp') or 'SMOOTH').upper() in _ATTR_INTERP_ORDER
                                    else 'SMOOTH'),
                    'origin_ids': list(ch.get('origin_ids') or []),
                    'chain_id': ch.get('chain_id'),
                    'in_front': bool(ch.get('in_front', True)),
                    'vg_name': ch.get('vg_name') or '',
                })
            self.spine_chains = restored
            # A restored history state must not retain solver caches from the
            # state that was just undone.
            for _ch in self.spine_chains:
                _ch['_fw'] = None
                _ch['_sw_key'] = None
                _ch['_soft_w'] = None
            self._influence_overlay_cache = None
            self.active_chain = int(entry.get('active_chain', 0) or 0)
            if self.spine_chains:
                self._spine_load_active_chain()
        bez = entry.get('bez')
        if bez is not None and not entry.get('chains'):
            self.bez = copy_bezier_points(bez)
            self.spine_points = [bp['co'].copy() for bp in self.bez]
        rest = entry.get('rest_bez')
        if rest is not None and not entry.get('chains'):
            self.rest_bez = copy_bezier_points(rest)
        modes = entry.get('modes')
        if modes is not None and not entry.get('chains'):
            self.point_modes = list(modes)
        if entry.get('tilt') is not None and not entry.get('chains'):
            self.spine_tilt = list(entry['tilt'])
        if entry.get('radius') is not None and not entry.get('chains'):
            self.spine_radius = list(entry['radius'])
        if entry.get('influence') is not None and not entry.get('chains'):
            self.spine_influence = float(entry['influence'])
        if entry.get('point_inf_falloff') is not None and not entry.get('chains'):
            self.point_inf_falloff = ensure_point_inf_falloff(
                len(getattr(self, 'bez', None) or []),
                entry.get('point_inf_falloff'),
                default='CONSTANT',
            )
        if entry.get('attr_interp') is not None and not entry.get('chains'):
            _ai = str(entry.get('attr_interp') or 'SMOOTH').upper()
            self.spine_attr_interp = _ai if _ai in _ATTR_INTERP_ORDER else 'SMOOTH'
        bind = entry.get('bind')
        if bind is not None and not entry.get('chains'):
            self.spine_bind = [tuple(item) for item in bind]
        mesh = entry.get('mesh') or {}
        if context is not None and 'vg_weights' in entry:
            try:
                self._spine_restore_vg_state(context, entry.get('vg_weights') or {})
            except Exception:
                pass
        # Restore the exact mesh coordinates captured by the history entry.
        # Do NOT re-run the live Spine solver here: after operations such as
        # Shift+L, the restored curve/bind can be valid while a fresh _spine_apply
        # still recomputes the mesh from caches/rest data and leaves a subtle
        # distortion.  Undo must be an exact state restoration, not a re-deform.
        if context is not None and mesh:
            obj, bm = self.get_obj_bm(context)
            if bm is not None:
                bm.verts.ensure_lookup_table()
                for vidx, co in mesh.items():
                    if 0 <= vidx < len(bm.verts):
                        bm.verts[vidx].co = co.copy()
                try:
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                except Exception:
                    pass
        # Sync face winding with snapshot parity (undo/redo after mirror)
        if context is not None and isinstance(entry, dict) and 'face_flip_parity' in entry:
            target = int(entry.get('face_flip_parity', 0) or 0)
            cur = int(getattr(self, '_face_flip_parity', 0) or 0)
            need_flip = (target % 2) != (cur % 2)
            self._face_flip_parity = target
            try:
                self._spine_recalc_normals(context, flip=need_flip)
            except Exception:
                pass

    def _spine_undo(self, context):
        if not self.undo_stack:
            return
        self.redo_stack.append(self._spine_snapshot(context))
        self._spine_restore_entry(self.undo_stack.pop(), context)
        # Face winding may still be flipped from a prior mirror — fix outward normals
        try:
            self._spine_recalc_normals(context, flip=False)
        except Exception:
            pass
        context.area.tag_redraw()

    def _spine_redo(self, context):
        if not self.redo_stack:
            return
        self.undo_stack.append(self._spine_snapshot(context))
        self._spine_restore_entry(self.redo_stack.pop(), context)
        try:
            self._spine_recalc_normals(context, flip=False)
        except Exception:
            pass
        context.area.tag_redraw()

    def _spine_toggle_place_in_volume(self, context):
        """P: Place in Volume — enable snap+VOLUME, or restore / turn off."""
        ts = context.tool_settings
        on = bool(getattr(self, '_place_in_volume', False))
        if not on:
            # Save current and switch to VOLUME
            elems = set(getattr(ts, 'snap_elements', set()) or set())
            self._snap_backup_use = bool(ts.use_snap)
            self._snap_backup_elements = elems.copy() if elems else set()
            try:
                ts.use_snap = True
                # Blender 4.x: snap_elements is a set enum
                ts.snap_elements = {'VOLUME'}
            except Exception:
                try:
                    ts.snap_elements_base = {'VOLUME'}
                except Exception:
                    pass
            self._place_in_volume = True
            self.report({'INFO'}, "Place in Volume: On")
        else:
            # Turn off volume mode
            backup_elems = set(getattr(self, '_snap_backup_elements', set()) or set())
            backup_use = bool(getattr(self, '_snap_backup_use', False))
            was_already_volume = backup_use and ('VOLUME' in backup_elems)
            try:
                if was_already_volume:
                    # Started as volume → second P turns snap completely off
                    ts.use_snap = False
                else:
                    ts.use_snap = backup_use
                    if backup_elems:
                        ts.snap_elements = backup_elems
                    else:
                        # fallback if empty backup
                        ts.use_snap = False
            except Exception:
                ts.use_snap = False
            self._place_in_volume = False
            self.report({'INFO'}, "Place in Volume: Off")
        context.area.tag_redraw()

    def _spine_restore_snap(self, context):
        """Always restore snap to the state before Place in Volume (P)."""
        if not getattr(self, '_place_in_volume', False):
            # Still restore if we have a backup and snap is stuck on VOLUME from us
            if not hasattr(self, '_snap_backup_use'):
                return
        try:
            ts = context.tool_settings
            bak_use = getattr(self, '_snap_backup_use', None)
            bak_elems = getattr(self, '_snap_backup_elements', None)
            if bak_use is None and bak_elems is None:
                return
            if bak_use is not None:
                ts.use_snap = bool(bak_use)
            if bak_elems is not None:
                try:
                    ts.snap_elements = set(bak_elems) if bak_elems else set()
                except Exception:
                    try:
                        if bak_elems:
                            ts.snap_elements_base = set(bak_elems)
                    except Exception:
                        pass
        except Exception:
            pass
        self._place_in_volume = False

    def _spine_capture_attr_layers(self, context, original_coords=None):
        """Capture Tilt and Shrink/Inflate as explicit mesh layers.

        The solver remains rest-based.  At Confirm time we evaluate the same
        controller pose three times: base (no attributes), Tilt-only, and full
        attributes.  The two deltas are stored separately so an external mesh
        edit can later be absorbed into the base without baking Inflate/Shrink
        into the true Spine rest.
        """
        obj, bm = self.get_obj_bm(context)
        chains = getattr(self, 'spine_chains', None) or []
        if obj is None or bm is None or not chains:
            return {'tilt': {}, 'inflate': {}}
        bm.verts.ensure_lookup_table()
        if original_coords is None:
            original_coords = {v.index: v.co.copy() for v in bm.verts}

        saved_attrs = []
        for ch in chains:
            saved_attrs.append((ch, list(ch.get('tilt') or []), list(ch.get('radius') or [])))

        # Capture must never leak drag/xform flags into the live modal.
        # Leaving dragging=True made the first Deform click-drag act as
        # "confirm previous grab" and do nothing.
        _prev_dragging = bool(getattr(self, 'dragging', False))
        _prev_xform = getattr(self, '_xform_mode', None)

        def _eval():
            self.dragging = True
            self._xform_mode = 'ATTR_LAYER_CAPTURE'
            self._influence_overlay_cache = None
            for ch in chains:
                ch['_fw'] = None
                ch['_sw_key'] = None
                ch['_soft_w'] = None
            self._spine_apply(context, auto_soft=False)
            return {v.index: v.co.copy() for v in bm.verts}

        try:
            # Base/controller layer.
            for ch, tilt, radius in saved_attrs:
                n = len(ch.get('bez') or [])
                ch['tilt'] = [0.0] * n
                ch['radius'] = [1.0] * n
            base = _eval()

            # Tilt layer only.
            for ch, tilt, radius in saved_attrs:
                ch['tilt'] = list(tilt)
                ch['radius'] = [1.0] * len(ch.get('bez') or [])
            tilt_mesh = _eval()

            # Full Tilt + Shrink/Inflate layer.
            for ch, tilt, radius in saved_attrs:
                ch['tilt'] = list(tilt)
                ch['radius'] = list(radius)
            full = _eval()

            tilt_delta = {}
            inflate_delta = {}
            for vidx in base.keys():
                b = base.get(vidx)
                t = tilt_mesh.get(vidx)
                f = full.get(vidx)
                if b is None or t is None or f is None:
                    continue
                dt = t - b
                di = f - t
                if dt.length > 1e-8:
                    tilt_delta[int(vidx)] = dt.copy()
                if di.length > 1e-8:
                    inflate_delta[int(vidx)] = di.copy()
            return {'tilt': tilt_delta, 'inflate': inflate_delta}
        finally:
            for ch, tilt, radius in saved_attrs:
                ch['tilt'] = list(tilt)
                ch['radius'] = list(radius)
                ch['_fw'] = None
                ch['_sw_key'] = None
                ch['_soft_w'] = None
            self._influence_overlay_cache = None
            try:
                self.dragging = True
                self._xform_mode = 'ATTR_LAYER_CAPTURE'
                for v in bm.verts:
                    co = original_coords.get(v.index)
                    if co is not None:
                        v.co = co.copy()
                bm.normal_update()
            except Exception:
                pass
            self.dragging = _prev_dragging
            self._xform_mode = _prev_xform

    def _spine_apply_external_attr_layers(self, obj, bm, attr_layers, mesh_snap):
        """Absorb an external mesh edit into the base/rest.

        attr_layers is optional metadata; the external displacement itself is
        always computed from the last confirmed mesh snapshot.
        """
        if not mesh_snap:
            return False
        try:
            bm.verts.ensure_lookup_table()
            ext = {}
            for vidx, old in mesh_snap.items():
                i = int(vidx)
                if i < 0 or i >= len(bm.verts):
                    continue
                ext[i] = bm.verts[i].co.copy() - old
            if not ext:
                return False
            changed = False
            for i, d in ext.items():
                if d.length <= 1e-8:
                    continue
                old_rest = self.all_rest.get(i)
                if old_rest is None:
                    continue
                self.all_rest[i] = old_rest + d
                changed = True
            return changed
        except Exception:
            return False

    def _spine_update_attr_layer_groups(self, obj, attr_layers):
        """Create non-deformation marker groups for captured Tilt/Inflate vertices."""
        if obj is None or obj.type != 'MESH':
            return
        specs = (
            ('BH_ATTR_Tilt', attr_layers.get('tilt') or {}),
            ('BH_ATTR_Inflate', attr_layers.get('inflate') or {}),
        )
        for name, layer in specs:
            try:
                vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
                ids = list(layer.keys())
                all_ids = list(range(len(obj.data.vertices)))
                if all_ids:
                    vg.remove(all_ids)
                if ids:
                    mags = [layer[i].length for i in ids]
                    mx = max(mags) if mags else 0.0
                    if mx > 1e-8:
                        for i in ids:
                            vg.add([int(i)], max(0.0, min(1.0, layer[i].length / mx)), 'REPLACE')
            except Exception:
                pass

    def _spine_clear_attr_layer_group(self, obj, name):
        if obj is None or obj.type != 'MESH':
            return
        try:
            vg = obj.vertex_groups.get(name)
            if vg is not None:
                obj.vertex_groups.remove(vg)
        except Exception:
            pass

    def _spine_save_recall(self, context):
        """Store full multi-chain spine session for this mesh.

        Must work outside Edit Mode too (Weight Paint / Object Mode). The modal
        auto-confirms on mode switch; if we skip save then, the next BH open
        reloads a stale curve and the blue line jumps.
        """
        global _vdh_spine_recall
        obj, bm = self.get_obj_bm(context)
        temp_bm = False
        if obj is None:
            return
        if bm is None:
            if obj.type != 'MESH' or obj.data is None:
                return
            try:
                bm = bmesh.new()
                bm.from_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                temp_bm = True
            except Exception:
                return
        if getattr(self, 'spine_chains', None):
            try:
                self._spine_store_active_chain()
            except Exception:
                pass
        chains = getattr(self, 'spine_chains', None) or []
        # Persist the effective influence reach in the bind itself before
        # writing recall.  A controller radius can be increased during Deform
        # and reach vertices that were not part of the original bind.  If those
        # vertices are not absorbed into the bind before leaving the tool, the
        # next tool activation restores the old bind and the same radius appears
        # to have "forgotten" its reach.
        try:
            for _ci, _ch in enumerate(chains):
                _pinf = _ch.get('point_influence') or []
                if not _pinf:
                    continue
                _active_inf = max([float(x) for x in _pinf] + [0.0])
                if _active_inf > 1e-8:
                    self._spine_absorb_vg_into_bind(obj, bm, _ch, _ci)
        except Exception:
            pass
        if not chains:
            # single-chain fallback
            if not getattr(self, 'bez', None) or len(self.bez) < 2:
                return
            if not getattr(self, 'spine_bind', None):
                return
            chains = [{
                'bez': self.bez,
                'rest_bez': getattr(self, 'rest_bez', None) or self.bez,
                'modes': getattr(self, 'point_modes', []),
                'tilt': getattr(self, 'spine_tilt', []),
                'radius': getattr(self, 'spine_radius', []),
                'handle_params': getattr(self, 'handle_params', []),
                'bind': self.spine_bind,
                'influence': getattr(self, 'spine_influence', 0.1),
                'point_influence': ensure_point_influence(
                    len(self.bez), getattr(self, 'point_influence', None),
                    default=float(getattr(self, 'spine_influence', 0.1) or 0.1),
                ),
                'point_influence_default': ensure_point_influence(
                    nbez, ch.get('point_influence_default'),
                    default=(ch.get('point_influence')[0] if ch.get('point_influence') else float(ch.get('influence', 0.1) or 0.1)),
                ),
                'point_inf_falloff': ensure_point_inf_falloff(
                    len(self.bez), getattr(self, 'point_inf_falloff', None), default='CONSTANT'
                ),
                'attr_interp': (str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper()
                                if str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper() in _ATTR_INTERP_ORDER
                                else 'SMOOTH'),
            }]
        bm.verts.ensure_lookup_table()
        _original_coords = {v.index: v.co.copy() for v in bm.verts}
        try:
            attr_layers = self._spine_capture_attr_layers(context, _original_coords)
        except Exception:
            attr_layers = {'tilt': {}, 'inflate': {}}
        chains_out = []
        mesh_snap = {}
        rest_snap = {}
        # Preserve the actual deformation rest base separately from the current
        # confirmed/deformed mesh.  _spine_apply() always starts from all_rest;
        # restoring all_rest from the current mesh makes the first post-reopen
        # transform effectively double-apply the saved deformation.
        try:
            all_rest_snap = {int(k): v.copy() for k, v in (getattr(self, 'all_rest', {}) or {}).items()}
        except Exception:
            all_rest_snap = {}
        for ch in chains:
            bez = ch.get('bez')
            rest_bez = ch.get('rest_bez') or bez
            if not bez or len(bez) < 2:
                continue
            bind_copy = []
            for item in (ch.get('bind') or []):
                vidx = item[0]
                t = float(item[1])
                offset = item[2].copy() if item[2] is not None else None
                tan = item[3].copy() if len(item) > 3 and item[3] is not None else None
                dist = float(item[4]) if len(item) > 4 else 0.0
                bind_copy.append((vidx, t, offset, tan, dist))
                if vidx < len(bm.verts):
                    mesh_snap[vidx] = bm.verts[vidx].co.copy()
                    if offset is not None and rest_bez:
                        on = eval_bezier_points(rest_bez, t)
                        rest_snap[vidx] = on + offset
            nbez = len(bez)
            chains_out.append({
                'bez': copy_bezier_points(bez),
                'rest_bez': copy_bezier_points(rest_bez),
                'modes': list(ch.get('modes') or ['AUTO'] * nbez),
                'tilt': list(ch.get('tilt') or [0.0] * nbez),
                'radius': list(ch.get('radius') or [1.0] * nbez),
                'handle_params': list(ch.get('handle_params') or []),
                'bind': bind_copy,
                'influence': float(ch.get('influence', 0.1) or 0.1),
                'point_influence': ensure_point_influence(
                    nbez, ch.get('point_influence'),
                    default=float(ch.get('influence', 0.1) or 0.1),
                ),
                'point_influence_default': ensure_point_influence(
                    nbez, ch.get('point_influence_default'),
                    default=(ch.get('point_influence')[0]
                             if ch.get('point_influence')
                             else float(ch.get('influence', 0.1) or 0.1)),
                ),
                'point_inf_falloff': ensure_point_inf_falloff(
                    nbez, ch.get('point_inf_falloff'), default='CONSTANT'
                ),
                # Persist the Shrink/Inflate/Tilt interpolation falloff in the
                # actual recall payload.  Without this field, the runtime chain
                # could show the previous falloff until the first controller
                # drag, then a reload would silently fall back to SMOOTH.
                'attr_interp': (
                    str(ch.get('attr_interp') or getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper()
                    if str(ch.get('attr_interp') or getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper() in _ATTR_INTERP_ORDER
                    else 'SMOOTH'
                ),
                'origin_ids': list(ch.get('origin_ids') or list(range(nbez))),
                'chain_id': ch.get('chain_id'),
                'in_front': bool(ch.get('in_front', True)),
                'vg_name': ch.get('vg_name') or '',
            })
        if not chains_out:
            if temp_bm:
                try:
                    bm.free()
                except Exception:
                    pass
            return
        entry = {
            'chains': chains_out,
            # Confirmed without Rest bake: mesh_snap is the current deformed
            # mesh, while each chain's rest_bez remains the true deformation
            # rest and point_influence remains the live controller reach.
            'confirm_preserves_deform': True,
            'active_chain': int(getattr(self, 'active_chain', 0) or 0),
            'display_scale': float(getattr(self, 'display_scale', 1.0) or 1.0),
            'vert_count': len(obj.data.vertices),
            'mesh_snap': mesh_snap,
            'rest_snap': rest_snap,
            'all_rest_snap': all_rest_snap,
            # Keep the original Spine rest forever, even when an external
            # Sculpt/Edit change is later rebased as the new working base.
            'true_rest_snap': all_rest_snap,
            'attr_layers': attr_layers,
        }
        # Keep the previous confirm so a later Blender Undo can restore that pose.
        # RAM is not reverted by Ctrl+Z; mesh custom props often are.
        try:
            key = _vdh_cache_key(obj)
            old = _vdh_spine_recall.get(key) if key is not None else None
            if not old:
                old = _vdh_load_recall_from_mesh(obj)
            if old and old is not entry:
                prev = dict(old)
                prev.pop('prev', None)
                # Skip prev if it is the same mesh pose (repeat save / no-op)
                old_snap = old.get('mesh_snap') or {}
                same_pose = False
                if old_snap and mesh_snap and len(old_snap) == len(mesh_snap):
                    acc = 0.0
                    ncmp = 0
                    for vidx, co in mesh_snap.items():
                        oco = old_snap.get(vidx)
                        if oco is None:
                            continue
                        acc += (co - oco).length
                        ncmp += 1
                    if ncmp > 0 and (acc / ncmp) < 1e-7:
                        same_pose = True
                if not same_pose:
                    entry['prev'] = prev
                elif old.get('prev'):
                    entry['prev'] = old.get('prev')
        except Exception:
            pass
        # Once a true Spine rest exists, never replace it with a later
        # externally-edited working mesh.  The working base may change, but
        # Inflate/Shrink and Reset operations still need the original rest.
        try:
            _key_now = _vdh_cache_key(obj)
            _old_recall = _vdh_spine_recall.get(_key_now) if _key_now is not None else None
            if not _old_recall:
                _old_recall = _vdh_load_recall_from_mesh(obj)
            if _old_recall and _old_recall.get('true_rest_snap'):
                entry['true_rest_snap'] = _old_recall.get('true_rest_snap')
        except Exception:
            pass
        _vdh_set_recall(obj, entry)
        # Persist inside .blend so reopen restores chains
        try:
            _vdh_persist_recall_to_mesh(obj, entry)
        except Exception:
            pass
        if temp_bm:
            try:
                bm.free()
            except Exception:
                pass

    def _spine_remove_new_topology_weights(self, obj, previous_vert_ids, chains=None):
        """Clean Blender-copied BH_Spine weights after an Edit Mode topology duplicate.

        Blender itself copies vertex-group assignments to duplicated vertices. Blue Handles
        must keep only the assignments that correspond to an actual chain deformation.
        Therefore, for NEW vertices only, remove a BH_Spine group's copied weight when that
        vertex is NOT present in that chain's rebuilt bind list. Existing vertices and any
        intentionally painted weights are left untouched.
        """
        if obj is None or obj.type != 'MESH' or not obj.vertex_groups:
            return
        previous = set(int(v) for v in (previous_vert_ids or set()))
        if not previous:
            return
        new_indices = {v.index for v in obj.data.vertices if v.index not in previous}
        if not new_indices:
            return
        chains = list(chains or [])
        if not chains:
            return
        for ch in chains:
            name = ch.get('vg_name') or ''
            if not name:
                continue
            vg = obj.vertex_groups.get(name)
            if vg is None:
                continue
            bound = {int(item[0]) for item in (ch.get('bind') or [])}
            remove = list(new_indices - bound)
            if not remove:
                continue
            try:
                vg.remove(remove)
            except Exception:
                pass

    def _spine_sync_bound_vertex_group_ownership(self, obj, chains=None):
        """Make BH_Spine* groups exactly match the chain vertices that actually deform.

        After Edit Place/retopology Blender can preserve copied vertex-group weights.
        Those stale weights must not survive on a chain group merely because the vertex
        used to belong to it. The rebuilt chain bind is the source of truth here: every
        vertex in a chain's bind belongs to that chain's BH_Spine* group, and every
        vertex outside that bind is removed from that group. This also transfers overlap
        cleanly when a new chain takes ownership of a duplicated mesh island.
        """
        if obj is None or obj.type != 'MESH':
            return
        chains = list(chains or [])
        if not chains:
            return
        for i, ch in enumerate(chains):
            ch['vg_name'] = _vdh_spine_vg_name(i, ch.get('vg_name'))
        groups = []
        for ch in chains:
            name = ch.get('vg_name') or ''
            groups.append(_vdh_ensure_vertex_group(obj, name) if name else None)

        # Preserve the user's existing Weight Paint values while rebuilding ownership.
        # The previous implementation cleared every BH_Spine group and then wrote 1.0
        # to the rebuilt bind. That silently destroyed painted weights during
        # Edit Place -> Enter/Rebind. Blender Undo could restore them, which made it
        # look as if the weights only started updating after an undo.
        #
        # Snapshot the strongest existing BH_Spine weight for each vertex first. If a
        # vertex was already owned by more than one stale/copied group, keeping the
        # maximum is the least destructive choice and preserves the visible Weight
        # Paint result. New vertices with no previous weight fall back to 1.0 below.
        preserved_w = {}
        for own in groups:
            if own is None:
                continue
            try:
                for v in obj.data.vertices:
                    try:
                        w = float(own.weight(v.index))
                    except Exception:
                        continue
                    if w > preserved_w.get(v.index, 0.0):
                        preserved_w[v.index] = max(0.0, min(1.0, w))
            except Exception:
                pass

        # The rebuilt bind lists are authoritative. Clear every BH_Spine group first,
        # then write exactly the vertices owned by that chain. This is deliberately
        # limited to BH_Spine* groups so user-created/non-spine groups are untouched.
        all_verts = list(range(len(obj.data.vertices)))
        for own in groups:
            if own is None:
                continue
            try:
                if all_verts:
                    own.remove(all_verts)
            except Exception:
                pass

        for ci, ch in enumerate(chains):
            own = groups[ci]
            if own is None:
                continue
            bound_ids = {int(item[0]) for item in (ch.get('bind') or [])}
            if not bound_ids:
                continue
            try:
                weighted = []
                unweighted = []
                for vidx in bound_ids:
                    w = preserved_w.get(vidx, 0.0)
                    if w > 1e-8:
                        weighted.append((vidx, w))
                    else:
                        unweighted.append(vidx)
                # Restore painted values exactly; only genuinely new/unweighted
                # vertices receive the normal bind seed of 1.0.
                for vidx, w in weighted:
                    own.add([vidx], w, 'REPLACE')
                if unweighted:
                    own.add(unweighted, 1.0, 'REPLACE')
            except Exception:
                pass

    def _spine_resync_chain_bind_from_mesh(self, bm, ch):
        """Rebuild one chain's offsets from current mesh; keep t weights."""
        bm.verts.ensure_lookup_table()
        rest_bez = ch.get('rest_bez') or ch.get('bez')
        new_bind = []
        for item in (ch.get('bind') or []):
            vidx = item[0]
            t = float(item[1])
            if vidx >= len(bm.verts):
                continue
            on = eval_bezier_points(rest_bez, t)
            tan = bezier_chain_tangent(rest_bez, t)
            offset = bm.verts[vidx].co - on
            new_bind.append((vidx, t, offset.copy(), tan.copy(), self._spine_item_radial_dist((vidx, t, offset, tan, 0.0))))
            self.all_rest[vidx] = bm.verts[vidx].co.copy()
        ch['bind'] = new_bind

    def _spine_resync_bind_from_mesh(self, bm, rest_bez):
        """Rebuild offsets/tangents from current mesh vs rest_bez, keep t weights."""
        bm.verts.ensure_lookup_table()
        new_bind = []
        for item in self.spine_bind:
            vidx = item[0]
            t = float(item[1])
            if vidx >= len(bm.verts):
                continue
            on = eval_bezier_points(rest_bez, t)
            tan = bezier_chain_tangent(rest_bez, t)
            offset = bm.verts[vidx].co - on
            dist = offset.length
            new_bind.append((vidx, t, offset.copy(), tan.copy(), dist))
            self.all_rest[vidx] = bm.verts[vidx].co.copy()
        self.spine_bind = new_bind

    def _spine_try_auto_restore(self, context):
        """On tool start: restore last multi-chain spine (no rebind).

        RAM cache is NOT reverted by Blender Undo. Mesh custom properties usually
        are. Pick the recall snapshot whose mesh_snap best matches the current
        mesh so Ctrl+Z after Confirm restores the previous spine pose too.
        """
        global _vdh_spine_recall
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return False
        key = _vdh_cache_key(obj)
        ram_data = _vdh_spine_recall.get(key) if key is not None else None
        mesh_data = None
        try:
            mesh_data = _vdh_load_recall_from_mesh(obj)
        except Exception:
            mesh_data = None

        bm.verts.ensure_lookup_table()
        cur_vc = len(bm.verts)
        scale = max((obj.dimensions.length if obj.dimensions.length > 1e-6 else 1.0), 1e-3)
        match_tol = scale * 1e-4

        best = None
        best_err = None
        for cand in _vdh_iter_recall_candidates(mesh_data, ram_data):
            err = _vdh_recall_snap_error(bm, cand.get('mesh_snap') or {})
            if err is None:
                continue
            if best_err is None or err < best_err:
                best_err = err
                best = cand

        # Prefer a snapshot that actually matches current verts (Blender Undo).
        data = None
        undid = False
        snap_lim = max(match_tol * 10.0, 1e-6)
        if best is not None and best_err is not None and best_err <= snap_lim:
            data = best
            ram_err = _vdh_recall_snap_error(bm, (ram_data or {}).get('mesh_snap') or {})
            if ram_data is not None and ram_data is not best:
                undid = True
            elif ram_err is not None and best_err < ram_err * 0.5 and ram_err > snap_lim:
                undid = True
        elif mesh_data:
            # External edit / topology: use persisted recall and re-sync later
            data = mesh_data
        elif ram_data:
            ram_err = _vdh_recall_snap_error(bm, ram_data.get('mesh_snap') or {})
            if ram_err is not None and ram_err > snap_lim:
                # Stale RAM after Blender Undo of the first Confirm — start fresh
                return False
            data = ram_data
        else:
            data = None

        if data:
            try:
                _vdh_set_recall(obj, data)
            except Exception:
                pass
        if not data:
            return False

        saved_vc = int(data.get('vert_count', -1))
        # Topology change (e.g. duplicate finger in Edit Mode): still restore chains
        topology_changed = (saved_vc >= 0 and saved_vc != cur_vc)

        # Prefer multi-chain format
        chains_data = data.get('chains')
        if not chains_data:
            # Legacy single-chain → wrap
            if not data.get('bez') or not data.get('spine_bind'):
                return False
            chains_data = [{
                'bez': data['bez'],
                'rest_bez': data.get('rest_bez') or data['bez'],
                'modes': data.get('modes'),
                'tilt': data.get('tilt'),
                'radius': data.get('radius'),
                'handle_params': data.get('handle_params'),
                'bind': data['spine_bind'],
                'influence': data.get('spine_influence', 0.1),
                'attr_interp': data.get('attr_interp') or 'SMOOTH',
            }]

        mesh_snap = data.get('mesh_snap') or {}
        rest_snap = data.get('rest_snap') or {}
        err_deform = best_err
        if err_deform is None:
            err_deform = _vdh_recall_snap_error(bm, mesh_snap)
        if err_deform is None:
            err_deform = 0.0
            n_cmp = 0
        else:
            n_cmp = len(mesh_snap)
        err_rest = 0.0
        n_rest = 0
        for vidx, co_snap in rest_snap.items():
            if vidx >= cur_vc:
                continue
            try:
                err_rest += (bm.verts[vidx].co - co_snap).length
                n_rest += 1
            except Exception:
                continue
        if n_rest > 0:
            err_rest /= n_rest
        else:
            err_rest = err_deform
        tol = match_tol
        # If the persisted recall exists but the current mesh does not match ANY
        # recall snapshot, this is an edit made while Blue Handles was closed
        # (Sculpt/Edit Mesh/etc.).  Do not let the older undo heuristic classify
        # it as Blender Undo merely because the sculpted mesh happens to be closer
        # to the true rest than to the last deformed pose.
        external_candidate = bool(
            (not topology_changed) and mesh_data is not None
            and (best_err is None or best_err > max(tol * 10.0, 1e-6))
        )
        if external_candidate:
            undid = False
        elif not undid:
            undid = (
                (not topology_changed) and n_cmp > 0
                and err_deform > max(tol, err_rest * 1.5)
                and err_rest < err_deform
            )
        # After topology change always re-sync offsets to current mesh.
        # If we already picked a snapshot that matches current verts, do NOT
        # resync (that used to keep the post-Confirm curve on the undone mesh).
        snap_matches = (best_err is not None and best_err <= max(tol * 10.0, 1e-6))
        need_resync = topology_changed or (
            (not snap_matches) and (undid or (n_cmp > 0 and err_deform > tol * 10))
        )
        # A mesh change while the tool was closed is an external base edit.
        # IMPORTANT: for now, the external mesh itself is the new Spine REST.
        # Do NOT try to reconstruct the old Tilt/Inflate layers here. That was
        # the source of the disappearing Sculpt/Edit changes: the first drag
        # could still rebuild vertices from the old rest pose.
        #
        # The rule for this version is intentionally simple: if the mesh was
        # edited while Blue Handles was closed, the exact current vertex state
        # becomes the new rest/base, while the controller positions are kept.
        # Inflate/Shrink preservation can be designed separately afterwards.
        external_base_edit = bool(
            (not topology_changed) and external_candidate and n_cmp > 0
            and err_deform > max(tol * 10.0, 1e-6)
        )
        if external_base_edit:
            need_resync = False

        restored = []
        total_ctrl = 0
        for ch in chains_data:
            bez = copy_bezier_points(ch['bez'])
            rest = copy_bezier_points(ch.get('rest_bez') or ch['bez'])
            modes = list(ch.get('modes') or ['AUTO'] * len(bez))
            while len(modes) < len(bez):
                modes.append('AUTO')
            tilt = list(ch.get('tilt') or [0.0] * len(bez))
            radius = list(ch.get('radius') or [1.0] * len(bez))
            while len(tilt) < len(bez):
                tilt.append(0.0)
            while len(radius) < len(bez):
                radius.append(1.0)
            hparams = list(ch.get('handle_params') or [])
            if len(hparams) != len(bez):
                n = len(bez)
                hparams = [i / max(1, n - 1) for i in range(n)]
                if hparams:
                    hparams[0], hparams[-1] = 0.0, 1.0
            bind = []
            for item in (ch.get('bind') or []):
                vidx, t = item[0], float(item[1])
                if vidx >= len(bm.verts):
                    continue
                offset = item[2].copy() if item[2] is not None else Vector((0, 0, 0))
                tan = item[3].copy() if len(item) > 3 and item[3] is not None else Vector((0, 0, 1))
                dist = float(item[4]) if len(item) > 4 else offset.length
                bind.append((vidx, t, offset, tan, dist))
            entry = {
                'chain_id': ch.get('chain_id') or f"chain_restored_{len(restored)}",
                'bez': bez,
                'rest_bez': rest,
                'modes': modes,
                'tilt': tilt,
                'radius': radius,
                'handle_params': hparams,
                'bind': bind,
                'influence': float(ch.get('influence', 0.1) or 0.1),
                'point_influence': ensure_point_influence(
                    len(bez), ch.get('point_influence'),
                    default=float(ch.get('influence', 0.1) or 0.1),
                ),
                 'point_influence_default': ensure_point_influence(
                     len(bez), ch.get('point_influence_default'),
                     default=(ch.get('point_influence')[0] if ch.get('point_influence') else float(ch.get('influence', 0.1) or 0.1)),
                 ),
                'point_inf_falloff': ensure_point_inf_falloff(
                    len(bez), ch.get('point_inf_falloff'), default='CONSTANT'
                ),
                # IMPORTANT: restore the Shrink/Inflate/Tilt interpolation
                # falloff into the live chain state.  The previous version
                # serialized attr_interp but forgot to put it back into the
                # restored chain dictionary.  That made _spine_load_active_chain
                # fall back to SMOOTH as soon as the user clicked/drug a
                # controller after reopening the tool.
                'attr_interp': (str(ch.get('attr_interp') or data.get('attr_interp') or 'SMOOTH').upper()
                                if str(ch.get('attr_interp') or data.get('attr_interp') or 'SMOOTH').upper() in _ATTR_INTERP_ORDER
                                else 'SMOOTH'),
                'origin_ids': list(ch.get('origin_ids') or list(range(len(bez)))),
                'in_front': bool(ch.get('in_front', True)),
                'vg_name': ch.get('vg_name') or _vdh_spine_vg_name(len(restored)),
            }
            restored.append(entry)
            total_ctrl += len(bez)

        if not restored:
            return False

        self.spine_chains = restored
        self.active_chain = int(data.get('active_chain', 0) or 0) % len(restored)
        self.display_scale = float(data.get('display_scale', 1.0) or 1.0)
        # Place-mode lists only — empty in Deform to avoid duplicating active chain
        self.spine_chains_pts = []
        self._spine_chains_origin_ids = []
        self._spine_edit_place = False

        # Always keep saved controller poses (bez / tilt / radius / modes).
        # Never reset bez → rest_bez (that made chains "forget" last shape).
        if need_resync:
            for ch in self.spine_chains:
                # Identity bake: rest matches saved controllers; offsets from current mesh
                ch['rest_bez'] = copy_bezier_points(ch['bez'])
                self._spine_resync_chain_bind_from_mesh(bm, ch)
            if topology_changed:
                msg = f"Spine resumed (mesh changed): {len(self.spine_chains)} chain(s), {total_ctrl} controllers"
            elif undid:
                msg = f"Spine resumed (mesh undo, re-synced): {len(self.spine_chains)} chain(s), {total_ctrl} controllers"
            else:
                msg = f"Spine resumed (re-synced): {len(self.spine_chains)} chain(s)"
        elif undid:
            msg = f"Spine resumed (Blender undo): {len(self.spine_chains)} chain(s), {total_ctrl} controllers"
        else:
            msg = f"Spine resumed: {len(self.spine_chains)} chain(s), {total_ctrl} controllers"

        # IMPORTANT: use the proven re-entry behavior from the older working
        # version.  The mesh that exists RIGHT NOW is authoritative.  Rebuild
        # each chain's bind offsets from the current mesh, then make that same
        # current mesh the working all_rest.  This means edits made in Edit Mode
        # / Sculpt while Blue Handles is closed are preserved when the first
        # controller is dragged.  We intentionally do not restore an older
        # rest snapshot over the visible mesh here.
        for ch in self.spine_chains:
            try:
                # Keep the saved controller pose and attributes; only refresh
                # the positional bind offset against the mesh that is visible.
                self._spine_resync_chain_bind_from_mesh(bm, ch)
                ch['_fw'] = None
                ch['_sw_key'] = None
                ch['_soft_w'] = None
            except Exception:
                pass

        self._spine_load_active_chain()
        # Every chain is In Front on every fresh entry/re-entry to Spine Mode.
        try:
            for _ch in (getattr(self, 'spine_chains', None) or []):
                _ch['in_front'] = True
        except Exception:
            pass
        self.spine_in_front = True
        # Auto-restore is a direct resume of the existing Spine session.
        # _invoke_spine starts in SPINE_PLACE while probing recall, so make the
        # successful restore explicitly enter Deform; otherwise a valid recall
        # can appear as Place/Edit-Place even though chains were restored.
        self.tool_mode = 'SPINE_DEFORM'
        self._spine_edit_place = False
        self._spine_placing_new_chain = False
        self.active_handle = 0 if self.bez else None
        self.active_bez_part = 'co'
        self.dragging = False
        self._pending_click_drag = False
        self._pending_place_drag = False
        self._xform_mode = None
        self.constraint_axis = None
        # The visible mesh at re-entry is the working base.
        self.all_rest = {v.index: v.co.copy() for v in bm.verts}
        self.initial_all_rest = {k: v.copy() for k, v in self.all_rest.items()}
        self._spine_reentry_external_pending = False
        # Do NOT bake here — that zeros tilt/radius and makes chains look wrong.
        self._spine_session_rest_bez = copy_bezier_points(self.bez)
        self._spine_session_rest_tilt = list(self.spine_tilt)
        self._spine_session_rest_radius = list(self.spine_radius)
        self._spine_session_rest_modes = list(self.point_modes)
        # Re-link chain_id from bind-rest via vert overlap (survives duplicate)
        try:
            br = _vdh_get_bind_rest(obj)
            saved_list = br.get('chains') or []
            saved_by_id = {s.get('chain_id'): s for s in saved_list if s.get('chain_id')}
            for ch in self.spine_chains:
                s = self._spine_match_saved_for_chain(ch, saved_list, saved_by_id)
                if s and s.get('chain_id'):
                    ch['chain_id'] = s['chain_id']
            self._spine_merge_bind_rest(context)
        except Exception:
            pass
        try:
            self._spine_sync_weight_groups(context, mode='SYNC')
        except Exception:
            pass
        # Only rewrite recall when topology changed. After Blender Undo the
        # matching snapshot is already correct — saving here used to overwrite
        # the undone pose with the last Confirm curve.
        if topology_changed:
            try:
                self._spine_save_recall(context)
            except Exception:
                pass
        self.report({'INFO'}, msg)
        return True


    def _spine_reset_chain_controllers(self, ch, saved_rest_bez):
        """Restore original controllers to bind rest; align inserted ones on the rest line
        while preserving relative spacing between neighbors.
        """
        rest_cos = [bp['co'].copy() for bp in saved_rest_bez]
        n_rest = len(rest_cos)
        if n_rest < 2:
            return False
        bez = ch.get('bez') or []
        m = len(bez)
        if m < 2:
            return False
        oids = list(ch.get('origin_ids') or [])
        if len(oids) != m:
            # no tracking: if same count, 1:1; if more, treat extras as new between
            if m == n_rest:
                oids = list(range(m))
            else:
                # map first n_rest as original, extras as None distributed... simple: all None except ends
                oids = [None] * m
                oids[0] = 0
                oids[-1] = n_rest - 1
                # assign remaining originals evenly? better: sequential match min(m,n_rest)
                for i in range(min(m, n_rest)):
                    oids[i] = i
                for i in range(n_rest, m):
                    oids[i] = None

        cur_cos = [bp['co'].copy() for bp in bez]
        new_cos = [None] * m
        for i, oid in enumerate(oids):
            if oid is not None and 0 <= int(oid) < n_rest:
                new_cos[i] = rest_cos[int(oid)].copy()

        # Cumulative length on current polyline
        def cumlen(pts):
            L = [0.0]
            for i in range(1, len(pts)):
                L.append(L[-1] + (pts[i] - pts[i - 1]).length)
            return L

        cur_L = cumlen(cur_cos)
        # Fill gaps between anchored controllers
        i = 0
        while i < m:
            if new_cos[i] is not None:
                i += 1
                continue
            # find left anchor
            left = i - 1
            while left >= 0 and new_cos[left] is None:
                left -= 1
            right = i
            while right < m and new_cos[right] is None:
                right += 1
            if left < 0 and right >= m:
                # nothing anchored — fall back to rest endpoints
                for j in range(m):
                    t = j / max(1, m - 1)
                    new_cos[j] = rest_cos[0].lerp(rest_cos[-1], t)
                break
            if left < 0:
                # before first anchor: place along rest from start to anchor
                a_rest = rest_cos[0]
                b_rest = new_cos[right]
                a_cur, b_cur = cur_cos[0], cur_cos[right]
                seg = max((b_cur - a_cur).length, 1e-12)
                for j in range(0, right):
                    frac = (cur_cos[j] - a_cur).length / seg if j > 0 else 0.0
                    # use cumulative
                    frac = (cur_L[j] - cur_L[0]) / max(cur_L[right] - cur_L[0], 1e-12)
                    new_cos[j] = a_rest.lerp(b_rest, max(0.0, min(1.0, frac)))
                i = right
                continue
            if right >= m:
                a_rest = new_cos[left]
                b_rest = rest_cos[-1]
                denom = max(cur_L[-1] - cur_L[left], 1e-12)
                for j in range(left + 1, m):
                    frac = (cur_L[j] - cur_L[left]) / denom
                    new_cos[j] = a_rest.lerp(b_rest, max(0.0, min(1.0, frac)))
                break
            # between two anchors
            a_rest = new_cos[left]
            b_rest = new_cos[right]
            denom = max(cur_L[right] - cur_L[left], 1e-12)
            for j in range(left + 1, right):
                frac = (cur_L[j] - cur_L[left]) / denom
                new_cos[j] = a_rest.lerp(b_rest, max(0.0, min(1.0, frac)))
            i = right

        for j in range(m):
            if new_cos[j] is None:
                new_cos[j] = cur_cos[j].copy()

        # Write positions and rebuild AUTO handles
        for j, bp in enumerate(ch['bez']):
            bp['co'] = new_cos[j]
        # Use temporary self.bez for rebuild
        return True


    def _spine_rebind_chain_from_mesh(self, bm, ch):
        """Full rebind of one chain from current mesh positions (identity at rest)."""
        if not ch.get('bez') or len(ch['bez']) < 2:
            return
        ch['rest_bez'] = copy_bezier_points(ch['bez'])
        n = len(ch['bez'])
        if len(ch.get('tilt') or []) != n:
            ch['tilt'] = [0.0] * n
        if len(ch.get('radius') or []) != n:
            ch['radius'] = [1.0] * n
        cos = [bp['co'].copy() for bp in ch['rest_bez']]
        lengths = [0.0]
        for i in range(1, len(cos)):
            lengths.append(lengths[-1] + (cos[i] - cos[i - 1]).length)
        total = lengths[-1] if lengths[-1] > 1e-12 else 1.0
        hparams = [L / total for L in lengths]
        if hparams:
            hparams[0], hparams[-1] = 0.0, 1.0
        ch['handle_params'] = hparams

        samples = max(64, n * 32)
        sample_pts = [eval_bezier_points(ch['rest_bez'], s / samples) for s in range(samples + 1)]
        influence = float(ch.get('influence') or 0.0)
        if influence < 1e-8:
            # estimate from current bind distances or bbox
            influence = 1e18
            for item in (ch.get('bind') or []):
                if len(item) > 4:
                    influence = min(influence, max(float(item[4]) * 2.0, 1e-4))
            if influence > 1e17:
                influence = 1.0
            ch['influence'] = influence

        # Prefer previous bound verts; if empty, find near curve
        prev = [item[0] for item in (ch.get('bind') or [])]
        if not prev:
            prev = [v.index for v in bm.verts]
        new_bind = []
        for vidx in prev:
            if vidx >= len(bm.verts):
                continue
            co = bm.verts[vidx].co
            best_t, best_d = 0.0, 1e18
            for s, p in enumerate(sample_pts):
                d = (p - co).length_squared
                if d < best_d:
                    best_d = d
                    best_t = s / samples
            best_d = math.sqrt(best_d)
            if best_d > influence * 1.5:
                continue
            on = eval_bezier_points(ch['rest_bez'], best_t)
            tan = bezier_chain_tangent(ch['rest_bez'], best_t)
            offset = co - on
            new_bind.append((vidx, best_t, offset.copy(), tan.copy(), float(best_d)))
            self.all_rest[vidx] = co.copy()
        ch['bind'] = new_bind
        try:
            self._spine_unify_bind_rings(bm, ch)
        except Exception:
            pass

    def _spine_reset_controllers(self, context, active_only=True):
        """Reset chain(s) to their first-bind snapshot (by chain_id) + full rebind.

        Works for chains added later (Alt+Enter → new chain → Enter).
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            self.report({'INFO'}, "Reset: nothing to restore")
            return False
        data = _vdh_get_bind_rest(obj)
        if not data or not data.get('chains'):
            self.report({'INFO'}, "Reset: no bind state (bind once first)")
            return False
        if not getattr(self, 'spine_chains', None):
            self.report({'INFO'}, "Reset: no active chains")
            return False

        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        self._spine_ensure_chain_ids()
        self._spine_push_undo(context)

        saved_list = data['chains']
        saved_by_id = {s.get('chain_id'): s for s in saved_list if s.get('chain_id')}
        mesh_rest = data.get('mesh_rest') or {}

        if active_only:
            indices = [int(getattr(self, 'active_chain', 0) or 0)]
            if indices[0] < 0 or indices[0] >= len(self.spine_chains):
                self.report({'INFO'}, "Reset: no active chain")
                return False
        else:
            indices = list(range(len(self.spine_chains)))

        bm.verts.ensure_lookup_table()
        init = getattr(self, 'initial_all_rest', None) or {}

        restored = 0
        for i in indices:
            ch = self.spine_chains[i]
            s = self._spine_match_saved_for_chain(ch, saved_list, saved_by_id)
            if s is None and i < len(saved_list):
                s = saved_list[i]
            if not s or not s.get('bez') or len(s['bez']) < 2:
                self.report({'INFO'}, f"Reset: chain {i + 1} has no first-bind snapshot")
                continue
            # re-attach stable id
            if s.get('chain_id'):
                ch['chain_id'] = s['chain_id']

            # verts: prefer live bind, else saved bind_verts
            verts = [item[0] for item in (ch.get('bind') or [])]
            if not verts:
                verts = list(s.get('bind_verts') or [])
            for vidx in verts:
                if vidx >= len(bm.verts):
                    continue
                if vidx in mesh_rest:
                    bm.verts[vidx].co = mesh_rest[vidx].copy()
                    self.all_rest[vidx] = mesh_rest[vidx].copy()
                elif vidx in init:
                    bm.verts[vidx].co = init[vidx].copy()
                    self.all_rest[vidx] = init[vidx].copy()

            # exact controller restore
            ch['bez'] = copy_bezier_points(s['bez'])
            ch['rest_bez'] = copy_bezier_points(s['bez'])
            n = len(ch['bez'])
            ch['modes'] = list(s.get('modes') or ['AUTO'] * n)[:n]
            while len(ch['modes']) < n:
                ch['modes'].append('AUTO')
            ch['tilt'] = list(s.get('tilt') or [0.0] * n)[:n]
            while len(ch['tilt']) < n:
                ch['tilt'].append(0.0)
            ch['radius'] = list(s.get('radius') or [1.0] * n)[:n]
            while len(ch['radius']) < n:
                ch['radius'].append(1.0)
            oids = list(s.get('origin_ids') or list(range(n)))
            ch['origin_ids'] = oids if len(oids) == n else list(range(n))
            if s.get('influence'):
                ch['influence'] = float(s['influence'])
            if s.get('chain_id'):
                ch['chain_id'] = s['chain_id']

            self._spine_rebind_chain_from_mesh(bm, ch)
            restored += 1

        if restored == 0:
            self.report({'INFO'}, "Reset: nothing restored")
            return False

        self._spine_load_active_chain()
        self._spine_apply(context, auto_soft=False)

        # Guard mesh to mesh_rest for all restored chain verts
        for i in indices:
            if i >= len(self.spine_chains):
                continue
            for item in (self.spine_chains[i].get('bind') or []):
                vidx = item[0]
                if vidx in mesh_rest and vidx < len(bm.verts):
                    bm.verts[vidx].co = mesh_rest[vidx].copy()
                    self.all_rest[vidx] = mesh_rest[vidx].copy()
        try:
            bm.normal_update()
        except Exception:
            pass
        try:
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        except Exception:
            pass

        if getattr(self, 'bez', None):
            self._spine_session_rest_bez = copy_bezier_points(self.bez)
            self._spine_session_rest_tilt = list(getattr(self, 'spine_tilt', []) or [])
            self._spine_session_rest_radius = list(getattr(self, 'spine_radius', []) or [])
            self._spine_session_rest_modes = list(getattr(self, 'point_modes', []) or [])

        context.area.tag_redraw()
        if active_only:
            self.report({'INFO'}, f"Reset chain {indices[0] + 1} → first bind + rebind")
        else:
            self.report({'INFO'}, f"Reset {restored} chain(s) → first bind + rebind")
        return True

    def _spine_bake_current_as_rest(self, context):
        """After Confirm: treat current mesh + controller pose as the new rest.
        Prevents vertex jumps when the tool is reopened and controllers move again.
        """
        obj, bm = self.get_obj_bm(context)
        temp_bm = False
        if obj is None:
            return False
        if bm is None:
            # Confirm while already leaving Edit Mode: read mesh datablock
            try:
                bm = bmesh.new()
                bm.from_mesh(obj.data)
                temp_bm = True
            except Exception:
                return False
        bm.verts.ensure_lookup_table()
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        chains = getattr(self, 'spine_chains', None) or []
        if not chains and getattr(self, 'bez', None) and len(self.bez) >= 2:
            chains = [{
                'bez': self.bez,
                'rest_bez': getattr(self, 'rest_bez', None) or self.bez,
                'bind': getattr(self, 'spine_bind', None) or [],
                'tilt': getattr(self, 'spine_tilt', None),
                'radius': getattr(self, 'spine_radius', None),
                'modes': getattr(self, 'point_modes', None),
                'handle_params': getattr(self, 'handle_params', None),
                'influence': getattr(self, 'spine_influence', 0.1),
                'point_inf_falloff': getattr(self, 'point_inf_falloff', None),
            }]
            self.spine_chains = chains
        for _ci, ch in enumerate(chains):
            bez = ch.get('bez')
            if not bez or len(bez) < 2:
                continue

            # Commit Radius reach before replacing the rest curve.  Any
            # already-painted Spine vertices that became reachable during the
            # session are then part of the serialized bind.
            try:
                self._spine_absorb_vg_into_bind(obj, bm, ch, _ci)
            except Exception:
                pass

            # Current controller pose becomes rest pose
            ch['rest_bez'] = copy_bezier_points(bez)
            # Mesh already includes tilt/radius — neutralize attrs to avoid double apply
            n = len(bez)
            ch['tilt'] = [0.0] * n
            ch['radius'] = [1.0] * n
            # Rebuild offsets from current mesh vs new rest
            try:
                self._spine_resync_chain_bind_from_mesh(bm, ch)
            except Exception:
                # Fallback: rebuild offsets manually
                rest_bez = ch['rest_bez']
                new_bind = []
                for item in (ch.get('bind') or []):
                    vidx = item[0]
                    t = float(item[1])
                    if vidx >= len(bm.verts):
                        continue
                    on = eval_bezier_points(rest_bez, t)
                    tan = bezier_chain_tangent(rest_bez, t)
                    offset = bm.verts[vidx].co - on
                    new_bind.append((vidx, t, offset.copy(), tan.copy(), self._spine_item_radial_dist((vidx, t, offset, tan, 0.0))))
                ch['bind'] = new_bind
        # Soft-influence base = confirmed mesh
        self.all_rest = {v.index: v.co.copy() for v in bm.verts}
        self.initial_all_rest = {k: v.copy() for k, v in self.all_rest.items()}
        try:
            self._spine_load_active_chain()
        except Exception:
            pass
        if temp_bm:
            try:
                bm.free()
            except Exception:
                pass
        return True

    def finish(self, context, cancel=False):
        if getattr(self, 'tool_mode', '') == 'VERTEX':
            try:
                obj = context.object
                _vdh_set_vertex_display_scale(obj, getattr(self, 'display_scale', 1.0))
            except Exception:
                pass
        # Save recall without baking the current Spine deformation into Rest.
        # Confirm must preserve the current Shrink/Inflate state and controller
        # influence reach so reopening the tool continues from the same live
        # deformation state.
        # Also save from Edit Place if bound chains exist (Tab out mid-edit).
        if not cancel and getattr(self, 'tool_mode', '') in ('SPINE_DEFORM', 'SPINE_PLACE'):
            try:
                self._spine_store_active_chain()
            except Exception:
                pass
            if getattr(self, 'tool_mode', '') == 'SPINE_DEFORM' or (getattr(self, 'spine_chains', None) or []):
                # Restore the old Confirm behavior first: bake the current
                # Spine mesh + controller pose as the new Rest before leaving.
                # This makes any mesh state that exists at Confirm the exact
                # state used when the tool is entered again.
                try:
                    self._spine_bake_current_as_rest(context)
                except Exception:
                    pass
                try:
                    self._spine_save_recall(context)
                except Exception:
                    pass
                try:
                    self._spine_sync_weight_groups(context, mode='SYNC')
                except Exception:
                    pass

        # Always restore snap when leaving the tool (Enter or Esc)
        try:
            self._spine_restore_snap(context)
        except Exception:
            pass

        if cancel:
            obj, bm = self.get_obj_bm(context)
            if obj and bm:
                # Spine mode: restore entire mesh
                if getattr(self, 'tool_mode', 'VERTEX') in ('SPINE_PLACE', 'SPINE_DEFORM'):
                    if getattr(self, 'initial_all_rest', None):
                        for idx, co in self.initial_all_rest.items():
                            if idx < len(bm.verts):
                                bm.verts[idx].co = co
                else:
                    # Vertex mode: restore selection + proportional region
                    for i, v_i in enumerate(self.vert_indices):
                        if v_i < len(bm.verts) and i < len(self.initial_rest_local):
                            bm.verts[v_i].co = self.initial_rest_local[i]
                    if getattr(self, 'initial_all_rest', None):
                        for idx, co in self.initial_all_rest.items():
                            if idx < len(bm.verts) and idx not in self.vert_indices:
                                bm.verts[idx].co = co
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                # ESC in Vertex Mode must restore the pre-operation normal state
                # without changing face winding.  Recompute BMesh normals only.
                if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX':
                    try:
                        bm.normal_update()
                        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                        obj.data.update()
                    except Exception:
                        pass

        # Vertex Mode must preserve the mesh face winding.  Do not recalculate
        # face/vertex normals here: for an open single-sided mesh (for example
        # a Plane), any normal recalculation can choose the opposite side on
        # confirm.  Just refresh the edit mesh while preserving its winding.
        try:
            if getattr(self, 'tool_mode', '') == 'VERTEX':
                obj, bm = self.get_obj_bm(context)
                if obj is not None and bm is not None:
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                    obj.data.update()
            else:
                self._spine_recalc_normals(context)
        except Exception:
            try:
                obj, bm = self.get_obj_bm(context)
                if obj is not None and bm is not None:
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                    obj.data.update()
            except Exception:
                pass

        if self._draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        if getattr(self, "_draw_text_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_text_handle, 'WINDOW')
            self._draw_text_handle = None
        # Restore gizmos
        try:
            sp = context.space_data
            for attr, val in getattr(self, '_gizmo_backup', {}).items():
                if hasattr(sp, attr):
                    setattr(sp, attr, val)
        except Exception:
            pass
        global _active_vdh_op
        if _active_vdh_op is self:
            _active_vdh_op = None
        context.area.tag_redraw()

    def handle_world(self, obj, i, part='co'):
        return obj.matrix_world @ self.bez[i][part]


    def pick_handle(self, context, event, pixel_dist=None):
        """Return (chain_idx, index, part) or None.
        chain_idx is index into spine_chains (0 if single/legacy).
        Does not change active_chain (caller decides).
        Hit radius uses base display_scale (not the Vertex visual 0.65 shrink)
        so smaller on-screen dots stay easy to click.
        """
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return None
        # Visual size can be smaller in Vertex; pick stays generous
        if pixel_dist is None:
            pixel_dist = 28.0 if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX' else 22.0
        ds = max(0.5, min(3.0, float(getattr(self, 'display_scale', 1.0) or 1.0)))
        pixel_dist = pixel_dist * ds
        region = context.region
        rv3d = context.region_data
        mx, my = event.mouse_region_x, event.mouse_region_y

        chains = getattr(self, 'spine_chains', None) or []
        if chains and getattr(self, 'tool_mode', '') == 'SPINE_DEFORM':
            best, best_d = None, pixel_dist
            for ci, ch in enumerate(chains):
                bez = ch.get('bez') or []
                n = len(bez)
                for i, bp in enumerate(bez):
                    parts = ['co']
                    if i == 0:
                        parts.append('hr')
                    elif i == n - 1:
                        parts.append('hl')
                    else:
                        parts.extend(['hl', 'hr'])
                    for part in parts:
                        if part != 'co' and (bp[part] - bp['co']).length < 1e-8:
                            continue
                        wco = obj.matrix_world @ bp[part]
                        sc = view3d_utils.location_3d_to_region_2d(region, rv3d, wco)
                        if sc is None:
                            continue
                        dist = math.hypot(sc.x - mx, sc.y - my)
                        if part != 'co':
                            dist *= 0.85
                        if dist < best_d:
                            best_d = dist
                            best = (ci, i, part)
            return best

        if not getattr(self, 'bez', None):
            return None
        best, best_d = None, pixel_dist
        n = len(self.bez)
        for i, bp in enumerate(self.bez):
            parts = ['co']
            if i == 0:
                parts.append('hr')
            elif i == n - 1:
                parts.append('hl')
            else:
                parts.extend(['hl', 'hr'])
            for part in parts:
                if part != 'co' and (bp[part] - bp['co']).length < 1e-8:
                    continue
                wco = obj.matrix_world @ bp[part]
                sc = view3d_utils.location_3d_to_region_2d(region, rv3d, wco)
                if sc is None:
                    continue
                dist = math.hypot(sc.x - mx, sc.y - my)
                if part != 'co':
                    dist *= 0.85
                if dist < best_d:
                    best_d = dist
                    # Vertex / single curve: (index, part) — NOT 3-tuple
                    best = (i, part)
        return best

    def pick_curve_local(self, context, event, max_dist=0.03):
        """Return (t, local_point) if mouse near curve in view, else None.
        max_dist is approximate world distance threshold via view."""
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return None
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        inv = obj.matrix_world.inverted()
        # sample curve in world, find closest to ray
        best_t, best_dist, best_local = None, 1e18, None
        samples = 80
        for s in range(samples + 1):
            t = s / samples
            local_p = eval_bezier_points(self.bez, t)
            world_p = obj.matrix_world @ local_p
            # distance from point to ray
            w = world_p - ray_origin
            # project
            proj = w.dot(view_vec)
            closest = ray_origin + view_vec * proj
            dist = (world_p - closest).length
            if dist < best_dist:
                best_dist = dist
                best_t = t
                best_local = local_p

        # pixel threshold roughly: convert small world dist
        # use view distance scale
        if best_local is None:
            return None
        # stricter: also check 2d pixel distance
        sc = view3d_utils.location_3d_to_region_2d(region, rv3d, obj.matrix_world @ best_local)
        if sc is None:
            return None
        pix = math.hypot(sc.x - event.mouse_region_x, sc.y - event.mouse_region_y)
        if pix > 10.0:
            return None
        return best_t, best_local


    def _transform_orient_matrix(self, context, obj):
        """3x3 matrix whose columns are X/Y/Z axes of the active transform orientation (world space)."""
        scene = context.scene
        slot = scene.transform_orientation_slots[0]
        otype = slot.type
        mw3 = obj.matrix_world.to_3x3().normalized()

        def basis_from_z(z):
            z = z.normalized() if z.length > 1e-12 else Vector((0, 0, 1))
            tmp = Vector((0, 0, 1)) if abs(z.z) < 0.9 else Vector((1, 0, 0))
            x = z.cross(tmp)
            if x.length < 1e-12:
                x = Vector((1, 0, 0))
            x.normalize()
            y = z.cross(x).normalized()
            m = Matrix.Identity(3)
            m.col[0] = x
            m.col[1] = y
            m.col[2] = z
            return m

        if otype == 'GLOBAL':
            return Matrix.Identity(3)
        if otype == 'LOCAL':
            return mw3.copy()
        if otype == 'VIEW':
            rv3d = context.region_data
            if rv3d is None:
                return Matrix.Identity(3)
            return rv3d.view_rotation.to_matrix()
        if otype == 'CURSOR':
            return scene.cursor.matrix.to_3x3().normalized()
        if otype in {'PARENT'}:
            if obj.parent is not None:
                return obj.parent.matrix_world.to_3x3().normalized()
            return mw3.copy()
        if otype == 'GIMBAL':
            # Approximate: object rotation euler axes
            return mw3.copy()
        if otype == 'NORMAL':
            # Average normal of tool verts (Edit Mode), else object Z
            try:
                _, bm = self.get_obj_bm(context)
                if bm is not None:
                    bm.verts.ensure_lookup_table()
                    idxs = list(getattr(self, 'vert_indices', None) or [])
                    if not idxs and getattr(self, 'spine_bind', None):
                        idxs = [item[0] for item in self.spine_bind[:32]]
                    acc = Vector((0, 0, 0))
                    n = 0
                    for i in idxs:
                        if 0 <= i < len(bm.verts):
                            # normal in world
                            acc += mw3 @ bm.verts[i].normal
                            n += 1
                    if n > 0 and acc.length > 1e-8:
                        return basis_from_z(acc)
            except Exception:
                pass
            return mw3.copy()
        # Custom orientation
        try:
            custom = slot.custom_orientation
            if custom is not None:
                return custom.matrix.copy()
        except Exception:
            pass
        return Matrix.Identity(3)

    def _apply_axis_constraint(self, context, obj, start_local, local_hit, axis):
        """Constrain local_hit delta to Blender transform orientation axes."""
        if not axis:
            return local_hit
        mw = obj.matrix_world
        mw3 = mw.to_3x3()
        try:
            imw3 = mw3.inverted()
        except Exception:
            imw3 = Matrix.Identity(3)
        orient = self._transform_orient_matrix(context, obj)
        # axes in world
        ax_x = orient.col[0].normalized()
        ax_y = orient.col[1].normalized()
        ax_z = orient.col[2].normalized()
        delta_world = mw3 @ (local_hit - start_local)
        if axis == 'X':
            delta_world = ax_x * delta_world.dot(ax_x)
        elif axis == 'Y':
            delta_world = ax_y * delta_world.dot(ax_y)
        elif axis == 'Z':
            delta_world = ax_z * delta_world.dot(ax_z)
        elif axis == 'YZ':
            delta_world = delta_world - ax_x * delta_world.dot(ax_x)
        elif axis == 'XZ':
            delta_world = delta_world - ax_y * delta_world.dot(ax_y)
        elif axis == 'XY':
            delta_world = delta_world - ax_z * delta_world.dot(ax_z)
        else:
            return local_hit
        return start_local + (imw3 @ delta_world)

    def start_drag(self, context, event, index, part='co'):
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return
        self.active_handle = index
        self.active_bez_part = part
        self.dragging = True
        self.constraint_axis = None
        ac = int(getattr(self, 'active_chain', 0) or 0)
        self.drag_start_mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self.drag_start_handle = self.bez[index][part].copy()
        self.drag_start_co = self.bez[index]['co'].copy()
        self.drag_start_hl = self.bez[index]['hl'].copy()
        self.drag_start_hr = self.bez[index]['hr'].copy()
        self.drag_start_sel = {}
        # Snapshot controller rest pose for Vertex-mode proportional editing.
        self._prop_ctrl_start = {}
        self._prop_ctrl_selected = set()
        # Snapshot handle tips separately: proportional handle editing needs its
        # own baseline and selected handle parts, rather than reusing controller
        # selection (which would make handle falloff uneven).
        self._prop_handle_start = {}
        self._prop_handle_selected = set()
        # Capture the CURRENT mesh state as the mirror baseline for this drag.
        # The mirror operation must preserve deformation that was created by a
        # previous Handle drag; otherwise a subsequent Controller drag would
        # rebuild the opposite side from all_rest and erase that deformation.
        if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX':
            try:
                axis = self._vertex_mirror_axis_resolved(obj)
                if axis != 'OFF':
                    # Cache immutable per-drag mirror metadata. Resolving axes and
                    # scanning selected rest coordinates on every mouse move is
                    # surprisingly expensive on dense meshes.
                    self._vertex_mirror_drag_axis = axis
                    self._vertex_mirror_drag_axis_items = tuple(
                        (ax, {'X': 0, 'Y': 1, 'Z': 2}[ax]) for ax in axis
                    )
                    selected_rest = [p for p in (self.rest_local or []) if p is not None]
                    source_sides = {}
                    side_eps = 1e-6
                    for ax, coord_i in self._vertex_mirror_drag_axis_items:
                        pos = 0
                        neg = 0
                        total = 0.0
                        for co in selected_rest:
                            value = float(co[coord_i])
                            total += value
                            if value > side_eps:
                                pos += 1
                            elif value < -side_eps:
                                neg += 1
                        if pos > neg:
                            source_sides[ax] = 1
                        elif neg > pos:
                            source_sides[ax] = -1
                        else:
                            source_sides[ax] = 1 if total >= 0.0 else -1
                    self._vertex_mirror_drag_source_sides = source_sides
                    bm = bmesh.from_edit_mesh(obj.data)
                    bm.verts.ensure_lookup_table()
                    pairs = self._vertex_mirror_build_pairs(obj)
                    self._vertex_mirror_drag_pairs = dict(pairs) if pairs else {}
                    self._vertex_mirror_drag_source_base = {}
                    self._vertex_mirror_drag_target_base = {}
                    for src, tgt in (pairs or {}).items():
                        if src < len(bm.verts) and tgt < len(bm.verts):
                            self._vertex_mirror_drag_source_base[int(src)] = bm.verts[src].co.copy()
                            self._vertex_mirror_drag_target_base[int(tgt)] = bm.verts[tgt].co.copy()
            except Exception:
                self._vertex_mirror_drag_pairs = None
                self._vertex_mirror_drag_source_base = {}
                self._vertex_mirror_drag_target_base = {}
        if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX' and getattr(self, 'bez', None):
            # Pivot for the proportional circle: median of selected controllers
            # (same idea as Blender's transform pivot), fallback to the dragged point.
            try:
                sel_cos = []
                for ci, bp in enumerate(self.bez):
                    if int(ci) == int(index) or self._sel_has(ci, 'co'):
                        sel_cos.append(bp['co'])
                if sel_cos:
                    acc = Vector((0.0, 0.0, 0.0))
                    for c in sel_cos:
                        acc += c
                    self._xform_center = acc / float(len(sel_cos))
                else:
                    self._xform_center = self.bez[index]['co'].copy()
            except Exception:
                self._xform_center = self.bez[index]['co'].copy()
            for ci, bp in enumerate(self.bez):
                self._prop_ctrl_start[ci] = {
                    'co': bp['co'].copy(),
                    'hl': bp['hl'].copy(),
                    'hr': bp['hr'].copy(),
                }
            self._prop_ctrl_selected = {
                int(k[0]) if isinstance(k, tuple) and len(k) == 2 else int(k[0])
                for k in (getattr(self, 'selected', set()) or set())
                if (isinstance(k, tuple) and len(k) >= 2 and (k[1] == 'co' or len(k) == 3 and k[2] == 'co'))
            }
            self._prop_ctrl_selected.add(int(index))
            # Handle drag: snapshot every handle tip and the currently selected
            # handle parts so proportional influence can be evaluated from the
            # stable curve baseline on every mouse move.
            for hi, bp in enumerate(self.bez):
                self._prop_handle_start[hi] = {
                    'hl': bp['hl'].copy(),
                    'hr': bp['hr'].copy(),
                    'co': bp['co'].copy(),
                }
            if part in ('hl', 'hr'):
                self._prop_handle_selected = {
                    (int(k[0]), k[1]) if isinstance(k, tuple) and len(k) >= 2 else (int(k), part)
                    for k in (getattr(self, 'selected', set()) or set())
                    if (isinstance(k, tuple) and len(k) >= 2 and k[1] in ('hl', 'hr'))
                }
                self._prop_handle_selected.add((int(index), part))
        is_spine = getattr(self, 'tool_mode', 'VERTEX') in ('SPINE_PLACE', 'SPINE_DEFORM')
        chains = getattr(self, 'spine_chains', None) or []
        if is_spine and hasattr(self, '_spine_norm_selected'):
            self.selected = self._spine_norm_selected()
            keys = set(self.selected) | {(ac, index, part)}
            for key in keys:
                if len(key) == 3:
                    ci, ti, tp = key
                else:
                    ci, ti, tp = ac, key[0], key[1]
                if chains and 0 <= ci < len(chains):
                    bez = chains[ci].get('bez') or []
                else:
                    bez = self.bez
                if 0 <= ti < len(bez):
                    bp = bez[ti]
                    self.drag_start_sel[(ci, ti, tp)] = {
                        'co': bp['co'].copy(),
                        'hl': bp['hl'].copy(),
                        'hr': bp['hr'].copy(),
                        'part_val': bp[tp].copy(),
                    }
        else:
            keys = set(getattr(self, 'selected', set()) or set()) | {(index, part)}
            for key in keys:
                if len(key) == 3:
                    ti, tp = key[1], key[2]
                else:
                    ti, tp = key[0], key[1]
                if 0 <= ti < len(self.bez):
                    bp = self.bez[ti]
                    self.drag_start_sel[(ti, tp)] = {
                        'co': bp['co'].copy(),
                        'hl': bp['hl'].copy(),
                        'hr': bp['hr'].copy(),
                        'part_val': bp[tp].copy(),
                    }
        # Automatic mirrored-chain coupling is only for Spine Deform/Edit Place.
        # It is detected from the actual chain geometry, so unrelated chains stay independent.
        if is_spine and getattr(self, 'spine_chains', None):
            self._spine_prepare_mirror_drag(context, ac)
        else:
            self._mirror_drag = None

        region = context.region
        rv3d = context.region_data
        wco = obj.matrix_world @ self.bez[index][part]
        self.drag_plane_point = wco.copy()
        self.drag_plane_normal = rv3d.view_rotation @ Vector((0, 0, 1))
        # Initial mouse hit on the same plane → relative grab (no jump to cursor)
        inv = obj.matrix_world.inverted()
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        normal = self.drag_plane_normal
        denom = view_vec.dot(normal)
        if abs(denom) > 1e-8:
            t = (self.drag_plane_point - ray_origin).dot(normal) / denom
            self.drag_start_mouse_local = inv @ (ray_origin + view_vec * t)
        else:
            self.drag_start_mouse_local = self.drag_start_handle.copy()

    def _apply_vertex_controller_proportional(self, context, delta):
        """Apply Blender-style proportional falloff to Blue Handles controllers.

        Vertex mode normally applies proportional editing only to mesh vertices.
        This keeps the custom Bezier controller rig in the same deformation field: 
        nearby controllers follow the dragged controller, and their handle offsets
        follow with them. Distances are measured in the controller rest pose.
        """
        if getattr(self, 'tool_mode', 'VERTEX') != 'VERTEX':
            return
        if not context.tool_settings.use_proportional_edit:
            return
        starts = getattr(self, '_prop_ctrl_start', None)
        if not starts:
            return
        selected = set(getattr(self, '_prop_ctrl_selected', set()) or set())
        if not selected:
            return
        radius = max(float(context.tool_settings.proportional_size), 1e-6)
        falloff = getattr(context.tool_settings, 'proportional_edit_falloff', 'SMOOTH')
        bez = getattr(self, 'bez', None) or []

        # Restore the whole controller set to the drag baseline first. This makes
        # every mouse move relative to the same pose instead of accumulating error.
        for i, st in starts.items():
            if 0 <= i < len(bez):
                bp = bez[i]
                bp['co'] = st['co'].copy()
                bp['hl'] = st['hl'].copy()
                bp['hr'] = st['hr'].copy()

        # Selected controllers receive the full drag delta.
        for i in selected:
            if 0 <= i < len(bez) and i in starts:
                bp = bez[i]
                bp['co'] = starts[i]['co'] + delta
                bp['hl'] = starts[i]['hl'] + delta
                bp['hr'] = starts[i]['hr'] + delta

        # Unselected controllers follow the nearest selected controller.
        for i, st in starts.items():
            if i in selected or not (0 <= i < len(bez)):
                continue
            best_d = 1e18
            for si in selected:
                sst = starts.get(si)
                if sst is None:
                    continue
                d = (st['co'] - sst['co']).length
                if d < best_d:
                    best_d = d
            if best_d > radius:
                continue
            w = prop_falloff_weight(best_d / radius, falloff)
            if w <= 1e-8:
                continue
            dlt = delta * w
            bp = bez[i]
            bp['co'] = st['co'] + dlt
            bp['hl'] = st['hl'] + dlt
            bp['hr'] = st['hr'] + dlt


    def _apply_vertex_handle_proportional(self, context, delta):
        """Vertex Mode handle proportional editing.

        Handle proportional editing affects the MESH deformation only.
        Neighboring Bezier handles must never be moved by proportional influence.
        The active handle is already updated by the normal drag path;
        apply_deform() uses the resulting Bezier curve to produce the smooth
        proportional vertex deformation.
        """
        return


    def update_drag(self, context, event):
        if not self.dragging or self.active_handle is None:
            return
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        # intersect ray with plane at handle (view-facing)
        normal = self.drag_plane_normal
        denom = view_vec.dot(normal)
        if abs(denom) < 1e-8:
            return
        t = (self.drag_plane_point - ray_origin).dot(normal) / denom
        world_hit = ray_origin + view_vec * t

        inv = obj.matrix_world.inverted()
        local_mouse = inv @ world_hit
        start = self.drag_start_handle
        # Relative grab: offset from where the mouse was when drag started
        mouse0 = getattr(self, 'drag_start_mouse_local', start)
        local_hit = start + (local_mouse - mouse0)

        axis = getattr(self, "constraint_axis", None)
        if axis:
            local_hit = self._apply_axis_constraint(context, obj, start, local_hit, axis)

        force = bool(getattr(self, "_ctrl_snap", False) or (event and event.ctrl))
        local_hit = self.snap_local(context, obj, local_hit, force=force)

        i = self.active_handle
        part = getattr(self, 'active_bez_part', 'co')
        if i is None:
            return

        delta = local_hit - start

        ac = int(getattr(self, 'active_chain', 0) or 0)
        is_spine = getattr(self, 'tool_mode', 'VERTEX') in ('SPINE_PLACE', 'SPINE_DEFORM')
        if is_spine and hasattr(self, '_spine_norm_selected'):
            self.selected = self._spine_norm_selected()
            targets = set(self.selected) if self.selected else {(ac, i, part)}
            if (ac, i, part) not in targets and (i, part) not in targets:
                targets = {(ac, i, part)}
        else:
            targets = set(self.selected) if self.selected else {(i, part)}
            if (i, part) not in targets:
                targets = {(i, part)}

        chains = getattr(self, 'spine_chains', None) or []
        if is_spine and chains:
            try:
                self._spine_store_active_chain()
            except Exception:
                pass

        for key in list(targets):
            if len(key) == 3:
                ci, ti, tpart = key
            else:
                ci, ti, tpart = ac, key[0], key[1]
            if chains and 0 <= ci < len(chains):
                bez = chains[ci].get('bez') or []
            else:
                bez = self.bez
            if ti < 0 or ti >= len(bez):
                continue
            bp = bez[ti]
            st = getattr(self, 'drag_start_sel', {}).get((ci, ti, tpart)) or getattr(self, 'drag_start_sel', {}).get((ti, tpart))
            if tpart == 'co':
                if ci == ac and ti == i and tpart == part:
                    new_co = local_hit.copy()
                    dlt = new_co - self.drag_start_co
                elif st:
                    dlt = delta
                    new_co = st['co'] + dlt
                else:
                    dlt = delta
                    new_co = bp['co'] + dlt
                bp['co'] = new_co
                # AUTO: tip handles locked to ends
                modes = (chains[ci].get('modes') if chains and 0 <= ci < len(chains) else None) or getattr(self, 'point_modes', [])
                mode = modes[ti] if ti < len(modes) else 'AUTO'
                if mode == 'AUTO':
                    if ti == 0:
                        bp['hl'] = bp['co'].copy()
                    if ti == len(bez) - 1:
                        bp['hr'] = bp['co'].copy()
                else:
                    if st:
                        bp['hl'] = st['hl'] + dlt
                        bp['hr'] = st['hr'] + dlt
                    else:
                        bp['hl'] = bp['hl'] + dlt
                        bp['hr'] = bp['hr'] + dlt
                    if ti == 0:
                        bp['hl'] = bp['co'].copy()
                    if ti == len(bez) - 1:
                        bp['hr'] = bp['co'].copy()
            else:
                # Handle tip drag — same as Vertex mode:
                # AUTO tip drag → ALIGNED (orange); Alt+drag → FREE (red)
                modes_list = None
                if chains and 0 <= ci < len(chains):
                    if not chains[ci].get('modes') or len(chains[ci]['modes']) != len(bez):
                        chains[ci]['modes'] = ['AUTO'] * len(bez)
                    modes_list = chains[ci]['modes']
                else:
                    if not hasattr(self, 'point_modes') or len(self.point_modes) != len(self.bez):
                        self.point_modes = ['AUTO'] * len(self.bez)
                    modes_list = self.point_modes
                cur_mode = modes_list[ti] if ti < len(modes_list) else 'AUTO'
                if event.alt:
                    modes_list[ti] = 'FREE'
                elif cur_mode == 'AUTO':
                    modes_list[ti] = 'ALIGNED'
                # keep active self.point_modes in sync for active chain
                if (not chains) or ci == ac:
                    if not hasattr(self, 'point_modes') or len(self.point_modes) != len(self.bez if not chains else bez):
                        self.point_modes = list(modes_list)
                    elif ti < len(self.point_modes):
                        self.point_modes[ti] = modes_list[ti]

                mode = modes_list[ti]
                # Absolute position for active tip; relative for others
                if ci == ac and ti == i and tpart == part:
                    bp[tpart] = local_hit.copy()
                elif st:
                    bp[tpart] = st['part_val'] + delta
                else:
                    bp[tpart] = bp[tpart] + delta

                # ALIGNED / was-AUTO: mirror opposite tip on interiors
                if mode in ('ALIGNED', 'AUTO') and ti not in (0, len(bez) - 1):
                    other = 'hl' if tpart == 'hr' else 'hr'
                    offset = bp[tpart] - bp['co']
                    bp[other] = bp['co'] - offset

                # Ends: unused side stays locked to co
                if ti == 0 and tpart != 'hl':
                    bp['hl'] = bp['co'].copy()
                if ti == len(bez) - 1 and tpart != 'hr':
                    bp['hr'] = bp['co'].copy()

                # Re-apply absolute tip after mirror
                if ci == ac and ti == i and tpart == part:
                    bp[tpart] = local_hit.copy()
                    if mode in ('ALIGNED', 'AUTO') and ti not in (0, len(bez) - 1):
                        other = 'hl' if tpart == 'hr' else 'hr'
                        offset = bp[tpart] - bp['co']
                        bp[other] = bp['co'] - offset
                continue
        # If the active chain has a geometrically mirrored partner, apply the reflected
        # translation to the matching controller/handle on that chain.  This follows
        # Blender's current Transform Pivot Point + Transform Orientation.
        if is_spine and getattr(self, '_mirror_drag', None):
            self._spine_sync_mirror_chain_live(context)

        # Vertex mode proportional editing also deforms the addon controllers,
        # so the Bezier rig and the mesh share the same falloff field.
        if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX':
            if part == 'co':
                self._apply_vertex_controller_proportional(context, delta)
            elif part in ('hl', 'hr'):
                self._apply_vertex_handle_proportional(context, delta)

        # Sync modes / rebuild AUTO handles on EVERY affected chain (not just active)
        def _unpack_tgt(key):
            if len(key) == 3:
                return key[0], key[1], key[2]
            return ac, key[0], key[1]

        dragging_any_tip = any(_unpack_tgt(k)[2] != 'co' for k in targets)
        dragging_any_co = any(_unpack_tgt(k)[2] == 'co' for k in targets)
        affected_chains = set()
        for key in targets:
            ci, ti, tpart = _unpack_tgt(key)
            affected_chains.add(ci)

        if chains:
            # Write active edits first
            try:
                ac2 = int(getattr(self, 'active_chain', 0) or 0)
                if 0 <= ac2 < len(chains) and getattr(self, 'point_modes', None):
                    chains[ac2]['modes'] = list(self.point_modes)
                self._spine_store_active_chain()
            except Exception:
                pass
            # Rebuild AUTO handles on each chain that was moved
            if dragging_any_co and not dragging_any_tip:
                old_bez, old_modes = self.bez, self.point_modes
                for ci in affected_chains:
                    if not (0 <= ci < len(chains)):
                        continue
                    ch = chains[ci]
                    bez = ch.get('bez') or []
                    if len(bez) < 2:
                        continue
                    modes = list(ch.get('modes') or ['AUTO'] * len(bez))
                    while len(modes) < len(bez):
                        modes.append('AUTO')
                    self.bez = bez
                    self.point_modes = modes
                    if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX':
                        # AUTO handles only depend on the moved controller and its
                        # immediate neighbors. Avoid scanning every controller on
                        # every mouse move.
                        moved_idxs = set()
                        for _key in targets:
                            _ci, _ti, _tp = _unpack_tgt(_key)
                            if _ci == ci and _tp == 'co':
                                moved_idxs.update((_ti - 1, _ti, _ti + 1))
                        self.rebuild_auto_handles(
                            only_indices=moved_idxs,
                            interior=True,
                        )
                    else:
                        # Preserve the existing full rebuild behavior for Spine.
                        self.rebuild_auto_handles(interior=True)
                    ch['bez'] = self.bez
                    ch['modes'] = self.point_modes
                self.bez, self.point_modes = old_bez, old_modes
            elif dragging_any_tip:
                # Tip edit: only rebuild AUTO ends on chains where tip was not the end itself
                old_bez, old_modes = self.bez, self.point_modes
                for ci in affected_chains:
                    if not (0 <= ci < len(chains)):
                        continue
                    ch = chains[ci]
                    bez = ch.get('bez') or []
                    n = len(bez)
                    if n < 2:
                        continue
                    modes = list(ch.get('modes') or ['AUTO'] * n)
                    while len(modes) < n:
                        modes.append('AUTO')
                    skip_ends = set()
                    for key in targets:
                        c2, ti, tpart = _unpack_tgt(key)
                        if c2 != ci or tpart == 'co':
                            continue
                        if ti == 0:
                            skip_ends.add(0)
                        if ti == n - 1:
                            skip_ends.add(n - 1)
                    end_idxs = [
                        ei for ei in (0, n - 1)
                        if ei < n and modes[ei] == 'AUTO' and ei not in skip_ends
                    ]
                    if end_idxs:
                        self.bez = bez
                        self.point_modes = modes
                        self.rebuild_auto_handles(only_indices=end_idxs, interior=False)
                        ch['bez'] = self.bez
                        ch['modes'] = self.point_modes
                self.bez, self.point_modes = old_bez, old_modes
            # AUTO handle rebuilding may have changed the source handles after the
            # first mirror pass. Recompute the mirrored partner's AUTO handles from
            # its current mirrored controller positions, then do one exact mirror pass.
            if is_spine and getattr(self, '_mirror_drag', None):
                info = self._mirror_drag
                pci = int(info.get('chain', -1))
                ac3 = int(info.get('source_chain', -1))
                chains3 = getattr(self, 'spine_chains', None) or []
                if 0 <= pci < len(chains3) and 0 <= ac3 < len(chains3):
                    try:
                        src_ch = chains3[ac3]
                        dst_ch = chains3[pci]
                        src_modes = list(src_ch.get('modes') or ['AUTO'] * len(src_ch.get('bez') or []))
                        dst_modes = list(dst_ch.get('modes') or ['AUTO'] * len(dst_ch.get('bez') or []))
                        rev3 = bool(info.get('reverse'))
                        n3 = len(src_ch.get('bez') or [])
                        if len(dst_modes) != len(dst_ch.get('bez') or []):
                            dst_modes = ['AUTO'] * len(dst_ch.get('bez') or [])
                        for si3 in range(min(n3, len(dst_modes))):
                            di3 = n3 - 1 - si3 if rev3 else si3
                            if 0 <= si3 < len(src_modes) and 0 <= di3 < len(dst_modes):
                                dst_modes[di3] = src_modes[si3]
                        dst_ch['modes'] = dst_modes
                        old_bez3, old_modes3 = self.bez, self.point_modes
                        self.bez = dst_ch.get('bez') or []
                        self.point_modes = dst_modes
                        if len(self.bez) >= 2:
                            self.rebuild_auto_handles(interior=True)
                        dst_ch['bez'] = self.bez
                        dst_ch['modes'] = self.point_modes
                        self.bez, self.point_modes = old_bez3, old_modes3
                    except Exception:
                        pass
                # Final exact reflection keeps FREE/ALIGNED handles and any AUTO
                # result perfectly paired after both chains have been updated.
                self._spine_sync_mirror_chain_live(context)
            # Keep self.bez as the active chain's list (same identity)
            try:
                ac2 = int(getattr(self, 'active_chain', 0) or 0)
                if 0 <= ac2 < len(chains) and chains[ac2].get('bez') is not None:
                    self.bez = chains[ac2]['bez']
                    if chains[ac2].get('modes') is not None:
                        self.point_modes = chains[ac2]['modes']
            except Exception:
                pass
        else:
            # Vertex mode / single bez
            n_bez = len(self.bez) if self.bez else 0
            skip_ends = set()
            for key in targets:
                _ci, ti, tpart = _unpack_tgt(key)
                if tpart != 'co':
                    if ti == 0:
                        skip_ends.add(0)
                    if ti == n_bez - 1:
                        skip_ends.add(n_bez - 1)
            if dragging_any_co and not dragging_any_tip:
                self.rebuild_auto_handles(interior=True)
            elif dragging_any_tip:
                end_idxs = [
                    ei for ei in (0, n_bez - 1)
                    if ei < n_bez and self.point_mode(ei) == 'AUTO' and ei not in skip_ends
                ]
                if end_idxs:
                    self.rebuild_auto_handles(only_indices=end_idxs, interior=False)

        # Vertex mode only — spine modal calls _spine_apply after update_drag
        if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX':
            self.apply_deform(context)
        context.area.tag_redraw()

    def snap_local(self, context, obj, local_co, force=False):
        """Apply Blender snap settings to a local-space point.
        force=True when Ctrl is held during drag (temporary snap)."""
        ts = context.tool_settings
        if not (ts.use_snap or force):
            return local_co

        world = obj.matrix_world @ local_co
        elements = ts.snap_elements

        # Grid / increment snap
        if 'INCREMENT' in elements:
            try:
                step = max(float(context.space_data.overlay.grid_scale), 1e-6)
            except Exception:
                step = 1.0
            if getattr(ts, "use_snap_grid_absolute", True):
                world = Vector((
                    round(world.x / step) * step,
                    round(world.y / step) * step,
                    round(world.z / step) * step,
                ))
            else:
                start_w = obj.matrix_world @ self.drag_start_handle if getattr(self, "drag_start_handle", None) else world
                delta = world - start_w
                delta = Vector((
                    round(delta.x / step) * step,
                    round(delta.y / step) * step,
                    round(delta.z / step) * step,
                ))
                world = start_w + delta

        # Vertex snap - nearest vertex on active / visible meshes (limited)
        if 'VERTEX' in elements:
            best = None
            best_d = 1e18
            thresh = 0.75
            targets = []
            if context.active_object and context.active_object.type == 'MESH':
                targets.append(context.active_object)
            for other in context.selected_objects:
                if other.type == 'MESH' and other not in targets:
                    targets.append(other)
            for other in targets:
                try:
                    mw = other.matrix_world
                    for v in other.data.vertices:
                        w = mw @ v.co
                        d = (w - world).length_squared
                        if d < best_d:
                            best_d = d
                            best = w
                except Exception:
                    continue
            if best is not None and best_d < thresh * thresh:
                world = best

        # Edge / Face: raycast from view toward world point (surface)
        # VOLUME is handled in _spine_mouse_local (multi-hit); here only FACE/EDGE
        if ('EDGE' in elements or 'FACE' in elements) and 'VOLUME' not in elements:
            try:
                region = context.region
                rv3d = context.region_data
                sc = view3d_utils.location_3d_to_region_2d(region, rv3d, world)
                if sc is not None:
                    view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, (sc.x, sc.y))
                    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, (sc.x, sc.y))
                    depsgraph = context.evaluated_depsgraph_get()
                    hit, loc, normal, face_index, hit_obj, matrix = context.scene.ray_cast(
                        depsgraph, origin, view_vec
                    )
                    if hit:
                        world = loc
            except Exception:
                pass

        return obj.matrix_world.inverted() @ world


    def point_mode(self, i):
        modes = getattr(self, 'point_modes', None)
        if not modes or i < 0 or i >= len(modes):
            return 'AUTO'
        return modes[i]

    def rebuild_auto_handles(self, only_indices=None, interior=True):
        """Recompute AUTO handles from control positions only.

        Continuous & deterministic: direction always follows chain order
        (index 0 → n-1). No dependence on previous-frame handles → no jumps.
        """
        n = len(self.bez)
        if n < 2:
            return

        # --- Interior ---
        # When only_indices is supplied, update only the affected AUTO points.
        # Full rebuild behavior remains unchanged when it is None.
        if interior:
            if only_indices is None:
                interior_indices = range(1, n - 1)
            else:
                interior_indices = sorted({
                    int(i) for i in only_indices
                    if 1 <= int(i) < n - 1
                })
            for i in interior_indices:
                if self.point_mode(i) != 'AUTO':
                    continue
                prev_co = self.bez[i - 1]['co']
                cur_co = self.bez[i]['co']
                next_co = self.bez[i + 1]['co']
                # Tangent from prev → next (stable chain direction)
                tdir = next_co - prev_co
                if tdir.length < 1e-12:
                    tdir = next_co - cur_co
                if tdir.length < 1e-12:
                    tdir = cur_co - prev_co
                if tdir.length < 1e-12:
                    continue
                tdir.normalize()
                L_l = max((cur_co - prev_co).length / 3.0, 1e-5)
                L_r = max((next_co - cur_co).length / 3.0, 1e-5)
                self.bez[i]['hl'] = cur_co - tdir * L_l
                self.bez[i]['hr'] = cur_co + tdir * L_r

        # --- Ends: look-at neighbor (handle tip if valid, else co) ---
        # Proper outward tangent along the chain so ends do not collapse inward.
        update_ends = only_indices is None
        if only_indices is not None:
            try:
                update_set = {int(i) for i in only_indices}
                update_ends = (0 in update_set or (n - 1) in update_set)
            except Exception:
                update_ends = True

        if update_ends and self.point_mode(0) == 'AUTO':
            p0 = self.bez[0]['co']
            nb = self.bez[1]
            chain = nb['co'] - p0
            if chain.length > 1e-12:
                chain_dir = chain.normalized()
                # Look at neighbor's inward handle (hl) if it faces along the chain
                tip = nb['hl']
                tip_vec = tip - p0
                if tip_vec.length > 1e-8 and tip_vec.dot(chain_dir) > 1e-6:
                    tdir = tip_vec.normalized()
                else:
                    tdir = chain_dir
                dist = chain.length
                # Slightly longer than 1/3 so end does not ease-in too hard
                L = max(dist * 0.4, 1e-5)
                L = min(L, dist * 0.85)
                self.bez[0]['hr'] = p0 + tdir * L
            self.bez[0]['hl'] = p0.copy()

        if update_ends and self.point_mode(n - 1) == 'AUTO':
            p0 = self.bez[n - 1]['co']
            nb = self.bez[n - 2]
            chain = nb['co'] - p0  # toward previous along chain
            if chain.length > 1e-12:
                chain_dir = chain.normalized()
                tip = nb['hr']
                tip_vec = tip - p0
                if tip_vec.length > 1e-8 and tip_vec.dot(chain_dir) > 1e-6:
                    tdir = tip_vec.normalized()
                else:
                    tdir = chain_dir
                dist = chain.length
                L = max(dist * 0.4, 1e-5)
                L = min(L, dist * 0.85)
                self.bez[n - 1]['hl'] = p0 + tdir * L
            self.bez[n - 1]['hr'] = p0.copy()

    def select_only(self, *args):
        """select_only(idx, part) or select_only(chain_idx, idx, part)"""
        tm = getattr(self, 'tool_mode', 'VERTEX')
        if len(args) == 2:
            idx, part = args
            # Vertex + Place: 2-tuple keys
            if tm in ('VERTEX', 'SPINE_PLACE'):
                self.selected = {(idx, part)}
                self.active_handle = idx
                self.active_bez_part = part
                return
            ci = int(getattr(self, 'active_chain', 0) or 0)
            key = (ci, idx, part)
        else:
            ci, idx, part = args
            if tm in ('VERTEX', 'SPINE_PLACE'):
                self.selected = {(idx, part)}
                self.active_handle = idx
                self.active_bez_part = part
                return
            key = (ci, idx, part)
        self.selected = {key}
        prev_ci = int(getattr(self, 'active_chain', 0) or 0)
        if tm == 'SPINE_DEFORM' and getattr(self, 'spine_chains', None):
            if ci != prev_ci:
                try:
                    self._spine_store_active_chain()
                except Exception:
                    pass
                self.active_chain = ci
                try:
                    self._spine_load_active_chain()
                except Exception:
                    pass
            else:
                self.active_chain = ci
        else:
            self.active_chain = ci
        self.active_handle = idx
        self.active_bez_part = part

    def select_toggle(self, *args):
        """Shift+click: Blender-like multi-select.

        - Not selected → add to selection and make active
        - Selected but not active → keep selected, make active (last clicked)
        - Selected and already active → deselect; new active from remaining
        """
        tm = getattr(self, 'tool_mode', 'VERTEX')
        if len(args) == 2:
            idx, part = args
            if tm in ('VERTEX', 'SPINE_PLACE'):
                key = (idx, part)
                is_active = (
                    self.active_handle == idx
                    and getattr(self, 'active_bez_part', 'co') == part
                )
                if key in self.selected and is_active:
                    self.selected.discard(key)
                    if self.selected:
                        it = next(iter(self.selected))
                        if len(it) == 2:
                            self.active_handle, self.active_bez_part = it[0], it[1]
                        else:
                            self.active_handle, self.active_bez_part = it[1], it[2]
                    else:
                        self.active_handle = None
                        self.active_bez_part = 'co'
                else:
                    # Add or keep; always make this the active (last Shift+click)
                    self.selected.add(key)
                    self.active_handle = idx
                    self.active_bez_part = part
                return
            ci = int(getattr(self, 'active_chain', 0) or 0)
            key = (ci, idx, part)
        else:
            ci, idx, part = args
            if tm in ('VERTEX', 'SPINE_PLACE'):
                key = (idx, part)
                is_active = (
                    self.active_handle == idx
                    and getattr(self, 'active_bez_part', 'co') == part
                )
                if key in self.selected and is_active:
                    self.selected.discard(key)
                    if self.selected:
                        it = next(iter(self.selected))
                        self.active_handle = it[0] if len(it) == 2 else it[1]
                        self.active_bez_part = it[1] if len(it) == 2 else it[2]
                    else:
                        self.active_handle = None
                        self.active_bez_part = 'co'
                else:
                    self.selected.add(key)
                    self.active_handle = idx
                    self.active_bez_part = part
                return
            key = (ci, idx, part)

        # SPINE_DEFORM: 3-tuple keys
        self.selected = self._spine_norm_selected()
        is_active = (
            self.active_handle == idx
            and getattr(self, 'active_bez_part', 'co') == part
            and int(getattr(self, 'active_chain', 0) or 0) == ci
        )
        if key in self.selected and is_active:
            self.selected.discard(key)
            if self.selected:
                nci, ni, np = next(iter(self.selected))
                if getattr(self, 'spine_chains', None) and nci != int(getattr(self, 'active_chain', 0) or 0):
                    try:
                        self._spine_store_active_chain()
                    except Exception:
                        pass
                    self.active_chain = nci
                    try:
                        self._spine_load_active_chain()
                    except Exception:
                        pass
                else:
                    self.active_chain = nci
                self.active_handle = ni
                self.active_bez_part = np
            else:
                self.active_handle = None
                self.active_bez_part = 'co'
        else:
            # Add (or already selected): always activate last Shift+clicked
            self.selected.add(key)
            prev_ci = int(getattr(self, 'active_chain', 0) or 0)
            if getattr(self, 'spine_chains', None) and ci != prev_ci:
                try:
                    self._spine_store_active_chain()
                except Exception:
                    pass
                self.active_chain = ci
                try:
                    self._spine_load_active_chain()
                except Exception:
                    pass
            else:
                self.active_chain = ci
            self.active_handle = idx
            self.active_bez_part = part

    def _spine_norm_selected(self):
        """Normalize selected keys to (chain_idx, idx, part)."""
        out = set()
        ac = int(getattr(self, 'active_chain', 0) or 0)
        for item in (getattr(self, 'selected', None) or set()):
            if len(item) == 3:
                out.add(item)
            elif len(item) == 2:
                out.add((ac, item[0], item[1]))
        return out

    def _sel_has(self, idx, part='co', chain_idx=None):
        """True if (idx, part) is selected (supports 2-tuple and 3-tuple keys)."""
        s = getattr(self, 'selected', None) or set()
        if not s:
            return False
        if (idx, part) in s:
            return True
        ci = int(getattr(self, 'active_chain', 0) or 0) if chain_idx is None else int(chain_idx)
        if (ci, idx, part) in s:
            return True
        # any chain
        for item in s:
            if len(item) == 3 and item[1] == idx and item[2] == part:
                if chain_idx is None or item[0] == ci:
                    return True
        return False


    def _finish_box_select(self, context, event):
        """Box select controllers (co) or, with Ctrl+Shift, handle tips only."""
        if not self.box_start or not self.box_end:
            return
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return
        region = context.region
        rv3d = context.region_data
        x0, y0 = self.box_start
        x1, y1 = self.box_end
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        # Tiny drag = click, already cleared selection
        if (xmax - xmin) < 3 and (ymax - ymin) < 3:
            return
        handles_only = self.box_handles_only
        add_mode = bool(event.shift)
        if not add_mode:
            self.selected = set()
        n = len(self.bez)
        mw = obj.matrix_world
        found = []
        for i, bp in enumerate(self.bez):
            parts = []
            if handles_only:
                if i == 0:
                    parts = ['hr']
                elif i == n - 1:
                    parts = ['hl']
                else:
                    parts = ['hl', 'hr']
            else:
                parts = ['co']
            for part in parts:
                if part != 'co' and (bp[part] - bp['co']).length < 1e-8:
                    continue
                sc = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ bp[part])
                if sc is None:
                    continue
                if xmin <= sc.x <= xmax and ymin <= sc.y <= ymax:
                    found.append((i, part))
        for key in found:
            self.selected.add(key)
        if found:
            self.active_handle, self.active_bez_part = found[-1]
        context.area.tag_redraw()

    def selected_point_indices(self):
        """Indices on the active chain only (for tilt/radius etc.)."""
        ac = int(getattr(self, 'active_chain', 0) or 0)
        idxs = set()
        for item in (getattr(self, 'selected', set()) or set()):
            if len(item) == 3:
                ci, i, p = item
                if ci == ac and p == 'co':
                    idxs.add(i)
            elif len(item) == 2:
                i, p = item
                if p == 'co':
                    idxs.add(i)
        if self.active_handle is not None:
            idxs.add(self.active_handle)
        return idxs

    def cycle_handle_mode(self, context):
        """Open handle type popup (Auto / Aligned / Free) like Blender Curve V menu."""
        idxs = self.selected_point_indices()
        if not idxs and self.active_handle is None:
            self.report({'INFO'}, "Select a controller first")
            return

        def draw_handle_popup(menu, _ctx):
            layout = menu.layout
            layout.label(text="Handle Type")
            for mid, name in (
                ('AUTO', 'Automatic'),
                ('ALIGNED', 'Aligned'),
                ('FREE', 'Free'),
            ):
                op = layout.operator("mesh.vdh_handle_type", text=name)
                op.mode = mid

        context.window_manager.popup_menu(draw_handle_popup, title="Handle Type")

    def restore_locked_selection(self, context):
        """Keep only the original verts selected; block other selection."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        lock = set(self._lock_selection)
        changed = False
        for v in bm.verts:
            want = v.index in lock
            if v.select != want:
                v.select = want
                changed = True
        if changed:
            bm.select_flush_mode()
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    def _sample_bezier_arc_length(self, pts, count):
        """Sample `count` points evenly by arc length along a Bezier point chain."""
        if count <= 1 or not pts:
            return [eval_bezier_points(pts, 0.0)] if pts else []
        # dense polyline approximation of the Bezier
        samples = max(128, len(pts) * 48)
        poly = []
        lengths = [0.0]
        for s in range(samples + 1):
            t = s / samples
            p = eval_bezier_points(pts, t)
            if poly:
                lengths.append(lengths[-1] + (p - poly[-1]).length)
            poly.append(p)
        total = lengths[-1]
        if total < 1e-12:
            return [poly[0].copy() for _ in range(count)]
        result = []
        for i in range(count):
            target = (i / (count - 1)) * total
            # binary search on cumulative length
            lo, hi = 0, len(lengths) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if lengths[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            j = max(1, lo)
            seg = lengths[j] - lengths[j - 1]
            u = 0.0 if seg < 1e-12 else (target - lengths[j - 1]) / seg
            result.append(poly[j - 1].lerp(poly[j], u))
        return result

    def _reparam_verts_on_curve(self):
        """Recompute self.params so each vert maps to nearest point on current rest_bez.
        Critical after insert/remove of handles so eval parameter stays consistent."""
        if not self.rest_local or not self.rest_bez:
            return
        new_params = []
        for co in self.rest_local:
            # sample the rest curve densely and find closest parameter
            best_t, best_d = 0.0, 1e18
            samples = max(64, len(self.rest_bez) * 24)
            for s in range(samples + 1):
                t = s / samples
                p = eval_bezier_points(self.rest_bez, t)
                d = (p - co).length_squared
                if d < best_d:
                    best_d = d
                    best_t = t
            new_params.append(best_t)
        # keep monotonic (small cleanup)
        for i in range(1, len(new_params)):
            if new_params[i] < new_params[i - 1]:
                new_params[i] = new_params[i - 1]
        if new_params:
            new_params[0] = 0.0
            new_params[-1] = 1.0
        self.params = new_params

    def _sync_rest_from_mesh(self, context, rebuild_handles=True):
        """After directly editing selected verts, update rest_local / params / curve."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        self.rest_local = [bm.verts[i].co.copy() for i in self.vert_indices]
        # re-parameterize by chord length first
        if len(self.rest_local) <= 1:
            self.params = [0.0] * len(self.rest_local)
        else:
            lengths = [0.0]
            for i in range(1, len(self.rest_local)):
                lengths.append(lengths[-1] + (self.rest_local[i] - self.rest_local[i - 1]).length)
            total = lengths[-1] if lengths[-1] > 1e-12 else 1.0
            self.params = [L / total for L in lengths]
        self._rebuild_prop_kdtree()
        if rebuild_handles:
            cos = auto_fit_handles(
                self.rest_local,
                max_count=self.handle_count,
                min_count=2,
            )
            if len(cos) < 2:
                cos = sample_polyline(self.rest_local, min(3, len(self.rest_local)))
            ok, center, radius, nrm = detect_circular_arc(self.rest_local)
            self._arc_center = center if ok else None
            self._arc_normal = nrm if ok else None
            self.bez = make_bezier_points(
                cos, poly=self.rest_local, center=self._arc_center, normal=self._arc_normal
            )
            self.rest_bez = copy_bezier_points(self.bez)
            self.handle_params = [nearest_param_on_polyline(self.rest_local, p['co']) for p in self.bez]
            if self.handle_params:
                self.handle_params[0] = 0.0
                self.handle_params[-1] = 1.0
            self.point_modes = ['AUTO'] * len(self.bez)
            self.rebuild_auto_handles()
            self.selected = set()
            self.active_handle = None
            # after rebuilding curve, reparam verts onto it for perfect rest match
            self._reparam_verts_on_curve()
        # also refresh all_rest for selected so proportional base stays consistent
        for i, v_i in enumerate(self.vert_indices):
            self.all_rest[v_i] = self.rest_local[i].copy()


    def _bake_prop_after_drag(self, context):
        """After a grab ends: commit mesh + curve so the next prop size
        starts from the current shape (no jump back to old rest).
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        if getattr(self, 'tool_mode', 'VERTEX') != 'VERTEX':
            return
        bm.verts.ensure_lookup_table()

        region = set(getattr(self, '_prop_last_affected', set()) or set())
        region |= set(self.vert_indices or [])
        # Include current prop radius neighborhood so edge verts are committed
        if self._all_kdtree is not None and self.rest_local:
            radius = max(
                float(getattr(context.tool_settings, 'proportional_size', self.prop_size)),
                1e-6,
            )
            sel_set = set(self.vert_indices)
            for rco in self.rest_local:
                for _co, tree_i, dist in self._all_kdtree.find_range(rco, radius):
                    vidx = self._all_kdtree_indices[tree_i]
                    if vidx not in sel_set:
                        region.add(vidx)

        for vidx in region:
            if vidx < len(bm.verts):
                self.all_rest[vidx] = bm.verts[vidx].co.copy()

        # Re-base selection + curve to current mesh/controllers
        self.rest_local = [
            bm.verts[i].co.copy()
            for i in self.vert_indices if i < len(bm.verts)
        ]
        if getattr(self, 'bez', None):
            self.rest_bez = copy_bezier_points(self.bez)
        if hasattr(self, '_reparam_verts_on_curve'):
            self._reparam_verts_on_curve()
        for i, v_i in enumerate(self.vert_indices):
            if i < len(self.rest_local):
                self.all_rest[v_i] = self.rest_local[i].copy()

        self._prop_last_affected = set()
        self.prop_size = float(context.tool_settings.proportional_size)
        self._rebuild_all_kdtree()
        self._rebuild_prop_kdtree()

    def _sync_prop_toggle(self, context, was_on):
        """Bake or re-baseline when proportional is toggled on/off."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        bm.verts.ensure_lookup_table()
        prop_now = context.tool_settings.use_proportional_edit
        if was_on and not prop_now:
            region = self._collect_prop_region(context, bm)
            region |= getattr(self, '_prop_last_affected', set())
            # prop is already off — collect used was_on path; use last_affected + kdtree manually
            if self._all_kdtree is not None:
                radius = max(self.prop_size, 1e-6)
                sel_set = set(self.vert_indices)
                for rco in self.rest_local:
                    for _co, tree_i, dist in self._all_kdtree.find_range(rco, radius):
                        vidx = self._all_kdtree_indices[tree_i]
                        if vidx not in sel_set:
                            region.add(vidx)
            for vidx in region:
                if vidx < len(bm.verts) and vidx not in self.vert_indices:
                    self.all_rest[vidx] = bm.verts[vidx].co.copy()
            self._prop_last_affected = set(region)
            self._rebuild_all_kdtree()
            self.apply_deform(context)
        elif not was_on and prop_now:
            self.rest_local = [
                bm.verts[i].co.copy()
                for i in self.vert_indices if i < len(bm.verts)
            ]
            self.rest_bez = copy_bezier_points(self.bez)
            self._reparam_verts_on_curve()
            for i, v_i in enumerate(self.vert_indices):
                if i < len(self.rest_local):
                    self.all_rest[v_i] = self.rest_local[i].copy()
            for v in bm.verts:
                if v.index not in self.vert_indices:
                    self.all_rest[v.index] = v.co.copy()
            self._prop_last_affected = set()
            self._rebuild_all_kdtree()
            self._rebuild_prop_kdtree()
            self.apply_deform(context)

    def _collect_prop_region(self, context, bm, controllers_only=False):
        """Vertex indices under proportional influence (excluding selection).

        For Vertex-mode Relax, ``controllers_only`` makes the proportional
        source the currently selected controller(s), instead of every point
        along the deformation chain. This keeps distant parts of the mesh out
        of Relax when only a local controller is selected.
        """
        region = set()
        prop_on = context.tool_settings.use_proportional_edit
        if not prop_on or self._all_kdtree is None:
            return region
        sel_set = set(self.vert_indices)
        radius = max(self.prop_size, 1e-6)

        if controllers_only:
            controller_indices = set()
            for key in (getattr(self, 'selected', set()) or set()):
                if isinstance(key, tuple) and len(key) >= 2:
                    try:
                        if key[1] == 'co':
                            controller_indices.add(int(key[0]))
                    except Exception:
                        pass
            if not controller_indices:
                active = getattr(self, 'active_handle', None)
                if active is not None:
                    try:
                        controller_indices.add(int(active))
                    except Exception:
                        pass

            rest_bez = getattr(self, 'rest_bez', None) or []
            for ci in controller_indices:
                if ci < 0 or ci >= len(rest_bez):
                    continue
                rco = rest_bez[ci]['co']
                for _co, tree_i, dist in self._all_kdtree.find_range(rco, radius):
                    vidx = self._all_kdtree_indices[tree_i]
                    if vidx not in sel_set:
                        region.add(vidx)
            return region

        for rco in self.rest_local:
            for _co, tree_i, dist in self._all_kdtree.find_range(rco, radius):
                vidx = self._all_kdtree_indices[tree_i]
                if vidx not in sel_set:
                    region.add(vidx)
        region |= getattr(self, '_prop_last_affected', set())
        return region

    def _bake_region_to_all_rest(self, bm, region):
        for vidx in region:
            if vidx < len(bm.verts):
                self.all_rest[vidx] = bm.verts[vidx].co.copy()

    def _prop_move_neighbors(self, context, bm, old_sel_cos, new_sel_cos):
        """Move proportional neighbors by the same deltas as selection (with falloff).
        Distance is measured in rest space (like Blender prop). Returns affected indices."""
        affected = set()
        if not context.tool_settings.use_proportional_edit:
            return affected
        if getattr(self, '_all_kdtree_dirty', False):
            self._rebuild_all_kdtree()
        if self._all_kdtree is None or not old_sel_cos:
            return affected
        radius = max(self.prop_size, 1e-6)
        sel_set = set(self.vert_indices)
        n = min(len(old_sel_cos), len(new_sel_cos), len(self.rest_local))
        deltas = [new_sel_cos[i] - old_sel_cos[i] for i in range(n)]
        # Shift+Scroll can fire many events in a row.  Do not rebuild/search the
        # dense-mesh KDTree on every wheel tick.  During one continuous align
        # session the proportional neighborhood is kept fixed; this is also
        # how the visual falloff feels stable while repeatedly nudging toward
        # the blue curve.  The cache is discarded as soon as another event
        # occurs or the proportional radius changes.
        cache = getattr(self, '_align_prop_cache', None)
        cache_key = (round(radius, 6), tuple(self.vert_indices))
        if cache is not None and cache.get('key') == cache_key:
            candidates = cache.get('candidates', {})
        else:
            candidates = {}  # vidx -> (dist, sel_i)
            for i in range(n):
                rco = self.rest_local[i]
                for _co, tree_i, dist in self._all_kdtree.find_range(rco, radius):
                    vidx = self._all_kdtree_indices[tree_i]
                    if vidx in sel_set:
                        continue
                    prev = candidates.get(vidx)
                    if prev is None or dist < prev[0]:
                        candidates[vidx] = (float(dist), i)
            self._align_prop_cache = {
                'key': cache_key,
                'candidates': candidates,
            }
        falloff = getattr(
            context.tool_settings, 'proportional_edit_falloff', 'SMOOTH'
        )
        for vidx, (dist, sel_i) in candidates.items():
            if vidx >= len(bm.verts):
                continue
            if self.all_rest.get(vidx) is None:
                continue
            if self._pending_undo_prop_before is not None:
                self._pending_undo_prop_before.setdefault(vidx, self.all_rest[vidx].copy())
            w = prop_falloff_weight(dist / radius, falloff)
            bm.verts[vidx].co = bm.verts[vidx].co + deltas[sel_i] * w
            self.all_rest[vidx] = bm.verts[vidx].co.copy()
            affected.add(vidx)
        if affected:
            self._prop_last_affected = set(affected) | getattr(self, '_prop_last_affected', set())
            # Do not rebuild the dense all-vertex KDTree during Shift+Scroll.
            # It is refreshed when the align session ends.
            self._align_prop_kdtree_dirty = True
        return affected

    def space_selection(self, context):
        """Evenly redistribute selected verts along the current blue curve by arc length (Space).
        With Proportional ON, neighbors move with the same deltas (falloff)."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None or len(self.vert_indices) < 2:
            return
        _mirror_state = self._vertex_mirror_operation_begin(context)
        self.push_undo(context)
        bm.verts.ensure_lookup_table()
        n = len(self.vert_indices)

        old_sel = [bm.verts[i].co.copy() for i in self.vert_indices]

        # Sample current blue curve at equal arc-length → verts land ON the curve
        spaced = self._sample_bezier_arc_length(self.bez, n)
        for i, v_i in enumerate(self.vert_indices):
            if i < len(spaced):
                bm.verts[v_i].co = spaced[i]
        new_sel = [bm.verts[i].co.copy() for i in self.vert_indices]

        # Proportional neighbors follow selection deltas
        self._prop_move_neighbors(context, bm, old_sel, new_sel)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

        # Update rest; reparam so offset≈0 — prevents jump on next handle drag
        self.rest_local = [c.copy() for c in new_sel]
        self.rest_bez = copy_bezier_points(self.bez)
        self.handle_params = [nearest_param_on_polyline(self.rest_local, p['co']) for p in self.bez]
        if self.handle_params:
            self.handle_params[0] = 0.0
            self.handle_params[-1] = 1.0
        self._reparam_verts_on_curve()
        self._rebuild_prop_kdtree()
        for i, v_i in enumerate(self.vert_indices):
            self.all_rest[v_i] = self.rest_local[i].copy()

        self.apply_deform(context)
        self._vertex_mirror_operation_end(context, _mirror_state)
        self.report({'INFO'}, "Selection spaced & aligned to curve")
        context.area.tag_redraw()

    def _vdh_refresh_edit_normals(self, context):
        """Refresh mesh normals after vertex editing operations."""
        try:
            obj = context.object
            if obj and obj.type == 'MESH':
                import bmesh
                bm = bmesh.from_edit_mesh(obj.data)
                bm.normal_update()
                bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
                obj.data.update()
                if context.area:
                    context.area.tag_redraw()
        except Exception:
            pass

    def relax_selection(self, context, iterations=8):
        """Relax / smooth selected verts along the chain (R).
        If Proportional is ON, also smooth the influenced region (like Sculpt Smooth)
        and bake so those verts do not snap back."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None or len(self.vert_indices) < 3:
            return
        _mirror_state = self._vertex_mirror_operation_begin(context)
        self.push_undo(context)
        idxs = self.vert_indices
        bm.verts.ensure_lookup_table()
        mode = getattr(self, 'smooth_mode', None) or VDH_SMOOTH_MODE

        # --- Proportional region ---
        # When Proportional Editing is ON, Relax must use only the
        # controller(s) currently selected as the proportional source.
        # The non-proportional Relax path is intentionally unchanged.
        region = self._collect_prop_region(context, bm, controllers_only=True)
        self._bake_region_to_all_rest(bm, region)

        # --- Smooth the selection chain (along ordered indices) ---
        for _ in range(iterations):
            new_cos = [bm.verts[i].co.copy() for i in idxs]
            for j in range(1, len(idxs) - 1):
                a = bm.verts[idxs[j - 1]].co
                b = bm.verts[idxs[j + 1]].co
                new_cos[j] = bm.verts[idxs[j]].co.lerp((a + b) * 0.5, 0.35)
            for j, v_i in enumerate(idxs):
                bm.verts[v_i].co = new_cos[j]

        # --- Smooth proportional region (algorithm from Shift+R popup) ---
        if region:
            steps = max(3, iterations // 2)
            factor = 0.4
            for step in range(steps):
                new_pos = {}
                for vidx in region:
                    if vidx >= len(bm.verts):
                        continue
                    v = bm.verts[vidx]
                    linked = [e.other_vert(v) for e in v.link_edges]
                    if not linked:
                        continue
                    avg = Vector((0, 0, 0))
                    for ov in linked:
                        avg += ov.co
                    avg /= len(linked)
                    delta = avg - v.co

                    if mode == 'TANGENTIAL':
                        # Project delta onto tangent plane (keep normal component)
                        nrm = v.normal
                        if nrm.length > 1e-8:
                            nrm = nrm.normalized()
                            delta = delta - nrm * delta.dot(nrm)
                        new_pos[vidx] = v.co + delta * factor
                    elif mode == 'TAUBIN':
                        # Positive then negative laplacian (less shrinkage)
                        if step % 2 == 0:
                            new_pos[vidx] = v.co + delta * factor
                        else:
                            new_pos[vidx] = v.co - delta * (factor * 0.55)
                    else:
                        # LAPLACIAN
                        new_pos[vidx] = v.co.lerp(avg, factor)

                for vidx, co in new_pos.items():
                    bm.verts[vidx].co = co

            self._bake_region_to_all_rest(bm, region)
            self._prop_last_affected = set(region)
            self._rebuild_all_kdtree()

        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

        # Update rest from smoothed selection chain
        self.rest_local = [bm.verts[i].co.copy() for i in self.vert_indices]
        if len(self.rest_local) <= 1:
            self.params = [0.0] * len(self.rest_local)
        else:
            lengths = [0.0]
            for i in range(1, len(self.rest_local)):
                lengths.append(lengths[-1] + (self.rest_local[i] - self.rest_local[i - 1]).length)
            total = lengths[-1] if lengths[-1] > 1e-12 else 1.0
            self.params = [L / total for L in lengths]

        # Keep the same number of controllers — handles carry the curvature
        n_handles = max(2, len(self.bez))
        cos = sample_polyline(self.rest_local, n_handles)
        if len(cos) >= 2:
            cos[0] = self.rest_local[0].copy()
            cos[-1] = self.rest_local[-1].copy()
        cleaned = [cos[0]]
        for p in cos[1:]:
            if (p - cleaned[-1]).length > 1e-6:
                cleaned.append(p)
        cos = cleaned if len(cleaned) >= 2 else cos

        ok, center, radius, nrm = detect_circular_arc(self.rest_local)
        self._arc_center = center if ok else None
        self._arc_normal = nrm if ok else None

        # Preserve modes, handle selection, and handle orientation so repeated
        # R presses keep the same controller source for proportional Relax.
        old_modes = list(getattr(self, 'point_modes', ['AUTO'] * n_handles))
        old_selected = set(getattr(self, 'selected', set()) or set())
        old_active_handle = getattr(self, 'active_handle', None)
        old_active_bez_part = getattr(self, 'active_bez_part', 'co')
        old_bez = copy_bezier_points(self.bez)
        self.bez = make_bezier_points(
            cos, poly=self.rest_local, center=self._arc_center, normal=self._arc_normal
        )
        # Stabilize handle directions against previous frame
        for i in range(len(self.bez)):
            if i >= len(old_bez):
                break
            for part in ('hl', 'hr'):
                old_off = old_bez[i][part] - old_bez[i]['co']
                new_off = self.bez[i][part] - self.bez[i]['co']
                if old_off.length > 1e-8 and new_off.length > 1e-8:
                    if old_off.dot(new_off) < 0:
                        # flipped → keep new length but restore previous side
                        self.bez[i][part] = self.bez[i]['co'] - new_off
        self.rest_bez = copy_bezier_points(self.bez)
        self.handle_params = [nearest_param_on_polyline(self.rest_local, p['co']) for p in self.bez]
        if self.handle_params:
            self.handle_params[0] = 0.0
            self.handle_params[-1] = 1.0
        self.point_modes = []
        for i in range(len(self.bez)):
            if i < len(old_modes):
                self.point_modes.append(old_modes[i])
            else:
                self.point_modes.append('AUTO')
        # Keep the user's controller selection/active controller intact so a
        # second R uses the same proportional source instead of losing the region.
        self.selected = old_selected
        self.active_handle = old_active_handle
        self.active_bez_part = old_active_bez_part
        self._reparam_verts_on_curve()
        self._rebuild_prop_kdtree()
        for i, v_i in enumerate(self.vert_indices):
            self.all_rest[v_i] = self.rest_local[i].copy()

        self.apply_deform(context)
        self._vertex_mirror_operation_end(context, _mirror_state)
        mode = getattr(self, 'smooth_mode', None) or VDH_SMOOTH_MODE
        if region:
            msg = f"Smoothed ({mode.title()}) + proportional region"
        else:
            msg = f"Selection relaxed ({mode.title()})"
        self.report({'INFO'}, msg)
        context.area.tag_redraw()

    def straighten_selection(self, context):
        """Make selected verts a straight line between ends (L). Blue curve becomes perfectly straight."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None or len(self.vert_indices) < 2:
            return
        _mirror_state = self._vertex_mirror_operation_begin(context)
        self.push_undo(context)
        idxs = self.vert_indices
        p0 = bm.verts[idxs[0]].co.copy()
        p1 = bm.verts[idxs[-1]].co.copy()
        n = len(idxs)
        for i, v_i in enumerate(idxs):
            t = 0.0 if n == 1 else i / (n - 1)
            bm.verts[v_i].co = p0.lerp(p1, t)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        self._sync_rest_from_mesh(context, rebuild_handles=True)

        # Force control points + handles perfectly onto the straight line (zero curvature)
        line_dir = p1 - p0
        line_len = line_dir.length
        nb = len(self.bez)
        if nb >= 2 and line_len > 1e-12:
            line_dir = line_dir / line_len
            # project every control point onto the line segment
            for i in range(nb):
                t = i / (nb - 1)
                self.bez[i]['co'] = p0 + line_dir * (t * line_len)
            for i in range(nb):
                bp = self.bez[i]
                if i == 0:
                    nxt = self.bez[1]['co']
                    bp['hr'] = bp['co'] + (nxt - bp['co']) / 3.0
                    bp['hl'] = bp['co'].copy()
                elif i == nb - 1:
                    prv = self.bez[nb - 2]['co']
                    bp['hl'] = bp['co'] + (prv - bp['co']) / 3.0
                    bp['hr'] = bp['co'].copy()
                else:
                    prv = self.bez[i - 1]['co']
                    nxt = self.bez[i + 1]['co']
                    bp['hl'] = bp['co'] + (prv - bp['co']) / 3.0
                    bp['hr'] = bp['co'] + (nxt - bp['co']) / 3.0
            self.rest_bez = copy_bezier_points(self.bez)
            self.point_modes = ['AUTO'] * nb
            self.handle_params = [i / (nb - 1) for i in range(nb)]
            self._reparam_verts_on_curve()

        self.apply_deform(context)
        self._vertex_mirror_operation_end(context, _mirror_state)
        self.report({'INFO'}, "Selection straightened")
        context.area.tag_redraw()

    def set_flow_selection(self, context, tension=1.8, iterations=8, min_angle_deg=0.0):
        """Set Flow (F) — EdgeFlow-style: Hermite through transverse neighbour loops.
        For each edge of the selection chain, sample C1..C4 across the ring and
        place the vertex on the interpolated surface flow (orthogonal to the loop).
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None or len(self.vert_indices) < 2:
            return
        _mirror_state = self._vertex_mirror_operation_begin(context)
        self.push_undo(context)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        idxs = self.vert_indices
        sel_set = set(idxs)
        min_angle = math.radians(min_angle_deg)

        # Map consecutive selected verts -> mesh edge
        def edge_between(a_idx, b_idx):
            va = bm.verts[a_idx]
            for e in va.link_edges:
                if e.other_vert(va).index == b_idx:
                    return e
            return None

        old_sel = [bm.verts[i].co.copy() for i in idxs]

        for _it in range(max(1, iterations)):
            # Collect targets so simultaneous update is stable
            targets = {}  # vert_index -> list of candidate positions (avg if multiple)

            for ei in range(len(idxs) - 1):
                e = edge_between(idxs[ei], idxs[ei + 1])
                if e is None or e.is_boundary:
                    continue

                for loop in e.link_loops:
                    # Walk ring like EdgeFlow (Benjamin Sauder)
                    ring1 = loop.link_loop_next.link_loop_next
                    ring2 = loop.link_loop_radial_prev.link_loop_prev.link_loop_prev

                    center = e.other_vert(loop.vert)

                    p2 = ring1.vert
                    p3 = ring2.link_loop_radial_next.vert

                    # Outer control p1
                    if not ring1.edge.is_boundary:
                        final = ring1.link_loop_radial_next.link_loop_next
                        a, b = final.edge.verts
                        p1 = b.co.copy() if p2 == a else a.co.copy()
                        aa = (p1 - p2.co).normalized()
                        bb = (center.co - p2.co).normalized()
                        dot = min(1.0, max(-1.0, aa.dot(bb)))
                        angle = math.acos(dot)
                        if angle < min_angle:
                            p1 = p2.co - (p3.co - p2.co) * 0.5
                    else:
                        p1 = p2.co - (p3.co - p2.co)

                    # Outer control p4
                    p3c = p3.co.copy()
                    if not ring2.edge.is_boundary:
                        final = ring2.link_loop_radial_prev.link_loop_prev
                        a, b = final.edge.verts
                        p4 = b.co.copy() if p3 == a else a.co.copy()
                        aa = (p4 - p3c).normalized()
                        bb = (center.co - p3c).normalized()
                        dot = min(1.0, max(-1.0, aa.dot(bb)))
                        angle = math.acos(dot)
                        if angle < min_angle:
                            p4 = p3c - (p2.co - p3c) * 0.5
                    else:
                        # radial_next doesn't work at boundary (EdgeFlow)
                        p3_b = ring2.edge.other_vert(p3)
                        p3c = p3_b.co.copy()
                        p4 = p3c - (p2.co - p3c)

                    p2c = p2.co.copy()

                    if (p1 - p2c).length < 1e-12 or (p4 - p3c).length < 1e-12:
                        continue
                    if (p2c - p3c).length < 1e-12:
                        continue

                    # Normalize point distances so long edges don't skew the curve
                    d = (p2c - p3c).length * 0.5
                    p1n = p2c + (d * (p1 - p2c).normalized())
                    p4n = p3c + (d * (p4 - p3c).normalized())

                    # Hermite at mu=0.5 (same as EdgeFlow)
                    # tension: EdgeFlow uses -tension where tension prop is ~1.8 default
                    def hermite_1d(y0, y1, y2, y3, mu, tens, bias):
                        mu2 = mu * mu
                        mu3 = mu2 * mu
                        m0 = (y1 - y0) * (1 + bias) * (1 - tens) / 2
                        m0 += (y2 - y1) * (1 - bias) * (1 - tens) / 2
                        m1 = (y2 - y1) * (1 + bias) * (1 - tens) / 2
                        m1 += (y3 - y2) * (1 - bias) * (1 - tens) / 2
                        a0 = 2 * mu3 - 3 * mu2 + 1
                        a1 = mu3 - 2 * mu2 + mu
                        a2 = mu3 - mu2
                        a3 = -2 * mu3 + 3 * mu2
                        return a0 * y1 + a1 * m0 + a2 * m1 + a3 * y2

                    tens = -float(tension)
                    res = Vector((
                        hermite_1d(p1n.x, p2c.x, p3c.x, p4n.x, 0.5, tens, 0.0),
                        hermite_1d(p1n.y, p2c.y, p3c.y, p4n.y, 0.5, tens, 0.0),
                        hermite_1d(p1n.z, p2c.z, p3c.z, p4n.z, 0.5, tens, 0.0),
                    ))
                    targets.setdefault(center.index, []).append(res)

            for vidx, pts in targets.items():
                if vidx not in sel_set or vidx >= len(bm.verts):
                    continue
                acc = Vector((0, 0, 0))
                for p in pts:
                    acc += p
                bm.verts[vidx].co = acc / len(pts)

        new_sel = [bm.verts[i].co.copy() for i in idxs]
        self._prop_move_neighbors(context, bm, old_sel, new_sel)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

        # Sync rest + blue curve
        self.rest_local = [p.copy() for p in new_sel]
        if len(self.rest_local) <= 1:
            self.params = [0.0] * len(self.rest_local)
        else:
            lengths = [0.0]
            for i in range(1, len(self.rest_local)):
                lengths.append(lengths[-1] + (self.rest_local[i] - self.rest_local[i - 1]).length)
            total = lengths[-1] if lengths[-1] > 1e-12 else 1.0
            self.params = [L / total for L in lengths]

        n_handles = max(2, len(self.bez))
        cos = sample_polyline(self.rest_local, n_handles)
        if len(cos) >= 2:
            cos[0] = self.rest_local[0].copy()
            cos[-1] = self.rest_local[-1].copy()
        cleaned = [cos[0]]
        for p in cos[1:]:
            if (p - cleaned[-1]).length > 1e-6:
                cleaned.append(p)
        cos = cleaned if len(cleaned) >= 2 else cos

        ok, center, radius, nrm = detect_circular_arc(self.rest_local)
        self._arc_center = center if ok else None
        self._arc_normal = nrm if ok else None

        old_modes = list(getattr(self, 'point_modes', ['AUTO'] * n_handles))
        old_bez = copy_bezier_points(self.bez)
        self.bez = make_bezier_points(
            cos, poly=self.rest_local, center=self._arc_center, normal=self._arc_normal
        )
        for i in range(len(self.bez)):
            if i >= len(old_bez):
                break
            for part in ('hl', 'hr'):
                old_off = old_bez[i][part] - old_bez[i]['co']
                new_off = self.bez[i][part] - self.bez[i]['co']
                if old_off.length > 1e-8 and new_off.length > 1e-8:
                    if old_off.dot(new_off) < 0:
                        self.bez[i][part] = self.bez[i]['co'] - new_off
        self.rest_bez = copy_bezier_points(self.bez)
        self.handle_params = [nearest_param_on_polyline(self.rest_local, p['co']) for p in self.bez]
        if self.handle_params:
            self.handle_params[0] = 0.0
            self.handle_params[-1] = 1.0
        self.point_modes = []
        for i in range(len(self.bez)):
            if i < len(old_modes):
                self.point_modes.append(old_modes[i])
            else:
                self.point_modes.append('AUTO')
        self.selected = set()
        self.active_handle = None
        self._reparam_verts_on_curve()
        self._rebuild_prop_kdtree()
        for i, v_i in enumerate(self.vert_indices):
            self.all_rest[v_i] = self.rest_local[i].copy()

        self.apply_deform(context)
        self._vertex_mirror_operation_end(context, _mirror_state)
        self.report({'INFO'}, "Set Flow applied")
        context.area.tag_redraw()

    def align_selection_to_curve(self, context, amount=0.15):
        """Pull selected verts toward the blue curve (Shift+Scroll).
        With Proportional ON, neighbors follow with falloff."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None or len(self.vert_indices) < 1:
            return
        _mirror_state = self._vertex_mirror_operation_begin(context)
        bm.verts.ensure_lookup_table()
        # Each Shift+Scroll tick is a real edit operation, so make it an
        # undoable step before changing the mesh/handles. This is especially
        # important when proportional editing also moves non-selected verts.
        try:
            self._pending_undo_prop_before = {} if context.tool_settings.use_proportional_edit else None
            self.push_undo(context, fast_undo=True)
        except Exception:
            self._pending_undo_prop_before = None
        amt = max(-1.0, min(1.0, amount))
        old_sel = [bm.verts[i].co.copy() for i in self.vert_indices]
        for i, v_i in enumerate(self.vert_indices):
            if v_i >= len(bm.verts):
                continue
            u = self.params[i] if i < len(self.params) else 0.0
            on_curve = eval_bezier_points(self.bez, u)
            bm.verts[v_i].co = old_sel[i].lerp(on_curve, amt)
        new_sel = [bm.verts[i].co.copy() for i in self.vert_indices]
        self._prop_move_neighbors(context, bm, old_sel, new_sel)
        if self._pending_undo_prop_before is not None and self.undo_stack:
            snap = self.undo_stack[-1]
            if isinstance(snap, tuple) and len(snap) >= 11 and isinstance(snap[8], dict):
                snap[8].update(self._pending_undo_prop_before)
        self._pending_undo_prop_before = None
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        self.rest_local = [c.copy() for c in new_sel]
        self.rest_bez = copy_bezier_points(self.bez)
        self._reparam_verts_on_curve()
        self._rebuild_prop_kdtree()
        for i, v_i in enumerate(self.vert_indices):
            self.all_rest[v_i] = self.rest_local[i].copy()
        self._vertex_mirror_operation_end(context, _mirror_state)
        self.report({'INFO'}, "Selection aligned to curve" if amount > 0 else "Selection pushed from curve")
        context.area.tag_redraw()

    def snapshot_handles(self, context=None, fast_undo=False):
        # Snapshot the current proportional BASE for the whole neighborhood, not
        # only the last frame's affected set.  A controller drag with proportional
        # editing can use a different neighborhood on the next drag; restoring only
        # _prop_last_affected would fall back to initial_all_rest for the newly
        # affected vertices and make Undo jump them all the way back to the tool's
        # initial state instead of one history step.
        prop_snap = {}
        aff = set(getattr(self, '_prop_last_affected', set()) or set())
        sel = set(self.vert_indices)
        # Controller clicks call push_undo() without passing Blender context.
        # Do not let that make the proportional snapshot empty: after a previous
        # drag _prop_last_affected is intentionally cleared by _bake_prop_after_drag.
        # Use the tool's cached proportional state/size when context is unavailable.
        try:
            if context is not None:
                prop_enabled = bool(getattr(context.tool_settings, 'use_proportional_edit', False))
                radius = max(float(getattr(context.tool_settings, 'proportional_size', self.prop_size)), 1e-6)
            else:
                prop_enabled = bool(getattr(self, '_prop_was_on', False))
                radius = max(float(getattr(self, 'prop_size', 1.0)), 1e-6)
            if prop_enabled and self._all_kdtree is not None:
                for rco in (self.rest_local or []):
                    for _co, tree_i, _dist in self._all_kdtree.find_range(rco, radius):
                        vidx = self._all_kdtree_indices[tree_i]
                        if vidx not in sel:
                            aff.add(vidx)
        except Exception:
            pass
        for vidx in aff:
            if vidx in sel:
                continue
            co = self.all_rest.get(vidx)
            if co is not None:
                prop_snap[vidx] = co.copy()
        return (
            copy_bezier_points(self.bez),
            copy_bezier_points(self.rest_bez),
            list(self.handle_params),
            list(getattr(self, 'point_modes', ['AUTO'] * len(self.bez))),
            self.active_handle,
            getattr(self, 'active_bez_part', 'co'),
            [p.copy() for p in self.rest_local],
            list(self.params),
            prop_snap,
            aff,
            bool(fast_undo),
        )

    def restore_handles(self, snap):
        (bez, rest_bez, handle_params, point_modes, active, part,
         rest_local, params, prop_snap, prop_aff, fast_undo) = snap
        self.bez = copy_bezier_points(bez)
        self.rest_bez = copy_bezier_points(rest_bez)
        self.handle_params = list(handle_params)
        self.point_modes = list(point_modes)
        self.active_handle = active
        self.active_bez_part = part
        self.rest_local = [p.copy() for p in rest_local]
        self.params = list(params)
        self._rebuild_prop_kdtree()
        # selected rest
        for i, v_i in enumerate(self.vert_indices):
            if i < len(self.rest_local):
                self.all_rest[v_i] = self.rest_local[i].copy()
        # proportional: reset non-selected to initial, then apply snap
        sel = set(self.vert_indices)
        if getattr(self, 'initial_all_rest', None):
            for idx, co in self.initial_all_rest.items():
                if idx not in sel:
                    self.all_rest[idx] = co.copy()
        for vidx, co in prop_snap.items():
            self.all_rest[vidx] = co.copy()
        self._prop_last_affected = set(prop_aff)
        self._history_prop_snap = dict(prop_snap)
        self._history_fast_undo = bool(fast_undo)
        self._all_kdtree_dirty = True
        # Undo/redo restores the proportional base, so the Shift+Scroll
        # neighborhood cache must not survive across a history jump.
        self._align_prop_cache = None
        self._align_prop_session = False
        self._align_prop_kdtree_dirty = False

    def push_undo(self, context=None, fast_undo=False):
        self.undo_stack.append(self.snapshot_handles(context=context, fast_undo=fast_undo))
        if len(self.undo_stack) > 64:
            self.undo_stack = self.undo_stack[-64:]
        self.redo_stack.clear()

    def _restore_history_mesh(self, context):
        obj, bm = self.get_obj_bm(context)
        if not obj or not bm:
            return
        bm.verts.ensure_lookup_table()
        sel_set = set(self.vert_indices)
        if context.tool_settings.use_proportional_edit and getattr(self, '_history_fast_undo', False):
            for i, vidx in enumerate(self.vert_indices):
                if i < len(self.rest_local) and vidx < len(bm.verts):
                    bm.verts[vidx].co = self.rest_local[i]
            for vidx in getattr(self, '_history_prop_snap', {}).keys():
                if vidx not in sel_set and vidx in self.all_rest and vidx < len(bm.verts):
                    bm.verts[vidx].co = self.all_rest[vidx]
        else:
            for vidx, co in self.all_rest.items():
                if vidx not in sel_set and vidx < len(bm.verts):
                    bm.verts[vidx].co = co
            self.apply_deform(context)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    def do_undo(self, context):
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.snapshot_handles(context=context))
        self.restore_handles(self.undo_stack.pop())
        self._restore_history_mesh(context)
        context.area.tag_redraw()
        return True

    def do_redo(self, context):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.snapshot_handles(context=context))
        self.restore_handles(self.redo_stack.pop())
        self._restore_history_mesh(context)
        context.area.tag_redraw()
        return True



    def _spine_event_in_ui(self, context, event):
        """True if mouse is over a non-viewport UI region (snap, prop, headers...)."""
        try:
            mx, my = event.mouse_x, event.mouse_y
            for area in context.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for r in area.regions:
                    if r.type == 'WINDOW':
                        continue
                    if r.x <= mx < r.x + r.width and r.y <= my < r.y + r.height:
                        return True
        except Exception:
            pass
        return False


    def _spine_smooth(self, context, iterations=1):
        """Set Flow: even angular spacing on rings around the spine,
        keeping each vert's radius — loops follow the curve cleanly
        without thinning the mesh.
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None or not self.spine_bind:
            return
        bm.verts.ensure_lookup_table()
        self._spine_push_undo(context)

        n_bins = max(6, min(48, len(self.spine_bind) // 3))
        bins = [[] for _ in range(n_bins)]
        for item in self.spine_bind:
            vidx, t = item[0], float(item[1])
            if vidx >= len(bm.verts):
                continue
            bi = min(n_bins - 1, max(0, int(t * n_bins)))
            bins[bi].append(vidx)

        for group in bins:
            if len(group) < 4:
                continue
            # average t of group
            ts = []
            for vidx in group:
                for item in self.spine_bind:
                    if item[0] == vidx:
                        ts.append(float(item[1]))
                        break
            if not ts:
                continue
            t_avg = sum(ts) / len(ts)
            center = eval_bezier_points(self.bez, t_avg)
            tan = bezier_chain_tangent(self.bez, t_avg)
            tmp = Vector((0, 0, 1)) if abs(tan.z) < 0.9 else Vector((1, 0, 0))
            x_axis = tan.cross(tmp)
            if x_axis.length < 1e-8:
                continue
            x_axis.normalize()
            y_axis = tan.cross(x_axis).normalized()

            polar = []
            for vidx in group:
                rel = bm.verts[vidx].co - center
                rel = rel - tan * rel.dot(tan)
                radius = rel.length
                if radius < 1e-10:
                    continue
                ang = math.atan2(rel.dot(y_axis), rel.dot(x_axis))
                polar.append([vidx, ang, radius])
            if len(polar) < 4:
                continue
            polar.sort(key=lambda it: it[1])
            n = len(polar)
            a0 = polar[0][1]
            # Blend toward even spacing (not 100% — keeps character)
            blend = 0.55
            for i, (vidx, ang, radius) in enumerate(polar):
                target = a0 + (2.0 * math.pi * i) / n
                # shortest angular lerp
                d = (target - ang + math.pi) % (2.0 * math.pi) - math.pi
                new_ang = ang + d * blend
                new_rel = (
                    x_axis * (math.cos(new_ang) * radius)
                    + y_axis * (math.sin(new_ang) * radius)
                )
                bm.verts[vidx].co = center + new_rel

        # Soft pass then re-base bind so next grab is clean
        self._spine_auto_soft(bm, iterations=5, factor=0.45)
        self._spine_rebase_from_mesh(context)
        context.area.tag_redraw()
        self.report({'INFO'}, "Set Flow (loops around spine)")





    def _spine_rebase_from_mesh(self, context):
        """Current mesh + controllers become the new rest (keeps deformation baked)."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None or not getattr(self, 'bez', None):
            return
        bm.verts.ensure_lookup_table()
        self.rest_bez = copy_bezier_points(self.bez)
        self.spine_points = [bp['co'].copy() for bp in self.bez]
        self.spine_rest_poly = [bp['co'].copy() for bp in self.bez]
        n = len(self.bez)
        self.handle_params = [i / max(1, n - 1) for i in range(n)]
        if self.handle_params:
            self.handle_params[0] = 0.0
            self.handle_params[-1] = 1.0

        samples = max(64, n * 32)
        sample_pts = [eval_bezier_points(self.rest_bez, s / samples) for s in range(samples + 1)]
        influence = getattr(self, 'spine_influence', None)
        # Prefer previous bound set; if empty, re-find near curve
        prev = [item[0] for item in (self.spine_bind or [])]
        candidates = prev if prev else [v.index for v in bm.verts]
        new_bind = []
        for vidx in candidates:
            if vidx >= len(bm.verts):
                continue
            co = bm.verts[vidx].co.copy()
            best_t, best_d = 0.0, 1e18
            for s, p in enumerate(sample_pts):
                d = (p - co).length_squared
                if d < best_d:
                    best_d = d
                    best_t = s / samples
            if influence is not None and math.sqrt(best_d) > influence * 1.35:
                continue
            on = eval_bezier_points(self.rest_bez, best_t)
            tan = bezier_chain_tangent(self.rest_bez, best_t)
            new_bind.append((vidx, best_t, (co - on).copy(), tan.copy(), float(math.sqrt(best_d))))
            self.all_rest[vidx] = co
        self.spine_bind = new_bind
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)




    def _spine_toggle_in_front(self, context, all_chains=False):
        """Toggle draw In-Front (no depth test) per chain.
        N = active chain only.
        Shift+N = if mixed → force all ON; if uniform → flip all.
        """
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            # Single live curve (place / no multi-chain yet)
            cur = bool(getattr(self, 'spine_in_front', True))
            self.spine_in_front = not cur
            state = "ON" if self.spine_in_front else "OFF"
            self.report({'INFO'}, f"In Front: {state}")
            context.area.tag_redraw()
            return True
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        if all_chains:
            flags = [bool(ch.get('in_front', True)) for ch in chains]
            if any(flags) and not all(flags):
                # Mixed → first unify to all ON
                for ch in chains:
                    ch['in_front'] = True
                msg = f"In Front: ALL ON ({len(chains)} chains)"
            else:
                # Uniform → flip all
                new_val = not flags[0]
                for ch in chains:
                    ch['in_front'] = new_val
                msg = f"In Front: ALL {'ON' if new_val else 'OFF'} ({len(chains)} chains)"
        else:
            ai = int(getattr(self, 'active_chain', 0) or 0)
            if not (0 <= ai < len(chains)):
                return False
            ch = chains[ai]
            ch['in_front'] = not bool(ch.get('in_front', True))
            msg = f"Chain {ai + 1} In Front: {'ON' if ch['in_front'] else 'OFF'}"
        # Mirror active flag onto operator for single-chain draw paths
        ai = int(getattr(self, 'active_chain', 0) or 0)
        if 0 <= ai < len(chains):
            self.spine_in_front = bool(chains[ai].get('in_front', True))
        self.report({'INFO'}, msg)
        context.area.tag_redraw()
        return True

    def _spine_set_influence_falloff(self, context, mode='CONSTANT'):
        """Set influence falloff on selected controllers (or active / all on chain)."""
        mode = str(mode or 'CONSTANT').upper().replace(' ', '_')
        if mode not in _INFLUENCE_FALLOFF_ORDER:
            mode = 'CONSTANT'
        chains = getattr(self, 'spine_chains', None) or []
        if not chains and getattr(self, 'bez', None):
            chains = [{'bez': self.bez}]
            self.spine_chains = chains
        if not chains:
            self.report({'INFO'}, "No chain")
            return False
        try:
            self._spine_push_undo(context)
        except Exception:
            pass
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        ac = int(getattr(self, 'active_chain', 0) or 0)
        targets = []
        for item in (getattr(self, 'selected', None) or set()):
            if len(item) == 3:
                ci, i, p = item
                if p == 'co':
                    targets.append((ci, i))
            elif len(item) == 2:
                i, p = item
                if p == 'co':
                    targets.append((ac, i))
        if not targets and getattr(self, 'active_handle', None) is not None:
            targets = [(ac, int(self.active_handle))]
        if not targets:
            n = len((chains[ac].get('bez') if 0 <= ac < len(chains) else None) or [])
            targets = [(ac, i) for i in range(n)]
        n_changed = 0
        for ci, i in targets:
            if not (0 <= ci < len(chains)):
                continue
            ch = chains[ci]
            bez = ch.get('bez') or []
            n = len(bez)
            pfo = ensure_point_inf_falloff(n, ch.get('point_inf_falloff'), default='CONSTANT')
            if not (0 <= i < n):
                continue
            pfo[i] = mode
            ch['point_inf_falloff'] = pfo
            # Falloff is part of the live deform envelope. Invalidate all
            # cached soft weights immediately so changing falloff updates
            # the mesh on the very next apply (and during subsequent R/S).
            ch['_fw'] = None
            ch['_sw_key'] = None
            ch['_soft_w'] = None
            n_changed += 1

        # The influence-marker overlay caches its computed weights separately
        # from the live deform weights.  Falloff changes must invalidate that
        # cache too, otherwise the mesh updates while the visible markers keep
        # showing the previous falloff until another event rebuilds them.
        self._influence_overlay_cache = None
        self.influence_falloff = mode
        if 0 <= ac < len(chains):
            ch = chains[ac]
            self.point_inf_falloff = ensure_point_inf_falloff(
                len(ch.get('bez') or []), ch.get('point_inf_falloff'), default='CONSTANT'
            )
        if getattr(self, 'tool_mode', '') == 'SPINE_DEFORM':
            try:
                self._spine_apply(context)
            except Exception:
                pass
        label = mode.replace('_', ' ').title()
        self.report({'INFO'}, f"Controller falloff: {label}" + (f"  (x{n_changed})" if n_changed > 1 else ""))
        context.area.tag_redraw()
        return True

    def _spine_cycle_influence_falloff(self, context):
        """Legacy cycle — prefers popup; kept for compatibility."""
        order = _INFLUENCE_FALLOFF_ORDER
        ac = int(getattr(self, 'active_chain', 0) or 0)
        chains = getattr(self, 'spine_chains', None) or []
        cur = 'CONSTANT'
        if chains and 0 <= ac < len(chains):
            pfo = chains[ac].get('point_inf_falloff') or []
            i = getattr(self, 'active_handle', None)
            if i is not None and 0 <= int(i) < len(pfo):
                cur = str(pfo[int(i)] or 'CONSTANT').upper()
            elif pfo:
                cur = str(pfo[0] or 'CONSTANT').upper()
        try:
            idx = order.index(cur)
        except ValueError:
            idx = 0
        return self._spine_set_influence_falloff(context, order[(idx + 1) % len(order)])

    def _spine_reset_influence_radius(self, context):
        """Reset selected Spine controller influence radii to their bind-time defaults."""
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            self.report({'INFO'}, "No bound chains — bind first")
            return False

        # Radius expansion must operate on the current BMesh.  The previous
        # version called _spine_absorb_vg_into_bind(obj, bm, ...) here without
        # defining obj/bm, so the exception was silently swallowed and the
        # expanded reach was never made persistent.
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return False
        bm.verts.ensure_lookup_table()
        # First persist the live active-chain values, then capture the undo
        # snapshot.  The old order captured the snapshot BEFORE syncing the
        # live controller state back into spine_chains.  In practice that could
        # make Ctrl+Z restore a stale chain state (especially after interactive
        # radius changes), so Alt+R appeared to have no undoable previous value.
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        ac = int(getattr(self, 'active_chain', 0) or 0)
        raw = []
        for item in (getattr(self, 'selected', None) or set()):
            if len(item) == 3:
                ci, i, part = item
                if part == 'co':
                    raw.append((int(ci), int(i)))
            elif len(item) == 2:
                i, part = item
                if part == 'co':
                    raw.append((ac, int(i)))
        if not raw and getattr(self, 'active_handle', None) is not None:
            raw = [(ac, int(self.active_handle))]
        if not raw:
            self.report({'INFO'}, "Select a controller first")
            return False

        # Capture exactly the state immediately before Alt+R changes anything.
        # This is deliberately after selection validation and after syncing the
        # active chain, so Ctrl+Z always has a valid, current pre-reset state.
        try:
            self._spine_push_undo(context)
        except Exception:
            pass

        changed = 0
        touched = set()
        for ci, i in raw:
            if ci < 0 or ci >= len(chains):
                continue
            ch = chains[ci]
            bez = ch.get('bez') or []
            if i < 0 or i >= len(bez):
                continue
            n = len(bez)
            legacy = float(ch.get('influence') or 0.1) or 0.1
            pinf = ensure_point_influence(n, ch.get('point_influence'), default=legacy)
            defaults = ch.get('point_influence_default')
            if not defaults:
                # Legacy chains created before persistent defaults: current
                # radius is the safest available reset baseline.
                defaults = list(pinf)
            defaults = ensure_point_influence(n, defaults, default=(defaults[0] if defaults else legacy))
            target = max(1e-4, min(1.0e6, float(defaults[i])))
            if abs(float(pinf[i]) - target) > 1e-8:
                pinf[i] = target
                changed += 1
            ch['point_influence'] = pinf
            ch['point_influence_default'] = defaults
            ch['influence'] = max(pinf) if pinf else legacy
            ch['_fw'] = None
            ch['_sw_key'] = None
            ch['_soft_w'] = None
            touched.add(ci)
            if ci == ac:
                self.point_influence = list(pinf)
                self.spine_influence = ch['influence']

        self.spine_chains = chains
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        if getattr(self, 'tool_mode', '') == 'SPINE_DEFORM' and touched:
            self._spine_apply(context, auto_soft=False)
        self._influence_overlay_cache = None
        context.area.tag_redraw()
        self.report({'INFO'}, f"Influence radius reset: {changed} controller(s)")
        return True

    def _spine_adjust_influence(self, context, delta):
        """Adjust influence radius of SELECTED controllers only.

        Uses a fixed additive step, so increasing and decreasing move by exactly
        the same amount. Keyboard and wheel changes are intentionally Spine-only.
        Increasing Radius also expands the bind from the active Spine vertex
        group so newly reached painted vertices remain reachable after Confirm
        and after reopening the tool.
        """
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            self.report({'INFO'}, "No bound chains — bind first")
            return False
        # Radius expansion needs the live mesh/BMesh. Older revisions called
        # _spine_absorb_vg_into_bind() with undefined obj/bm here, so the
        # exception was silently swallowed and the reach was never expanded.
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            self.report({'WARNING'}, "Radius: mesh is not available")
            return False
        now = time.monotonic()
        last_undo = float(getattr(self, '_influence_adjust_last_undo', 0.0) or 0.0)
        if now - last_undo > 0.45:
            self._spine_push_undo(context)
            self._influence_adjust_last_undo = now
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        ai = int(getattr(self, 'active_chain', 0) or 0)
        if ai < 0 or ai >= len(chains):
            return False

        # Selected controllers on ANY chain. Fallback: active handle, else whole active chain.
        raw = []
        for item in (getattr(self, 'selected', None) or set()):
            if len(item) == 3:
                ci, i, p = item
                if p == 'co' and 0 <= ci < len(chains):
                    raw.append((ci, i))
            elif len(item) == 2:
                i, p = item
                if p == 'co':
                    raw.append((ai, i))
        if not raw and getattr(self, 'active_handle', None) is not None:
            raw.append((ai, int(self.active_handle)))
        if not raw:
            n_ai = len(chains[ai].get('bez') or [])
            raw = [(ai, i) for i in range(n_ai)]

        by_chain = {}
        for ci, i in raw:
            bez_n = len(chains[ci].get('bez') or [])
            if 0 <= i < bez_n:
                by_chain.setdefault(ci, []).append(i)
        if not by_chain:
            return False

        last_pinf = None
        last_pfo = None
        last_targets = []
        n_ctrl_total = 0

        for ci, targets in by_chain.items():
            ch = chains[ci]
            bez = ch.get('bez') or []
            n = len(bez)
            if n < 2:
                continue
            targets = sorted(set(targets))
            n_ctrl_total += len(targets)
            last_targets = targets
            legacy = float(ch.get('influence') or 0.1) or 0.1
            pinf = ensure_point_influence(n, ch.get('point_influence'), default=legacy)
            pfo = ensure_point_inf_falloff(
                n, ch.get('point_inf_falloff'),
                default=ch.get('inf_falloff') or 'CONSTANT',
            )
            # Fixed additive step: both directions change the radius by the
            # exact same amount.  Scale the step mildly with the current radius
            # only for very large radii, so tiny influences remain controllable.
            step = float(delta)
            for i in targets:
                base = float(pinf[i] or legacy)
                amount = 0.01 if abs(base) < 1.0 else 0.02
                nxt = base + (amount if step > 0.0 else -amount)
                pinf[i] = max(1e-4, min(nxt, 1.0e6))
            ch['point_influence'] = pinf
            ch['point_inf_falloff'] = pfo
            ch['influence'] = max(pinf) if pinf else legacy
            ch['_sw_key'] = None
            ch['_soft_w'] = None
            ch['_fw'] = None
            last_pinf, last_pfo = pinf, pfo
            if ci == ai:
                self.point_influence = list(pinf)
                self.point_inf_falloff = list(pfo)
                self.spine_influence = ch['influence']

            # When Radius is increased, include any already-painted Blender
            # Weight Paint vertices that are now reachable.  This is important
            # for the Blender-weight workflow: a painted vertex that was outside
            # the old bind must become part of the Spine bind as soon as the
            # influence range reaches it.  We intentionally do this only while
            # increasing Radius; decreasing Radius must not remove painted/bound
            # vertices.
            if step > 0.0:
                try:
                    self._spine_absorb_vg_into_bind(obj, bm, ch, ci)
                except Exception:
                    pass
            chains[ci] = ch

        self.spine_chains = chains

        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        if getattr(self, 'tool_mode', '') == 'SPINE_DEFORM':
            # Tell _spine_apply this is an interactive local edit so, with
            # multiple chains, it only recomputes the active/selected chains.
            old_xform = getattr(self, '_xform_mode', None)
            self._xform_mode = 'INFLUENCE_ADJUST'
            try:
                self._spine_apply(context, auto_soft=False)
            finally:
                self._xform_mode = old_xform
        if n_ctrl_total == 1 and last_pinf is not None:
            i = last_targets[0]
            fo = str(last_pfo[i] if i < len(last_pfo) else 'CONSTANT').replace('_', ' ').title()
            self.report(
                {'INFO'},
                f"Ctrl[{i}] Influence: {last_pinf[i]:.4f}  |  Falloff: {fo}",
            )
        else:
            direction = 'increase' if delta > 0.0 else 'decrease'
            self.report(
                {'INFO'},
                f"Influence {direction}: {n_ctrl_total} controller(s)",
            )
        context.area.tag_redraw()
        return True

    def _spine_mirror_fn(self, context, obj, axis='X', space='LOCAL'):
        """Return a function local_point -> mirrored local_point."""
        axis = (axis or 'X').upper()
        space = (space or 'LOCAL').upper()
        if axis not in {'X', 'Y', 'Z'}:
            axis = 'X'
        if space not in {'LOCAL', 'CURSOR', 'WORLD'}:
            space = 'LOCAL'
        mw = obj.matrix_world.copy()
        try:
            imw = mw.inverted()
        except Exception:
            imw = Matrix.Identity(4)

        def mirror_local(p_local):
            p_local = p_local.copy()
            if space == 'LOCAL':
                d = p_local
                if axis == 'X':
                    d.x = -d.x
                elif axis == 'Y':
                    d.y = -d.y
                else:
                    d.z = -d.z
                return d
            if space == 'CURSOR':
                c_local = imw @ context.scene.cursor.location.copy()
                d = p_local - c_local
                if axis == 'X':
                    d.x = -d.x
                elif axis == 'Y':
                    d.y = -d.y
                else:
                    d.z = -d.z
                return c_local + d
            w = mw @ p_local
            if axis == 'X':
                w.x = -w.x
            elif axis == 'Y':
                w.y = -w.y
            else:
                w.z = -w.z
            return imw @ w

        return mirror_local, axis, space

    def _spine_mirror_bez_points(self, bez, mirror_fn):
        """Mirror co/hl/hr of each Bezier point; keep order. Modes applied by caller."""
        out = []
        for bp in (bez or []):
            out.append({
                'co': mirror_fn(bp['co']),
                'hl': mirror_fn(bp.get('hl', bp['co'])),
                'hr': mirror_fn(bp.get('hr', bp['co'])),
            })
        return out

    def _spine_copy_chain_dict(self, ch, new_id=None):
        """Deep-copy one spine chain (bez, modes, attrs, bind structure)."""
        n = len(ch.get('bez') or [])
        return {
            'chain_id': new_id or f"dup_{id(ch) & 0xFFFFFF:x}",
            'bez': copy_bezier_points(ch.get('bez')),
            'rest_bez': copy_bezier_points(ch.get('rest_bez') or ch.get('bez')),
            'modes': list(ch.get('modes') or ['AUTO'] * n),
            'tilt': list(ch.get('tilt') or [0.0] * n),
            'radius': list(ch.get('radius') or [1.0] * n),
            'handle_params': list(ch.get('handle_params') or []),
            'bind': [
                (
                    int(item[0]),
                    float(item[1]),
                    item[2].copy() if hasattr(item[2], 'copy') else item[2],
                    item[3].copy() if len(item) > 3 and hasattr(item[3], 'copy') else (item[3] if len(item) > 3 else None),
                    float(item[4]) if len(item) > 4 else 0.0,
                )
                for item in (ch.get('bind') or [])
            ],
            'influence': float(ch.get('influence') or 0.1),
            'point_influence': ensure_point_influence(
                n, ch.get('point_influence'), default=float(ch.get('influence') or 0.1),
            ),
            'point_inf_falloff': ensure_point_inf_falloff(
                n, ch.get('point_inf_falloff'), default='CONSTANT',
            ),
            'origin_ids': list(ch.get('origin_ids') or list(range(n))),
            'in_front': bool(ch.get('in_front', True)),
        }

    def _spine_copy_bind_rest_history(self, context, src_ch, new_ch, mirror_fn=None):
        """Copy first-bind rest history from src chain_id to the duplicated chain.

        Ensures Ctrl+R / reset works on the duplicate with the same baseline
        (controller pose + bind_verts + mesh_rest) as the source chain.
        When mirror_fn is set, mirror the saved bez snapshot for the copy.
        """
        obj, _bm = self.get_obj_bm(context)
        if obj is None or not src_ch or not new_ch:
            return False
        src_cid = src_ch.get('chain_id')
        new_cid = new_ch.get('chain_id')
        if not new_cid:
            return False

        data = _vdh_get_bind_rest(obj)
        saved_list = list(data.get('chains') or [])
        saved_by_id = {s.get('chain_id'): s for s in saved_list if s.get('chain_id')}
        src_entry = None
        if src_cid and src_cid in saved_by_id:
            src_entry = saved_by_id[src_cid]
        if src_entry is None:
            src_entry = self._spine_match_saved_for_chain(src_ch, saved_list, saved_by_id)
        if src_entry is None or not src_entry.get('bez') or len(src_entry.get('bez') or []) < 2:
            # Fallback: build a history entry from the source chain's current rest
            rest_src = src_ch.get('rest_bez') or src_ch.get('bez')
            if not rest_src or len(rest_src) < 2:
                return False
            n = len(rest_src)
            src_entry = {
                'chain_id': src_cid,
                'bez': copy_bezier_points(rest_src),
                'tilt': list(src_ch.get('tilt') or [0.0] * n),
                'radius': list(src_ch.get('radius') or [1.0] * n),
                'modes': list(src_ch.get('modes') or ['AUTO'] * n),
                'origin_ids': list(src_ch.get('origin_ids') or list(range(n))),
                'influence': float(src_ch.get('influence') or 0.1),
                'point_influence': ensure_point_influence(
                    n, src_ch.get('point_influence'),
                    default=float(src_ch.get('influence') or 0.1),
                ),
                'point_inf_falloff': ensure_point_inf_falloff(
                    n, src_ch.get('point_inf_falloff'), default='CONSTANT',
                ),
                'bind_verts': [int(item[0]) for item in (src_ch.get('bind') or [])],
            }

        # Deep-copy under new chain_id
        bez = copy_bezier_points(src_entry.get('bez'))
        if mirror_fn is not None and bez:
            try:
                bez = self._spine_mirror_bez_points(bez, mirror_fn)
            except Exception:
                pass
        n = len(bez or [])
        new_entry = {
            'chain_id': new_cid,
            'bez': bez,
            'tilt': list(src_entry.get('tilt') or [0.0] * n)[:n],
            'radius': list(src_entry.get('radius') or [1.0] * n)[:n],
            'modes': list(src_entry.get('modes') or ['AUTO'] * n)[:n],
            'origin_ids': list(src_entry.get('origin_ids') or list(range(n))),
            'influence': float(src_entry.get('influence') or 0.1),
            'point_influence': ensure_point_influence(
                n, src_entry.get('point_influence'),
                default=float(src_entry.get('influence') or 0.1),
            ),
            'point_inf_falloff': ensure_point_inf_falloff(
                n, src_entry.get('point_inf_falloff'), default='CONSTANT',
            ),
            # Same verts as source until rebind updates them
            'bind_verts': [int(v) for v in (src_entry.get('bind_verts') or [])],
        }
        while len(new_entry['tilt']) < n:
            new_entry['tilt'].append(0.0)
        while len(new_entry['radius']) < n:
            new_entry['radius'].append(1.0)
        while len(new_entry['modes']) < n:
            new_entry['modes'].append('AUTO')
        if len(new_entry['origin_ids']) != n:
            new_entry['origin_ids'] = list(range(n))

        # Replace any existing entry with same new_cid, keep the rest
        out_chains = [s for s in saved_list if s.get('chain_id') != new_cid]
        out_chains.append(new_entry)
        mesh_rest = dict(data.get('mesh_rest') or {})
        payload = {'chains': out_chains, 'mesh_rest': mesh_rest}
        _vdh_set_bind_rest(obj, payload)
        try:
            _vdh_persist_bind_rest_to_mesh(obj, payload)
        except Exception:
            pass
        return True


    def _spine_prune_empty_active_chain(self, context):
        """If active chain has < 2 controllers, remove it and activate previous/last.

        Used in Edit Place after deleting all (or enough) controllers of a chain.
        """
        if not getattr(self, '_spine_edit_place', False):
            return False
        chains = getattr(self, 'spine_chains', None)
        if not chains:
            return False
        ac = int(getattr(self, 'active_chain', 0) or 0)
        if not (0 <= ac < len(chains)):
            return False

        n_bez = len(getattr(self, 'bez', None) or [])
        n_pts = len(getattr(self, 'spine_points', None) or [])
        if n_bez >= 2 or n_pts >= 2:
            return False

        # Remove active chain without storing empty state into it.
        # IMPORTANT: Edit Place chain deletion must also remove ONLY this
        # chain's vertex group, using the group's stored name.
        removed_chain = None
        try:
            removed_chain = chains[ac]
            vg_name = str((removed_chain or {}).get('vg_name') or '').strip()
            if vg_name:
                _vdh_remove_spine_vertex_group(getattr(context, 'object', None), vg_name)
            del chains[ac]
        except Exception:
            return False
        self.spine_chains = chains

        if not chains:
            self.active_chain = 0
            self.bez = []
            self.rest_bez = []
            self.point_modes = []
            self.spine_points = []
            self.spine_tilt = []
            self.spine_radius = []
            self.handle_params = []
            self.spine_bind = []
            self._spine_origin_ids = []
            self.selected = set()
            self.active_handle = None
            self.active_bez_part = 'co'
            self._spine_placing_new_chain = True
            self._spine_last_add_idx = None
            self.report({'INFO'}, "Chain removed — no chains left")
            return True

        # Previous chain if possible, else last remaining
        new_ac = ac - 1 if ac > 0 else 0
        if new_ac >= len(chains):
            new_ac = len(chains) - 1
        self.active_chain = new_ac
        self._spine_placing_new_chain = False
        try:
            self._spine_load_active_chain()
        except Exception:
            ch = chains[new_ac]
            self.bez = ch.get('bez') or []
            self.rest_bez = ch.get('rest_bez') or self.bez
            self.point_modes = ch.get('modes') or ['AUTO'] * len(self.bez or [])
            self.spine_points = [bp['co'].copy() for bp in (self.bez or [])]
        self.spine_points = [bp['co'].copy() for bp in (self.bez or [])]
        self.selected = set()
        self.active_handle = 0 if self.bez else None
        self.active_bez_part = 'co'
        self._spine_last_add_idx = len(self.spine_points) - 1 if self.spine_points else None
        self.report({'INFO'}, f"Chain removed → active chain {new_ac + 1}/{len(chains)}")
        return True


    def _spine_rebind_chain_proximity(self, context, ch):
        """Rebuild bind for one chain from current mesh near the curve."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        bm.verts.ensure_lookup_table()
        rest = ch.get('rest_bez') or ch.get('bez')
        if not rest or len(rest) < 2:
            ch['bind'] = []
            return
        samples = max(32, len(rest) * 16)
        sample_pts = [eval_bezier_points(rest, s / float(samples)) for s in range(samples + 1)]
        influence = max(float(ch.get('influence') or 0.1), 1e-6)
        pinf = ch.get('point_influence') or [influence]
        if pinf:
            influence = max(influence, max(float(x) for x in pinf))
        new_bind = []
        for v in bm.verts:
            co = v.co
            best_t, best_d = 0.0, 1e18
            for s, p in enumerate(sample_pts):
                d = (p - co).length_squared
                if d < best_d:
                    best_d = d
                    best_t = s / float(samples)
            best_d = best_d ** 0.5
            if best_d > influence * 1.5:
                continue
            on = eval_bezier_points(rest, best_t)
            tan = bezier_chain_tangent(rest, best_t)
            new_bind.append((v.index, float(best_t), (co - on).copy(), tan.copy(), float(best_d)))
            self.all_rest[v.index] = co.copy()
        ch['bind'] = new_bind
        try:
            self._spine_unify_bind_rings(bm, ch)
        except Exception:
            pass


    def _spine_start_grab_active_chain(self, context, event):
        """Enter translate on every controller of the active chain (post Shift+D)."""
        if event is None:
            return
        tm = getattr(self, 'tool_mode', '')
        if tm == 'SPINE_PLACE':
            pts = getattr(self, 'spine_points', None) or []
            n = len(pts)
            if n < 1:
                return
            self.selected = {(i, 'co') for i in range(n)}
            self.active_handle = 0
            self.active_bez_part = 'co'
            try:
                self._spine_start_place_drag(context, event, 0, part='co')
            except Exception:
                pass
            return
        # Deform
        chains = getattr(self, 'spine_chains', None) or []
        ai = int(getattr(self, 'active_chain', 0) or 0)
        bez = getattr(self, 'bez', None) or []
        if chains and 0 <= ai < len(chains):
            bez = chains[ai].get('bez') or bez
        n = len(bez or [])
        if n < 1:
            return
        self.selected = {(ai, i, 'co') for i in range(n)}
        self.active_handle = 0
        self.active_bez_part = 'co'
        try:
            self.start_drag(context, event, 0, 'co')
        except Exception:
            pass


    def _spine_curve_arc_length(self, bez, samples=64):
        if not bez or len(bez) < 2:
            return 0.0
        total = 0.0
        prev = eval_bezier_points(bez, 0.0)
        for s in range(1, samples + 1):
            p = eval_bezier_points(bez, s / float(samples))
            total += (p - prev).length
            prev = p
        return float(total)

    def _spine_make_straight_bez(self, bez, axis='FREE'):
        """Build a straight Bezier: same arc length along chosen axis.

        axis: 'FREE' = endpoint direction (current behavior)
              'X'/'Y'/'Z' = force that world/local object axis (object-local)
        Returns (new_bez, axis_dir, p0, arc).
        """
        if not bez or len(bez) < 2:
            return (copy_bezier_points(bez) if bez else []), Vector((0, 0, 1)), Vector(), 0.0
        cos_old = [bp['co'].copy() for bp in bez]
        n = len(cos_old)
        ax = str(axis or 'FREE').upper()

        # Centroid of controllers (anchor for forced axes)
        centroid = sum(cos_old, Vector((0, 0, 0))) / float(n)

        if ax in ('X', 'Y', 'Z'):
            axis_dir = {'X': Vector((1, 0, 0)), 'Y': Vector((0, 1, 0)), 'Z': Vector((0, 0, 1))}[ax]
            # Orient along the dominant side of the current curve
            p_start = cos_old[0]
            p_end = cos_old[-1]
            if (p_end - p_start).dot(axis_dir) < 0:
                axis_dir = -axis_dir
            # X/Y/Z: keep the START (first controller) fixed and move only the END.
            # This is the requested reversed anchoring.
            anchor_start = cos_old[0].copy()
            arc = self._spine_curve_arc_length(bez)
            if arc < 1e-8:
                arc = max((cos_old[-1] - cos_old[0]).length, 1e-4)
            p0 = anchor_start.copy()
            axis_vec = axis_dir * arc
        else:
            # FREE: direction from first to last controller
            p0 = cos_old[0].copy()
            p_end = cos_old[-1].copy()
            axis_vec = p_end - p0
            if axis_vec.length < 1e-10:
                best_d, axis_dir = 0.0, Vector((0, 0, 1))
                for p in cos_old:
                    d = p - centroid
                    if d.length > best_d:
                        best_d = d.length
                        if d.length > 1e-12:
                            axis_dir = d.normalized()
                p0 = centroid - axis_dir * (best_d * 0.5)
            else:
                axis_dir = axis_vec.normalized()
            # FREE: keep BOTH endpoints exactly where they are. Only the middle
            # controllers/handles are straightened between the existing endpoints.
            # Therefore use the endpoint chord, not the curved arc length.
            arc = axis_vec.length
            if arc < 1e-8:
                arc = 1e-4

        chord = [0.0]
        for i in range(1, n):
            chord.append(chord[-1] + (cos_old[i] - cos_old[i - 1]).length)
        chord_total = chord[-1] if chord[-1] > 1e-12 else 1.0

        new_cos = []
        for i in range(n):
            u = chord[i] / chord_total
            new_cos.append(p0 + axis_dir * (arc * u))

        new_bez = make_bezier_points(new_cos, poly=new_cos)
        for i in range(n):
            bp = new_bez[i]
            if i == 0:
                bp['hl'] = bp['co'].copy()
                L = (new_cos[1] - new_cos[0]).length / 3.0 if n > 1 else 0.0
                bp['hr'] = bp['co'] + axis_dir * L
            elif i == n - 1:
                bp['hr'] = bp['co'].copy()
                L = (new_cos[-1] - new_cos[-2]).length / 3.0
                bp['hl'] = bp['co'] - axis_dir * L
            else:
                L_l = (new_cos[i] - new_cos[i - 1]).length / 3.0
                L_r = (new_cos[i + 1] - new_cos[i]).length / 3.0
                bp['hl'] = bp['co'] - axis_dir * L_l
                bp['hr'] = bp['co'] + axis_dir * L_r
        return new_bez, axis_dir, p0, arc

    def _spine_regularize_rings_on_axis(self, bm, ch, axis_dir, p0, arc, circularize=True, even_spacing=True):
        """Topology-aware ring regularize.

        Interior cap verts (grid-fill / dense caps) are excluded from
        circularize. They keep a fixed 2D offset relative to the boundary
        ring so they stay inside the cap and do not collapse into each other.
        """
        bind = list(ch.get('bind') or [])
        if not bind or arc < 1e-8:
            return 0, 0
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        axis_dir = axis_dir.normalized() if axis_dir.length > 1e-12 else Vector((0, 0, 1))
        bound = set()
        for item in bind:
            vidx = int(item[0])
            if vidx < len(bm.verts):
                bound.add(vidx)
        if len(bound) < 6:
            return 0, 0

        def axis_s(vidx):
            return (bm.verts[vidx].co - p0).dot(axis_dir)

        # --- Detect interior cap verts (not boundary edges of tube) ---
        # Cap faces: normal almost parallel to tube axis
        cap_face_verts = set()
        for f in bm.faces:
            ids = [v.index for v in f.verts if v.index in bound]
            if len(ids) < 3:
                continue
            n = f.normal
            if n.length < 1e-12:
                continue
            if abs(n.normalized().dot(axis_dir)) > 0.72:
                for vi in ids:
                    cap_face_verts.add(vi)

        # Ring adjacency among bound verts
        ring_adj = {v: [] for v in bound}
        long_adj = {v: [] for v in bound}
        for e in bm.edges:
            i0, i1 = e.verts[0].index, e.verts[1].index
            if i0 not in bound or i1 not in bound:
                continue
            d = bm.verts[i1].co - bm.verts[i0].co
            if d.length < 1e-12:
                continue
            along = abs(d.normalized().dot(axis_dir))
            if along < 0.45:
                ring_adj[i0].append(i1)
                ring_adj[i1].append(i0)
            elif along > 0.55:
                long_adj[i0].append(i1)
                long_adj[i1].append(i0)

        def clean_val2(adj):
            out = {}
            for v, nbs in adj.items():
                uniq = list(dict.fromkeys(nbs))
                if len(uniq) <= 2:
                    out[v] = uniq
                else:
                    scored = []
                    for j in uniq:
                        d = bm.verts[j].co - bm.verts[v].co
                        if d.length < 1e-12:
                            continue
                        scored.append((1.0 - abs(d.normalized().dot(axis_dir)), j))
                    scored.sort(key=lambda x: -x[0])
                    out[v] = [j for _, j in scored[:2]]
            return out

        ring_adj = clean_val2(ring_adj)

        # Clean ring cycles (tube body loops)
        visited = set()
        cycles = []
        for start_v in bound:
            if start_v in visited:
                continue
            if len(ring_adj.get(start_v, [])) != 2:
                continue
            cycle = [start_v]
            prev, cur = start_v, ring_adj[start_v][0]
            ok = True
            for _ in range(len(bound) + 2):
                if cur == start_v:
                    break
                if cur in visited:
                    ok = False
                    break
                cycle.append(cur)
                nbs = ring_adj.get(cur, [])
                nxts = [n for n in nbs if n != prev]
                if len(nxts) != 1:
                    ok = False
                    break
                prev, cur = cur, nxts[0]
            else:
                ok = False
            if not (ok and cur == start_v and len(cycle) >= 4):
                continue
            for v in cycle:
                visited.add(v)
            cycles.append(cycle)

        on_clean_ring = set()
        for cy in cycles:
            for v in cy:
                on_clean_ring.add(v)

        # Interior = on a cap face, not on a clean tube ring cycle
        interior = set()
        for v in cap_face_verts:
            if v not in on_clean_ring:
                interior.add(v)
        # Also: high valence in ring-ish edges but not in any cycle → fill center
        for v in bound:
            if v in interior or v in on_clean_ring:
                continue
            if v in cap_face_verts and len(ring_adj.get(v, [])) != 2:
                interior.add(v)

        # Body verts for regularize
        body = bound - interior
        if len(body) < 6:
            return 0, 0

        # Plane frame
        tmp = Vector((0, 0, 1)) if abs(axis_dir.z) < 0.9 else Vector((1, 0, 0))
        x_axis = axis_dir.cross(tmp)
        if x_axis.length < 1e-12:
            x_axis = Vector((1, 0, 0))
        x_axis.normalize()
        y_axis = axis_dir.cross(x_axis).normalized()

        def polar_ang(vidx, center):
            d = bm.verts[vidx].co - center
            return math.atan2(d.dot(y_axis), d.dot(x_axis))

        def polar_r(vidx, center):
            d = bm.verts[vidx].co - center
            return (d - axis_dir * d.dot(axis_dir)).length

        # --- Snapshot interior relative to nearest end boundary ring ---
        # Boundary rings = clean cycles closest to min/max s among body
        ring_data = []  # (mean_s, cycle)
        for cy in cycles:
            if not cy:
                continue
            ms = sum(axis_s(v) for v in cy) / float(len(cy))
            ring_data.append((ms, cy))
        ring_data.sort(key=lambda x: x[0])

        end_rings = []  # list of (label, cycle, center, mean_s)
        if ring_data:
            end_rings.append(('lo', ring_data[0][1], ring_data[0][0]))
            if len(ring_data) > 1:
                end_rings.append(('hi', ring_data[-1][1], ring_data[-1][0]))

        # Precompute boundary centers before regularize
        def ring_center(cy):
            c = Vector((0, 0, 0))
            for v in cy:
                c += bm.verts[v].co
            return c / float(len(cy)) if cy else p0.copy()

        interior_snap = {}  # vidx -> (end_label, ux, uy)
        if interior and end_rings:
            for v in interior:
                if v >= len(bm.verts):
                    continue
                s = axis_s(v)
                # pick nearer end
                best = None
                best_d = 1e18
                for label, cy, ms in end_rings:
                    d = abs(s - ms)
                    if d < best_d:
                        best_d = d
                        best = (label, cy, ms)
                if best is None:
                    continue
                label, cy, ms = best
                cen = ring_center(cy)
                d = bm.verts[v].co - cen
                # remove axis component
                d = d - axis_dir * d.dot(axis_dir)
                ux = d.dot(x_axis)
                uy = d.dot(y_axis)
                interior_snap[v] = (label, ux, uy)

        # Rebuild ring_adj / cycles using only body verts
        ring_adj_b = {v: [n for n in ring_adj.get(v, []) if n in body] for v in body}
        ring_adj_b = clean_val2(ring_adj_b)

        visited = set()
        cycles = []
        for start_v in body:
            if start_v in visited:
                continue
            if len(ring_adj_b.get(start_v, [])) != 2:
                continue
            cycle = [start_v]
            prev, cur = start_v, ring_adj_b[start_v][0]
            ok = True
            for _ in range(len(body) + 2):
                if cur == start_v:
                    break
                if cur in visited or cur not in body:
                    ok = False
                    break
                cycle.append(cur)
                nbs = ring_adj_b.get(cur, [])
                nxts = [n for n in nbs if n != prev]
                if len(nxts) != 1:
                    ok = False
                    break
                prev, cur = cur, nxts[0]
            else:
                ok = False
            if not (ok and cur == start_v and len(cycle) >= 4):
                continue
            for v in cycle:
                visited.add(v)
            cycles.append(cycle)

        if len(cycles) < 2:
            # Still restore interior if we only have snap (nothing to circularize)
            if interior_snap:
                # no boundary moved — leave as-is
                return 0, 0
            return 0, 0

        # Sort rings along axis
        scored = []
        for cy in cycles:
            ms = sum(axis_s(v) for v in cy) / float(len(cy))
            scored.append((ms, cy))
        scored.sort(key=lambda x: x[0])
        cycles = [cy for _, cy in scored]
        n_rings = len(cycles)
        s_vals = [sum(axis_s(v) for v in cy) / float(len(cy)) for cy in cycles]

        if even_spacing and n_rings >= 2:
            s0, s1 = s_vals[0], s_vals[-1]
            targets_s = [s0 + (s1 - s0) * (i / (n_rings - 1)) for i in range(n_rings)]
        else:
            targets_s = list(s_vals)

        # Longitudinal columns
        long_adj_b = {v: [n for n in long_adj.get(v, []) if n in body] for v in body}
        columns = []
        used_all = set()
        seed = list(cycles[0])
        # order seed by angle
        c0 = p0 + axis_dir * targets_s[0]
        seed.sort(key=lambda v: polar_ang(v, c0))
        for start in seed:
            if start in used_all:
                continue
            col = [start]
            used_all.add(start)
            cur = start
            for ri in range(1, n_rings):
                # prefer long edge into next ring
                nxt = None
                for n in long_adj_b.get(cur, []):
                    if n in cycles[ri] and n not in used_all:
                        nxt = n
                        break
                if nxt is None:
                    cc = p0 + axis_dir * targets_s[ri]
                    pa = polar_ang(cur, p0 + axis_dir * targets_s[ri - 1])
                    best_d, best = 1e18, None
                    for n in cycles[ri]:
                        if n in used_all:
                            continue
                        da = abs(polar_ang(n, cc) - pa)
                        da = min(da, abs(da - 2.0 * math.pi))
                        if da < best_d:
                            best_d, best = da, n
                    nxt = best
                if nxt is None:
                    break
                col.append(nxt)
                used_all.add(nxt)
                cur = nxt
            if len(col) >= 2:
                columns.append(col)

        if not columns:
            return 0, 0

        n_loop = len(columns)
        all_r = []
        for col in columns:
            for ri, vidx in enumerate(col):
                if ri >= len(targets_s):
                    continue
                center = p0 + axis_dir * targets_s[ri]
                r = polar_r(vidx, center)
                if r > 1e-8:
                    all_r.append(r)
        global_r = (sum(all_r) / float(len(all_r))) if all_r else 1e-4
        global_r = max(global_r, 1e-6)

        total_verts = 0
        for j, col in enumerate(columns):
            ang = (2.0 * math.pi * j) / float(n_loop)
            ca, sa = math.cos(ang), math.sin(ang)
            for ri, vidx in enumerate(col):
                if ri >= len(targets_s) or vidx >= len(bm.verts):
                    break
                if vidx in interior:
                    continue  # never move interior
                center = p0 + axis_dir * targets_s[ri]
                r_use = global_r if circularize else max(polar_r(vidx, center), 1e-6)
                new_co = center + x_axis * (ca * r_use) + y_axis * (sa * r_use)
                bm.verts[vidx].co = new_co
                if hasattr(self, 'all_rest') and isinstance(self.all_rest, dict):
                    self.all_rest[vidx] = new_co.copy()
                total_verts += 1

        # --- Restore interior relative to updated boundary rings ---
        if interior_snap and ring_data:
            # Map label -> updated cycle center from cycles after move
            # Find current lo/hi body rings
            body_rings = []
            for cy in cycles:
                ms = sum(axis_s(v) for v in cy) / float(len(cy))
                body_rings.append((ms, cy))
            body_rings.sort(key=lambda x: x[0])
            label_to_cy = {}
            if body_rings:
                label_to_cy['lo'] = body_rings[0][1]
                label_to_cy['hi'] = body_rings[-1][1] if len(body_rings) > 1 else body_rings[0][1]

            for vidx, (label, ux, uy) in interior_snap.items():
                if vidx >= len(bm.verts):
                    continue
                cy = label_to_cy.get(label)
                if not cy:
                    continue
                cen = ring_center(cy)
                # place in same plane as boundary (center already on axis-ish)
                # project center onto axis for stability
                s_c = (cen - p0).dot(axis_dir)
                cen_ax = p0 + axis_dir * s_c
                new_co = cen_ax + x_axis * ux + y_axis * uy
                bm.verts[vidx].co = new_co
                if hasattr(self, 'all_rest') and isinstance(self.all_rest, dict):
                    self.all_rest[vidx] = new_co.copy()
                total_verts += 1

        return n_rings, total_verts


    def _spine_circularize_rings_to_axis(self, context, active_only=True):
        """Circularize each transverse tube ring around its own center.

        This is deliberately separate from Shift+E alignment.  It changes only
        the in-plane radial distances of each ring, using that ring's own mean
        radius, so thickness is made round without forcing one global thickness
        along the whole tube.  Vertex angles and the ring center are preserved.
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None or self.tool_mode != 'SPINE_DEFORM':
            return False
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return False
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        cis = [int(getattr(self, 'active_chain', 0) or 0)] if active_only else list(range(len(chains)))
        total_rings = 0
        total_changed = 0

        for ci in cis:
            if ci < 0 or ci >= len(chains):
                continue
            ch = chains[ci]

            # Tilt and Shrink/Inflate are neutralized by the centralized
            # geometry-layer wrapper; this method only edits base geometry.
            bind = list(ch.get('bind') or [])
            bez = ch.get('bez') or []
            if len(bind) < 6 or len(bez) < 2:
                continue

            p0 = bez[0]['co'].copy()
            p1 = bez[-1]['co'].copy()
            axis = p1 - p0
            if axis.length < 1e-10:
                continue
            axis.normalize()

            bound = set()
            for item in bind:
                try:
                    vi = int(item[0])
                except Exception:
                    continue
                if 0 <= vi < len(bm.verts):
                    bound.add(vi)
            if len(bound) < 6:
                continue

            # Transverse topology: edges whose direction is predominantly
            # perpendicular to the blue spine.
            ring_adj = {vi: [] for vi in bound}
            for e in bm.edges:
                a, b = e.verts[0].index, e.verts[1].index
                if a not in bound or b not in bound:
                    continue
                d = bm.verts[b].co - bm.verts[a].co
                if d.length < 1e-10:
                    continue
                along = abs(d.normalized().dot(axis))
                if along < 0.52:
                    ring_adj[a].append(b)
                    ring_adj[b].append(a)

            # Keep at most the two strongest transverse neighbors.
            for vi, nbs in list(ring_adj.items()):
                uniq = list(dict.fromkeys(nbs))
                if len(uniq) > 2:
                    scored = []
                    for nj in uniq:
                        d = bm.verts[nj].co - bm.verts[vi].co
                        if d.length > 1e-10:
                            scored.append((1.0 - abs(d.normalized().dot(axis)), nj))
                    scored.sort(reverse=True)
                    uniq = [nj for _, nj in scored[:2]]
                ring_adj[vi] = uniq

            # Extract clean closed transverse rings.
            visited = set()
            rings = []
            for start in sorted(bound):
                if start in visited or len(ring_adj.get(start, ())) != 2:
                    continue
                ring = [start]
                prev = start
                cur = ring_adj[start][0]
                ok = True
                for _ in range(len(bound) + 2):
                    if cur == start:
                        break
                    if cur in visited or cur not in bound:
                        ok = False
                        break
                    ring.append(cur)
                    nxts = [n for n in ring_adj.get(cur, ()) if n != prev]
                    if len(nxts) != 1:
                        ok = False
                        break
                    prev, cur = cur, nxts[0]
                else:
                    ok = False
                if ok and cur == start and len(ring) >= 4:
                    for vi in ring:
                        visited.add(vi)
                    rings.append(ring)

            if not rings:
                continue

            # Stable axis-aligned frame.  We preserve each vertex's angular
            # position, so circularization does not rotate/twist the ring.
            tmp = Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((1, 0, 0))
            x_axis = axis.cross(tmp)
            if x_axis.length < 1e-12:
                x_axis = Vector((1, 0, 0))
            x_axis.normalize()
            y_axis = axis.cross(x_axis).normalized()

            for ring in rings:
                center = Vector((0.0, 0.0, 0.0))
                for vi in ring:
                    center += bm.verts[vi].co
                center /= float(len(ring))

                # Project the center onto the plane normal to the blue axis,
                # but keep its lateral offset. This prevents Q from moving the
                # tube centerline or introducing a global translation.
                center = center - axis * ((center - p0).dot(axis)) + axis * ((center - p0).dot(axis))

                radii = []
                angles = []
                for vi in ring:
                    d = bm.verts[vi].co - center
                    d_perp = d - axis * d.dot(axis)
                    r = d_perp.length
                    if r > 1e-10:
                        radii.append(r)
                        angles.append((vi, math.atan2(d_perp.dot(y_axis), d_perp.dot(x_axis))))

                if len(radii) < 4:
                    continue
                target_r = sum(radii) / float(len(radii))
                if target_r < 1e-10:
                    continue

                for vi, ang in angles:
                    ca, sa = math.cos(ang), math.sin(ang)
                    new_co = center + x_axis * (ca * target_r) + y_axis * (sa * target_r)
                    old = bm.verts[vi].co.copy()
                    bm.verts[vi].co = new_co
                    if (new_co - old).length_squared > 1e-14:
                        total_changed += 1
                total_rings += 1

            try:
                self._spine_rebind_chain_from_mesh(bm, ch)
            except Exception:
                pass

        if total_changed:
            try:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            except Exception:
                pass
            try:
                self._spine_load_active_chain()
            except Exception:
                pass
            try:
                self._spine_recalc_normals(context)
            except Exception:
                try:
                    bm.normal_update()
                except Exception:
                    pass
            try:
                context.area.tag_redraw()
            except Exception:
                pass

        if total_rings:
            self.report({'INFO'}, f"Rings Circularized: {total_rings} rings (per-ring thickness preserved)")
            return True
        self.report({'INFO'}, "Circularize Rings: no transverse rings found")
        return False

    def _spine_has_filled_caps(self, bm, ch, axis_dir):
        """True if mesh has filled end caps (grid-fill / N-gon disk), not open tube.

        Detects faces nearly perpendicular to the tube axis among bound verts.
        """
        bind = list(ch.get('bind') or [])
        if not bind:
            return False
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        axis_dir = axis_dir.normalized() if axis_dir.length > 1e-12 else Vector((0, 0, 1))
        bound = set()
        for item in bind:
            try:
                vidx = int(item[0])
            except Exception:
                continue
            if vidx < len(bm.verts):
                bound.add(vidx)
        if len(bound) < 8:
            return False
        # Axis extent of bound verts
        dots = []
        for vidx in bound:
            dots.append(bm.verts[vidx].co.dot(axis_dir))
        smin, smax = min(dots), max(dots)
        span = smax - smin
        if span < 1e-8:
            return False
        # Cap zone near ends (12% of length)
        band = span * 0.12
        cap_faces = 0
        for f in bm.faces:
            ids = [v.index for v in f.verts if v.index in bound]
            if len(ids) < 3:
                continue
            # face near an end?
            fs = [bm.verts[i].co.dot(axis_dir) for i in ids]
            fm = sum(fs) / len(fs)
            near_end = (fm - smin) < band or (smax - fm) < band
            if not near_end:
                continue
            n = f.normal
            if n.length < 1e-12:
                continue
            if abs(n.normalized().dot(axis_dir)) > 0.65:
                cap_faces += 1
                if cap_faces >= 2:
                    return True
        return False




    def _spine_snap_mesh_to_curve(self, bm, ch):
        """Move each parametric slice so its centroid sits on the live blue curve.

        Curve is already straight/on-axis; this only shifts the mesh onto that curve.
        Relative layout inside a slice (including filled caps) is preserved.
        """
        bind = list(ch.get('bind') or [])
        bez = ch.get('bez') or []
        if not bind or len(bez) < 2:
            return 0
        bm.verts.ensure_lookup_table()

        # Group bound verts by quantized t
        buckets = {}  # q -> list of (vidx, t)
        n_q = 32
        for item in bind:
            try:
                vidx = int(item[0])
                t = float(item[1])
            except Exception:
                continue
            if vidx >= len(bm.verts):
                continue
            t = max(0.0, min(1.0, t))
            q = int(round(t * (n_q - 1)))
            buckets.setdefault(q, []).append((vidx, t))

        moved = 0
        for q, group in buckets.items():
            if len(group) < 1:
                continue
            t_mean = sum(t for _, t in group) / float(len(group))
            try:
                c = eval_bezier_points(bez, t_mean)
                T = bezier_chain_tangent(bez, t_mean)
            except Exception:
                continue
            if T.length < 1e-12:
                T = Vector((0, 0, 1))
            else:
                T = T.normalized()

            cen = Vector((0, 0, 0))
            for vidx, _ in group:
                cen += bm.verts[vidx].co
            cen /= float(len(group))

            delta = cen - c
            # only lateral (perpendicular to curve tangent)
            delta = delta - T * delta.dot(T)
            if delta.length < 1e-10:
                continue
            for vidx, _ in group:
                bm.verts[vidx].co = bm.verts[vidx].co - delta
                if hasattr(self, 'all_rest') and isinstance(self.all_rest, dict):
                    self.all_rest[vidx] = bm.verts[vidx].co.copy()
                moved += 1
        return moved

    def _spine_weight_mask(self, obj, bm, ch):
        """Return active Spine vertex-group weights for the current chain.

        Geometry cleanup tools in Spine are masked by the active chain's actual
        painted weight: zero-weight vertices are never moved by W/E, while
        partial weights blend the requested correction proportionally.
        """
        weights = {}
        try:
            vg_name = ch.get('vg_name') or ''
            vg = obj.vertex_groups.get(vg_name) if vg_name else None
            if vg is None:
                return weights
            dl = bm.verts.layers.deform.verify()
            gi = vg.index
            for v in bm.verts:
                try:
                    w = float(v[dl].get(gi, 0.0))
                except Exception:
                    w = 0.0
                weights[v.index] = max(0.0, min(1.0, w))
        except Exception:
            return {}
        return weights

    def _bh_safe_dot(self, a, b):
        """Safe Vector dot used by Spine geometry cleanup.

        Blender can occasionally hand the modal code a stale/non-Vector value
        after edit-mode topology changes. Normalize both operands here so a
        transient bad value cannot raise ``Vector.dot(other)`` and abort Spine.
        """
        try:
            if not isinstance(a, Vector):
                a = Vector(a)
            if not isinstance(b, Vector):
                b = Vector(b)
            return a.dot(b)
        except Exception:
            return 0.0

    def _spine_uniformize_thickness(self, context, active_only=True):
        """Uniformize tube thickness using the average transverse-ring radius.

        E intentionally changes thickness, unlike W/Q.  Each
        transverse ring keeps its own center, angular layout, and cross-section
        shape; only its in-plane radial scale is adjusted so every ring has the
        same mean radius.  The target radius is the average of the current mean
        radii across all detected rings, so no arbitrary thickness is imposed.
        """
        if getattr(self, 'tool_mode', None) != 'SPINE_DEFORM':
            return False

        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return False
        cis = [int(getattr(self, 'active_chain', 0) or 0)] if active_only else list(range(len(chains)))

        try:
            obj = context.object
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
        except Exception:
            return False

        import math
        ring_sets = []
        all_ring_means = []

        # First pass: detect transverse rings and measure their current mean
        # radial thickness.  We deliberately do not move anything yet so the
        # global average is based on the original state.
        for ci in cis:
            if ci < 0 or ci >= len(chains):
                continue
            ch = chains[ci]
            bez = ch.get('bez') or []
            bind = list(ch.get('bind') or [])
            if len(bez) < 2 or len(bind) < 6:
                continue

            p0 = bez[0]['co'].copy()
            p1 = bez[-1]['co'].copy()
            axis = p1 - p0
            if axis.length < 1e-10:
                continue
            axis.normalize()

            bound = set()
            for item in bind:
                try:
                    vi = int(item[0])
                except Exception:
                    continue
                if 0 <= vi < len(bm.verts):
                    bound.add(vi)
            if len(bound) < 6:
                continue

            weight_map = self._spine_weight_mask(obj, bm, ch)
            if not weight_map or max((weight_map.get(vi, 0.0) for vi in bound), default=0.0) <= 1e-8:
                continue

            ring_adj = {vi: [] for vi in bound}
            for e in bm.edges:
                a, b = e.verts[0].index, e.verts[1].index
                if a not in bound or b not in bound:
                    continue
                d = bm.verts[b].co - bm.verts[a].co
                if d.length < 1e-10:
                    continue
                if abs(self._bh_safe_dot(d.normalized(), axis)) < 0.52:
                    ring_adj[a].append(b)
                    ring_adj[b].append(a)

            for vi, nbs in list(ring_adj.items()):
                uniq = list(dict.fromkeys(nbs))
                if len(uniq) > 2:
                    scored = []
                    for nj in uniq:
                        d = bm.verts[nj].co - bm.verts[vi].co
                        if d.length > 1e-10:
                            scored.append((1.0 - abs(self._bh_safe_dot(d.normalized(), axis)), nj))
                    scored.sort(reverse=True)
                    uniq = [nj for _, nj in scored[:2]]
                ring_adj[vi] = uniq

            visited = set()
            rings = []
            for start_vi in sorted(bound):
                if start_vi in visited or len(ring_adj.get(start_vi, ())) != 2:
                    continue
                ring = [start_vi]
                prev = start_vi
                cur = ring_adj[start_vi][0]
                ok = True
                for _ in range(len(bound) + 2):
                    if cur == start_vi:
                        break
                    if cur in visited or cur not in bound:
                        ok = False
                        break
                    ring.append(cur)
                    nxts = [n for n in ring_adj.get(cur, ()) if n != prev]
                    if len(nxts) != 1:
                        ok = False
                        break
                    prev, cur = cur, nxts[0]
                else:
                    ok = False
                if ok and cur == start_vi and len(ring) >= 4:
                    for vi in ring:
                        visited.add(vi)
                    rings.append(ring)

            if not rings:
                continue

            chain_rings = []
            for ring in rings:
                center = Vector((0.0, 0.0, 0.0))
                for vi in ring:
                    center += bm.verts[vi].co
                center /= float(len(ring))

                radii = []
                for vi in ring:
                    rel = bm.verts[vi].co - center
                    radial = rel - axis * self._bh_safe_dot(rel, axis)
                    r = radial.length
                    if r > 1e-10:
                        radii.append(r)
                if len(radii) < 4:
                    continue
                mean_r = sum(radii) / float(len(radii))
                if mean_r <= 1e-10:
                    continue
                chain_rings.append((ring, center, mean_r, axis, weight_map))
                all_ring_means.append(mean_r)

            if chain_rings:
                ring_sets.append(chain_rings)

        if not all_ring_means:
            self.report({'INFO'}, "Uniform Thickness: no transverse rings found")
            return False

        # The requested "average" thickness: one global target radius for all
        # detected rings.  Median-like robustness is deliberately NOT used;
        # this is the literal arithmetic average requested by the user.
        target_r = sum(all_ring_means) / float(len(all_ring_means))
        total_changed = 0
        total_rings = 0

        for chain_rings in ring_sets:
            for ring, center, mean_r, axis, weight_map in chain_rings:
                scale = target_r / mean_r
                if abs(scale - 1.0) < 1e-8:
                    total_rings += 1
                    continue
                for vi in ring:
                    vtx = bm.verts[vi]
                    rel = vtx.co - center
                    axial = axis * self._bh_safe_dot(rel, axis)
                    radial = rel - axial
                    if radial.length < 1e-10:
                        continue
                    w = float(weight_map.get(vi, 0.0))
                    if w <= 1e-8:
                        continue
                    target_co = center + axial + radial * scale
                    new_co = vtx.co.lerp(target_co, w)
                    if (new_co - vtx.co).length_squared > 1e-14:
                        vtx.co = new_co
                        total_changed += 1
                total_rings += 1

        if total_changed:
            try:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            except Exception:
                pass
            try:
                for ci in cis:
                    if 0 <= ci < len(chains):
                        self._spine_rebind_chain_from_mesh(bm, chains[ci])
            except Exception:
                pass
            try:
                self._spine_load_active_chain()
            except Exception:
                pass
            try:
                self._spine_recalc_normals(context)
            except Exception:
                try:
                    bm.normal_update()
                except Exception:
                    pass
            try:
                context.area.tag_redraw()
            except Exception:
                pass

        self.report({'INFO'}, f"Thickness Uniformized: {total_rings} rings (average radius)")
        return True

    def _spine_align_rings_to_axis_base(self, context, active_only=True):
        """Align transverse tube rings perpendicular to the active blue spine axis.

        This is intentionally NOT a circularize/scale operation: every ring vertex
        keeps its perpendicular (radial) offset and only its axial/shear component
        is removed so the ring lies in one plane normal to the blue line.
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None or self.tool_mode != 'SPINE_DEFORM':
            return False
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return False
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        cis = [int(getattr(self, 'active_chain', 0) or 0)] if active_only else list(range(len(chains)))
        changed = 0
        rings_done = 0

        for ci in cis:
            if ci < 0 or ci >= len(chains):
                continue
            ch = chains[ci]
            bind = list(ch.get('bind') or [])
            bez = ch.get('bez') or []
            if len(bind) < 6 or len(bez) < 2:
                continue

            # The live blue spine is the exact reference axis.
            p0 = bez[0]['co'].copy()
            p1 = bez[-1]['co'].copy()
            axis = p1 - p0
            if axis.length < 1e-10:
                axis = bez[-1]['co'] - bez[0]['co']
            if axis.length < 1e-10:
                continue
            axis.normalize()

            bound = set()
            for item in bind:
                try:
                    vi = int(item[0])
                except Exception:
                    continue
                if 0 <= vi < len(bm.verts):
                    bound.add(vi)
            if len(bound) < 6:
                continue

            # Build transverse adjacency. Edges mostly perpendicular to the
            # spine belong to rings; edges mostly parallel to it are longitudinal.
            ring_adj = {vi: [] for vi in bound}
            for e in bm.edges:
                a, b = e.verts[0].index, e.verts[1].index
                if a not in bound or b not in bound:
                    continue
                d = bm.verts[b].co - bm.verts[a].co
                if d.length < 1e-10:
                    continue
                along = abs(d.normalized().dot(axis))
                if along < 0.52:
                    ring_adj[a].append(b)
                    ring_adj[b].append(a)

            # Keep the two strongest transverse neighbors. This makes the
            # detector robust against diagonal/extra topology edges.
            for vi, nbs in list(ring_adj.items()):
                uniq = list(dict.fromkeys(nbs))
                if len(uniq) > 2:
                    scored = []
                    for nj in uniq:
                        d = bm.verts[nj].co - bm.verts[vi].co
                        if d.length > 1e-10:
                            scored.append((1.0 - abs(d.normalized().dot(axis)), nj))
                    scored.sort(reverse=True)
                    uniq = [nj for _, nj in scored[:2]]
                ring_adj[vi] = uniq

            # Extract clean closed transverse loops.
            visited = set()
            rings = []
            max_steps = len(bound) + 2
            for start in bound:
                if start in visited or len(ring_adj.get(start, ())) != 2:
                    continue
                ring = [start]
                prev = start
                cur = ring_adj[start][0]
                ok = True
                for _ in range(max_steps):
                    if cur == start:
                        break
                    if cur in visited or cur not in bound:
                        ok = False
                        break
                    ring.append(cur)
                    nbs = ring_adj.get(cur, ())
                    nxt = [n for n in nbs if n != prev]
                    if len(nxt) != 1:
                        ok = False
                        break
                    prev, cur = cur, nxt[0]
                else:
                    ok = False
                if ok and cur == start and len(ring) >= 4:
                    for vi in ring:
                        visited.add(vi)
                    rings.append(ring)

            if not rings:
                continue

            # Sort rings by their center position along the blue line.
            scored_rings = []
            for ring in rings:
                c = Vector((0.0, 0.0, 0.0))
                for vi in ring:
                    c += bm.verts[vi].co
                c /= float(len(ring))
                scored_rings.append((c.dot(axis), ring, c))
            scored_rings.sort(key=lambda x: x[0])

            # Pure planar/shear correction: preserve every vertex's radial
            # offset exactly and remove only its axial deviation from the ring
            # centroid plane. Thus tube thickness cannot shrink/grow.
            for s, ring, center in scored_rings:
                target_s = (center - p0).dot(axis)
                for vi in ring:
                    v = bm.verts[vi]
                    d = v.co - center
                    axial_error = d.dot(axis)
                    if abs(axial_error) > 1e-10:
                        v.co = v.co - axis * axial_error
                        changed += 1
                rings_done += 1

            # Do NOT full-rebind the current deformed mesh here.  A full rebind
            # would bake the active Shrink/Inflate/Tilt into all_rest/bind.
            # Capture only the corrected base positions into the active chain's
            # existing rest/bind entries, then restore the attributes unchanged.
            for item in list(ch.get('bind') or []):
                try:
                    vi = int(item[0])
                except Exception:
                    continue
                if 0 <= vi < len(bm.verts):
                    co = bm.verts[vi].co.copy()
                    self.all_rest[vi] = co.copy()
                    if len(item) >= 4:
                        t = float(item[1])
                        on = eval_bezier_points(ch.get('rest_bez') or ch.get('bez') or [], t)
                        tan = bezier_chain_tangent(ch.get('rest_bez') or ch.get('bez') or [], t)
                        if tan.length > 1e-12:
                            tan.normalize()
                        item_list = list(item)
                        item_list[2] = (co - on).copy()
                        item_list[3] = tan.copy()
                        item_list[4] = float((co - on).length) if len(item_list) > 4 else 0.0
                        # Preserve tuple/list convention used by the existing bind.
                        ch['bind'][ch['bind'].index(item)] = tuple(item_list)

        if changed:
            try:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            except Exception:
                pass
            try:
                self._spine_recalc_normals(context)
            except Exception:
                try:
                    bm.normal_update()
                except Exception:
                    pass
            context.area.tag_redraw()
            self.report({'INFO'}, f"Edge Rings Aligned: {rings_done} rings")
            return True

        self.report({'INFO'}, "Edge Rings Aligned: no transverse rings found")
        return False

    def _spine_align_rings_to_axis(self, context, active_only=True):
        """E / Set-Flow style transverse-ring flow along the blue Spine.

        This is deliberately different from the old "rotate each detected ring"
        approach.  The ring topology is recovered from BIND parameter spacing:
        at a tube vertex the two edges with the smallest |dt| are the two
        circumferential edges.  That makes the ring detector independent of the
        current bend angle of the mesh.

        Once the real closed rings are found, E rebuilds every ring from its
        existing cross-section coordinates in a parallel-transport frame whose
        axis is the BLUE CURVE tangent at that ring's t.  Ring centers follow the
        blue curve, while the cross-section itself is rigidly preserved.  This
        gives the same kind of smooth flow result the user expects from Set Flow:
        the rings follow the curve progressively instead of being independently
        guessed from their current normals.

        Important invariants:
          * real mesh topology determines ring membership/order;
          * ring vertex count/order is preserved;
          * cross-section shape, radius and thickness are preserved;
          * no global-axis flattening;
          * first/last rings keep their current centers (caps are not pulled);
          * W is completely independent and is not changed here.
        """
        if getattr(self, 'tool_mode', None) != 'SPINE_DEFORM':
            return False
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return False
        try:
            obj = context.object
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
        except Exception:
            return False

        cis = [int(getattr(self, 'active_chain', 0) or 0)] if active_only else list(range(len(chains)))
        rings_done = 0
        verts_changed = 0

        for ci in cis:
            if ci < 0 or ci >= len(chains):
                continue
            ch = chains[ci]
            bez = ch.get('bez') or []
            bind = list(ch.get('bind') or [])
            if len(bez) < 2 or len(bind) < 8:
                continue

            # Stable vertex -> blue-curve parameter map from the persistent bind.
            tmap = {}
            bound = set()
            for item in bind:
                try:
                    vi = int(item[0])
                    tt = float(item[1])
                except Exception:
                    continue
                if 0 <= vi < len(bm.verts):
                    bound.add(vi)
                    tmap[vi] = max(0.0, min(1.0, tt))
            if len(bound) < 8:
                continue

            # -------------------------------------------------------------
            # 1) Recover the actual circumferential edges from topology.
            #    Bend angle is irrelevant: ring edges have nearly equal t at
            #    both ends, longitudinal edges do not.
            # -------------------------------------------------------------
            ring_adj = {vi: [] for vi in bound}
            for vi in bound:
                v = bm.verts[vi]
                scored = []
                for e in v.link_edges:
                    ov = e.other_vert(v)
                    oi = ov.index
                    if oi not in bound or oi not in tmap:
                        continue
                    dt = abs(tmap[oi] - tmap[vi])
                    # Wrap is not relevant for a 0..1 open spine.
                    scored.append((dt, oi, e))
                if len(scored) < 2:
                    continue
                scored.sort(key=lambda x: (x[0], x[1]))
                # The two smallest dt edges are the circumferential pair on a
                # normal tube.  Keep only genuinely local candidates; this also
                # rejects diagonal/accidental edges whose dt is comparable to a
                # longitudinal step.
                best_dt = scored[0][0]
                chosen = [x for x in scored if x[0] <= max(best_dt * 2.5, 1e-5)]
                if len(chosen) < 2:
                    chosen = scored[:2]
                chosen = chosen[:2]
                ring_adj[vi] = [x[1] for x in chosen]

            # Make the graph symmetric only when the reciprocal relation is
            # equally local. This prevents one-way accidental diagonals from
            # turning into fake rings.
            clean_adj = {vi: [] for vi in bound}
            for a, nbs in ring_adj.items():
                for b in nbs:
                    if a in ring_adj.get(b, ()):
                        clean_adj[a].append(b)
            ring_adj = {vi: list(dict.fromkeys(nbs)) for vi, nbs in clean_adj.items()}

            # -------------------------------------------------------------
            # 2) Extract closed cycles. A real transverse ring is a degree-2
            #    closed cycle and has a very small t spread.
            # -------------------------------------------------------------
            visited = set()
            rings = []
            for start_vi in sorted(bound):
                if start_vi in visited or len(ring_adj.get(start_vi, ())) != 2:
                    continue
                ring = [start_vi]
                prev = start_vi
                cur = ring_adj[start_vi][0]
                ok = True
                for _ in range(len(bound) + 2):
                    if cur == start_vi:
                        break
                    if cur in ring or cur not in bound or len(ring_adj.get(cur, ())) != 2:
                        ok = False
                        break
                    ring.append(cur)
                    nxt = ring_adj[cur][0] if ring_adj[cur][0] != prev else ring_adj[cur][1]
                    prev, cur = cur, nxt
                else:
                    ok = False
                if not ok or cur != start_vi or len(ring) < 6:
                    continue
                ts = [tmap[v] for v in ring]
                if max(ts) - min(ts) > 0.012:
                    continue
                for v in ring:
                    visited.add(v)
                rings.append(ring)

            if len(rings) < 3:
                continue

            # -------------------------------------------------------------
            # 3) Sort rings by their BIND t, not by current spatial position.
            # -------------------------------------------------------------
            ring_info = []
            for ring in rings:
                tt = sum(tmap[v] for v in ring) / float(len(ring))
                c = Vector((0.0, 0.0, 0.0))
                for vi in ring:
                    c += bm.verts[vi].co
                c /= float(len(ring))
                ring_info.append((tt, ring, c))
            ring_info.sort(key=lambda x: x[0])

            # Keep only the dominant regular ring family. If a malformed stray
            # cycle exists, the median ring size is the safest topology filter.
            sizes = sorted(len(r) for _, r, _ in ring_info)
            med_size = sizes[len(sizes) // 2]
            ring_info = [x for x in ring_info if abs(len(x[1]) - med_size) <= max(1, med_size // 5)]
            if len(ring_info) < 3:
                continue

            # -------------------------------------------------------------
            # 4) Establish longitudinal correspondence using REAL topology
            #    edges between consecutive rings. This is the same principle
            #    that made W robust: never invent a phase from world angles.
            # -------------------------------------------------------------
            ordered = []
            first_t, first_ring, first_center = ring_info[0]
            tan0 = bezier_chain_tangent(bez, first_t)
            if tan0.length < 1e-10:
                continue
            tan0.normalize()

            # Pick an initial radial axis from the first ring itself.
            u0 = None
            for vi in first_ring:
                r = bm.verts[vi].co - first_center
                r = r - tan0 * r.dot(tan0)
                if r.length > 1e-8:
                    u0 = r.normalized()
                    break
            if u0 is None:
                ref = Vector((0.0, 0.0, 1.0)) if abs(tan0.z) < 0.85 else Vector((0.0, 1.0, 0.0))
                u0 = tan0.cross(ref)
                if u0.length < 1e-8:
                    continue
                u0.normalize()
            v0 = tan0.cross(u0).normalized()

            # Stable angular order for the first ring.
            vals = []
            for vi in first_ring:
                d = bm.verts[vi].co - first_center
                vals.append((math.atan2(d.dot(v0), d.dot(u0)), vi))
            vals.sort(key=lambda x: x[0])
            ordered.append([vi for _, vi in vals])

            for ri in range(1, len(ring_info)):
                prev = ordered[-1]
                prev_set = set(prev)
                cur_ring = ring_info[ri][1]
                cur_set = set(cur_ring)
                topo = {vi: [] for vi in prev}
                for vi in prev:
                    for e in bm.verts[vi].link_edges:
                        oi = e.other_vert(bm.verts[vi]).index
                        if oi in cur_set:
                            topo[vi].append(oi)
                used = set()
                cur_order = []
                good = True
                for a in prev:
                    cands = [b for b in topo.get(a, ()) if b not in used]
                    if not cands:
                        good = False
                        break
                    if len(cands) > 1:
                        # Choose the candidate with the smallest |dt| and then
                        # the closest radial direction to the previous vertex.
                        scored = []
                        ra = bm.verts[a].co - ring_info[ri - 1][2]
                        for b in cands:
                            rb = bm.verts[b].co - ring_info[ri][2]
                            phase = ra.normalized().dot(rb.normalized()) if ra.length > 1e-8 and rb.length > 1e-8 else -1.0
                            scored.append((abs(tmap[b] - tmap[a]), -phase, b))
                        scored.sort()
                        b = scored[0][2]
                    else:
                        b = cands[0]
                    cur_order.append(b)
                    used.add(b)
                if not good or used != cur_set or len(cur_order) != len(prev):
                    # If topology is broken, do not touch this chain. A false
                    # ring correspondence is worse than leaving it unchanged.
                    ordered = []
                    break
                ordered.append(cur_order)
            if len(ordered) != len(ring_info):
                continue

            # -------------------------------------------------------------
            # 5) Set Flow-style reconstruction: centers follow the BLUE CURVE
            #    and the cross-section is transported smoothly from ring to
            #    ring. No independent arbitrary ring rotations.
            # -------------------------------------------------------------
            old_frames = []
            new_frames = []
            prev_tan = tan0
            prev_u = u0
            prev_center = first_center.copy()

            for ri, (tt, ring, center_now) in enumerate(ring_info):
                tan = bezier_chain_tangent(bez, tt)
                if tan.length < 1e-10:
                    tan = prev_tan.copy()
                else:
                    tan.normalize()

                if ri == 0:
                    u = u0.copy()
                else:
                    # Parallel transport: minimum rotation of the previous
                    # frame onto the new blue tangent. This is the crucial
                    # "follow the curve" part and avoids twisting the tube.
                    q = prev_tan.rotation_difference(tan)
                    u = q @ prev_u
                    u = u - tan * u.dot(tan)
                    if u.length < 1e-8:
                        u = prev_u - tan * prev_u.dot(tan)
                    if u.length < 1e-8:
                        ref = Vector((0.0, 0.0, 1.0)) if abs(tan.z) < 0.85 else Vector((0.0, 1.0, 0.0))
                        u = tan.cross(ref)
                    u.normalize()
                vv = tan.cross(u).normalized()

                # End rings stay centered exactly where they are. Interior
                # rings use the blue curve position, which gives the requested
                # smooth flow without pulling a cap/end section.
                if ri == 0 or ri == len(ring_info) - 1:
                    target_center = center_now.copy()
                else:
                    target_center = eval_bezier_points(bez, tt)

                old_frames.append((prev_tan.copy(), prev_u.copy(), prev_center.copy()))
                new_frames.append((tan.copy(), u.copy(), vv.copy(), target_center.copy()))
                prev_tan = tan
                prev_u = u
                prev_center = center_now.copy()

            # Reconstruct each ring from its existing cross-section coordinates.
            # This is rigid in the transported frame: no radius averaging,
            # circularization or uniform-thickness operation occurs.
            for ri, (tt, ring, old_center) in enumerate(ring_info):
                tan_old, u_old, _old_center_frame = old_frames[ri]
                v_old = tan_old.cross(u_old).normalized()
                tan_new, u_new, v_new, target_center = new_frames[ri]
                for li, vi in enumerate(ordered[ri]):
                    p = bm.verts[vi].co
                    rel = p - old_center
                    x = rel.dot(u_old)
                    y = rel.dot(v_old)
                    # Remove the old axial/shear component; the cross-section
                    # itself is preserved exactly in x/y.
                    target = target_center + u_new * x + v_new * y
                    if (target - p).length > 1e-10:
                        bm.verts[vi].co = target
                        verts_changed += 1
                rings_done += 1

            # Update only the active chain's bind offsets to the newly created
            # base geometry. The final Tilt/Shrink/Inflate layer is reapplied by
            # the central geometry wrapper after this method returns.
            rest_bez = ch.get('rest_bez') or ch.get('bez') or []
            if rest_bez:
                for item in list(ch.get('bind') or []):
                    try:
                        vi = int(item[0])
                    except Exception:
                        continue
                    if vi < 0 or vi >= len(bm.verts):
                        continue
                    tt = float(item[1])
                    on = eval_bezier_points(rest_bez, tt)
                    tan = bezier_chain_tangent(rest_bez, tt)
                    if tan.length > 1e-10:
                        tan.normalize()
                    co = bm.verts[vi].co.copy()
                    vals = list(item)
                    if len(vals) >= 3:
                        vals[2] = (co - on).copy()
                    if len(vals) >= 4:
                        vals[3] = tan.copy()
                    if len(vals) >= 5:
                        vals[4] = float((co - on).length)
                    idx = ch['bind'].index(item)
                    ch['bind'][idx] = tuple(vals)
                    self.all_rest[vi] = co.copy()

        if verts_changed:
            try:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            except Exception:
                pass
            try:
                self._spine_recalc_normals(context)
            except Exception:
                try:
                    bm.normal_update()
                except Exception:
                    pass
            try:
                context.area.tag_redraw()
            except Exception:
                pass
            self.report({'INFO'}, f"Set Flow — {rings_done} edge rings aligned to blue Spine")
            return True

        self.report({'INFO'}, "Set Flow — no clean transverse ring flow found")
        return False

    def _spine_align_longitudinal_loops_to_axis(self, context, active_only=True):
        """Straighten longitudinal edge loops while preserving radial thickness.

        This operation is intentionally different from Shift+E:
          * Shift+E makes each transverse ring perpendicular to the blue spine.
          * Shift+W makes every longitudinal loop run straight and parallel to
            the blue spine.

        The key constraint is that Shift+W must NOT make the tube uniformly
        thick.  For every vertex we preserve its exact distance from the blue
        spine axis.  Only the *angular phase* around that axis is changed.
        Consequently a vertex can slide around its own transverse ring, but it
        cannot move inward/outward relative to the spine.

        After the transverse rings have been detected and topologically
        ordered, each longitudinal loop is identified by its vertex index in
        those ordered rings.  We calculate one stable angular position for
        that loop from all of its rings, then place every vertex of the loop at
        that angle while keeping its original axial coordinate and radial
        distance.  Thus every longitudinal loop becomes a straight line
        parallel to the blue spine without averaging or rescaling thickness.
        """
        if getattr(self, 'tool_mode', None) != 'SPINE_DEFORM':
            return False

        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return False
        cis = [int(getattr(self, 'active_chain', 0) or 0)] if active_only else list(range(len(chains)))

        try:
            obj = context.object
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
        except Exception:
            return False

        import math
        total_changed = 0
        total_loops = 0
        total_rings = 0

        for ci in cis:
            if ci < 0 or ci >= len(chains):
                continue
            ch = chains[ci]
            bez = ch.get('bez') or []
            bind = list(ch.get('bind') or [])
            if len(bez) < 2 or len(bind) < 6:
                continue

            p0 = bez[0]['co'].copy()
            p1 = bez[-1]['co'].copy()
            axis = p1 - p0
            if axis.length < 1e-10:
                continue
            axis.normalize()

            bound = set()
            for item in bind:
                try:
                    vi = int(item[0])
                except Exception:
                    continue
                if 0 <= vi < len(bm.verts):
                    bound.add(vi)
            if len(bound) < 6:
                continue

            weight_map = self._spine_weight_mask(obj, bm, ch)
            if not weight_map or max((weight_map.get(vi, 0.0) for vi in bound), default=0.0) <= 1e-8:
                continue

            # Build the same transverse topology used by Shift+E.  Edges
            # nearly perpendicular to the blue axis belong to a ring.
            ring_adj = {vi: [] for vi in bound}
            for e in bm.edges:
                a, b = e.verts[0].index, e.verts[1].index
                if a not in bound or b not in bound:
                    continue
                d = bm.verts[b].co - bm.verts[a].co
                if d.length < 1e-10:
                    continue
                along = abs(self._bh_safe_dot(d.normalized(), axis))
                if along < 0.52:
                    ring_adj[a].append(b)
                    ring_adj[b].append(a)

            # Keep the two strongest transverse neighbours.  This removes
            # accidental diagonals/branching edges while preserving the ring.
            for vi, nbs in list(ring_adj.items()):
                uniq = list(dict.fromkeys(nbs))
                if len(uniq) > 2:
                    scored = []
                    for nj in uniq:
                        d = bm.verts[nj].co - bm.verts[vi].co
                        if d.length > 1e-10:
                            scored.append((1.0 - abs(self._bh_safe_dot(d.normalized(), axis)), nj))
                    scored.sort(reverse=True)
                    uniq = [nj for _, nj in scored[:2]]
                ring_adj[vi] = uniq

            visited = set()
            rings = []
            for start_vi in sorted(bound):
                if start_vi in visited or len(ring_adj.get(start_vi, ())) != 2:
                    continue
                ring = [start_vi]
                prev = start_vi
                cur = ring_adj[start_vi][0]
                ok = True
                for _ in range(len(bound) + 2):
                    if cur == start_vi:
                        break
                    if cur in visited or cur not in bound:
                        ok = False
                        break
                    ring.append(cur)
                    nxts = [n for n in ring_adj.get(cur, ()) if n != prev]
                    if len(nxts) != 1:
                        ok = False
                        break
                    prev, cur = cur, nxts[0]
                else:
                    ok = False
                if ok and cur == start_vi and len(ring) >= 4:
                    for vi in ring:
                        visited.add(vi)
                    rings.append(ring)

            if len(rings) < 2:
                self.report({'INFO'}, "Longitudinal Loops: not enough transverse rings")
                continue

            # Stable perpendicular basis around the blue axis.
            ref = Vector((0.0, 0.0, 1.0))
            if abs(self._bh_safe_dot(axis, ref)) > 0.85:
                ref = Vector((0.0, 1.0, 0.0))
            u = axis.cross(ref)
            if u.length < 1e-10:
                ref = Vector((1.0, 0.0, 0.0))
                u = axis.cross(ref)
            u.normalize()
            vaxis = axis.cross(u)
            vaxis.normalize()

            # Sort rings along the actual blue spine axis.
            ring_info = []
            for ring in rings:
                center = Vector((0.0, 0.0, 0.0))
                for vi in ring:
                    center += bm.verts[vi].co
                center /= float(len(ring))
                ring_info.append((self._bh_safe_dot(center - p0, axis), ring, center))
            ring_info.sort(key=lambda x: x[0])

            # Angularly order the first ring only to establish a stable
            # starting phase.  From the SECOND ring onward, follow the REAL
            # longitudinal mesh edges between the two rings.  This is the
            # important part: W must track the actual edge loop topology, not
            # invent a correspondence from angular proximity when a ring is
            # rotated.
            ordered = []
            first_center = ring_info[0][2]
            first_vals = []
            for vi in ring_info[0][1]:
                d = bm.verts[vi].co - first_center
                first_vals.append((math.atan2(self._bh_safe_dot(d, vaxis), self._bh_safe_dot(d, u)), vi))
            first_vals.sort(key=lambda x: x[0])
            ordered.append([vi for _, vi in first_vals])

            for ri in range(1, len(ring_info)):
                prev = ordered[-1]
                prev_ring = set(ring_info[ri - 1][1])
                cur_ring = ring_info[ri][1]
                cur_set = set(cur_ring)
                center_prev = ring_info[ri - 1][2]
                center_cur = ring_info[ri][2]

                # Build actual topology links prev-ring -> current-ring.
                # A proper tube normally gives one longitudinal edge per
                # vertex.  If there are accidental/diagonal alternatives,
                # choose the candidate whose edge is most aligned with the
                # local blue-spine direction.
                topo = {vi: [] for vi in prev}
                for e in bm.edges:
                    a = e.verts[0].index
                    b = e.verts[1].index
                    if a in prev_ring and b in cur_set:
                        topo.setdefault(a, []).append(b)
                    elif b in prev_ring and a in cur_set:
                        topo.setdefault(b, []).append(a)

                # Prefer a clean one-to-one topological correspondence.
                topo_map = {}
                used_cur = set()
                clean = True
                for a in prev:
                    cands = [b for b in topo.get(a, []) if b not in used_cur]
                    if not cands:
                        clean = False
                        break
                    if len(cands) == 1:
                        b = cands[0]
                    else:
                        mid = (bm.verts[a].co + center_prev)
                        tangent = center_cur - center_prev
                        if tangent.length > 1e-10:
                            tangent.normalize()
                        scored = []
                        for b0 in cands:
                            ed = bm.verts[b0].co - bm.verts[a].co
                            if ed.length < 1e-10:
                                continue
                            edn = ed.normalized()
                            # Local spine alignment first, then continuity of
                            # the radial phase as a tie-breaker.
                            align = abs(self._bh_safe_dot(edn, tangent))
                            ra = bm.verts[a].co - center_prev
                            rb = bm.verts[b0].co - center_cur
                            phase = 0.0
                            if ra.length > 1e-10 and rb.length > 1e-10:
                                phase = self._bh_safe_dot(ra.normalized(), rb.normalized())
                            scored.append((align, phase, b0))
                        if not scored:
                            clean = False
                            break
                        scored.sort(reverse=True)
                        b = scored[0][2]
                    topo_map[a] = b
                    used_cur.add(b)

                if clean and len(topo_map) == len(prev) and used_cur == cur_set:
                    ordered.append([topo_map[a] for a in prev])
                    continue

                # Fallback only when topology is incomplete (e.g. open/broken
                # strip).  Keep the old angular matching as a safe fallback;
                # normal manifold tube loops should use the topology branch.
                cur_vals = []
                for vi in cur_ring:
                    d = bm.verts[vi].co - center_cur
                    cur_vals.append((math.atan2(self._bh_safe_dot(d, vaxis), self._bh_safe_dot(d, u)), vi))
                cur_vals.sort(key=lambda x: x[0])
                cur = [vi for _, vi in cur_vals]

                if len(prev) != len(cur):
                    ordered.append(cur)
                    continue

                n = len(cur)
                best_cost = None
                best_seq = None
                for reverse in (False, True):
                    seq0 = list(reversed(cur)) if reverse else list(cur)
                    for shift in range(n):
                        seq = seq0[shift:] + seq0[:shift]
                        cost = 0.0
                        for a, b in zip(prev, seq):
                            da = bm.verts[a].co - center_prev
                            db = bm.verts[b].co - center_cur
                            cost += (da - db).length_squared
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_seq = seq
                ordered.append(best_seq if best_seq else cur)

            nloops = min(len(r) for r in ordered)
            if nloops < 4:
                continue

            # Compute one stable target angle for EACH longitudinal loop.
            # Circular averaging avoids the old "rotate one whole ring"
            # behaviour.  Every loop gets its own phase, so all loops are
            # straightened simultaneously.
            target_angles = []
            for li in range(nloops):
                sx = 0.0
                sy = 0.0
                for ri, (_, _, center) in enumerate(ring_info):
                    vi = ordered[ri][li]
                    rel = bm.verts[vi].co - center
                    r = math.hypot(self._bh_safe_dot(rel, u), self._bh_safe_dot(rel, vaxis))
                    if r < 1e-10:
                        continue
                    a = math.atan2(self._bh_safe_dot(rel, vaxis), self._bh_safe_dot(rel, u))
                    # Weight by radius so very small inner rings do not
                    # dominate the phase of a large outer section.
                    sx += math.cos(a) * r
                    sy += math.sin(a) * r
                if abs(sx) < 1e-14 and abs(sy) < 1e-14:
                    # Fallback to the first valid vertex of this loop.
                    rel = bm.verts[ordered[0][li]].co - ring_info[0][2]
                    target_angles.append(math.atan2(self._bh_safe_dot(rel, vaxis), self._bh_safe_dot(rel, u)))
                else:
                    target_angles.append(math.atan2(sy, sx))

            # Move every vertex only by rotating its radial vector around the
            # blue axis.  Its axial coordinate AND its exact radial distance
            # are preserved.  This is the critical thickness-preserving step.
            for ri, (_, ring, center) in enumerate(ring_info):
                for li in range(nloops):
                    vi = ordered[ri][li]
                    old = bm.verts[vi].co.copy()
                    axial = self._bh_safe_dot(old - p0, axis)
                    rel = old - center
                    radial_u = self._bh_safe_dot(rel, u)
                    radial_v = self._bh_safe_dot(rel, vaxis)
                    radius = math.hypot(radial_u, radial_v)
                    if radius < 1e-10:
                        continue

                    a = target_angles[li]
                    target_rel = u * (math.cos(a) * radius) + vaxis * (math.sin(a) * radius)
                    target = center + target_rel

                    # Keep the ring centroid on the blue axis.  If E was not
                    # run immediately before W, this removes only the centroid
                    # offset; it does not touch the radius of the vertex.
                    center_axial = self._bh_safe_dot(center - p0, axis)
                    target = p0 + axis * center_axial + target_rel

                    w = float(weight_map.get(vi, 0.0))
                    if w <= 1e-8:
                        continue
                    blended = old.lerp(target, w)
                    if (blended - old).length_squared > 1e-14:
                        bm.verts[vi].co = blended
                        total_changed += 1

                total_rings += 1
            total_loops += nloops

        if total_changed:
            try:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            except Exception:
                pass
            try:
                for ci in cis:
                    if 0 <= ci < len(chains):
                        self._spine_rebind_chain_from_mesh(bm, chains[ci])
            except Exception:
                pass
            try:
                self._spine_load_active_chain()
            except Exception:
                pass
            try:
                self._spine_recalc_normals(context)
            except Exception:
                try:
                    bm.normal_update()
                except Exception:
                    pass
            try:
                context.area.tag_redraw()
            except Exception:
                pass

        if total_loops:
            self.report({'INFO'}, f"Longitudinal Loops Aligned: {total_loops} loops / {total_rings} rings (radial thickness preserved)")
            return True
        self.report({'INFO'}, "Longitudinal Loops: no usable rings found")
        return False

    def _spine_straighten_tube(self, context, circularize=True, even_spacing=True, active_only=True, axis='FREE'):
        """Straighten the Spine controller curve only.

        This operation intentionally does NOT modify the mesh, rest state, bind,
        Tilt, Shrink/Inflate, or vertex-group weights.  Shift+L is a controller
        editing operation: it changes only the blue Bezier spine.
        """
        if getattr(self, 'tool_mode', '') != 'SPINE_DEFORM':
            self.report({'INFO'}, "Straighten Tube: use in Spine Deform")
            return False

        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            self.report({'WARNING'}, "Straighten Tube: no chains")
            return False

        try:
            self._spine_store_active_chain()
        except Exception:
            pass

        if active_only:
            indices = [int(getattr(self, 'active_chain', 0) or 0)]
            if indices[0] < 0 or indices[0] >= len(chains):
                self.report({'WARNING'}, "Straighten Tube: no active chain")
                return False
        else:
            indices = list(range(len(chains)))

        # Snapshot BEFORE changing the curve so Ctrl+Z restores the exact
        # previous controller positions/handles.  No mesh operation follows.
        self._spine_push_undo(context)
        done = 0

        old_active = getattr(self, 'active_chain', 0)
        old_bez = getattr(self, 'bez', None)
        old_rest_bez = getattr(self, 'rest_bez', None)
        old_modes = getattr(self, 'point_modes', None)
        old_handle_params = getattr(self, 'handle_params', None)

        try:
            for ci in indices:
                ch = chains[ci]
                current = ch.get('bez') or []
                if len(current) < 2:
                    continue

                self.active_chain = ci
                self.bez = ch.get('bez')
                self.rest_bez = ch.get('rest_bez')
                self.point_modes = list(ch.get('modes') or [])
                self.handle_params = ch.get('handle_params') or []

                # Build the straight controller curve from the CURRENT blue
                # curve.  This is the only geometry/state mutation performed.
                straight, axis_dir, p0, arc = self._spine_make_straight_bez(
                    copy_bezier_points(current), axis=axis
                )
                if not straight or len(straight) < 2:
                    continue

                ch['bez'] = straight
                self.bez = ch['bez']

                # Preserve the existing handle modes.  Rebuild AUTO tips only
                # for the controller representation; never call _spine_apply.
                try:
                    self.point_modes = list(ch.get('modes') or [])
                    self.rebuild_auto_handles()
                    ch['bez'] = copy_bezier_points(self.bez)
                    self.bez = ch['bez']
                except Exception:
                    pass

                # Keep the persistent chain data synchronized, but deliberately
                # leave rest_bez, bind, all_rest, weights and final attributes
                # untouched.
                ch['modes'] = list(getattr(self, 'point_modes', None) or ch.get('modes') or [])
                done += 1
        finally:
            self.active_chain = old_active
            if old_bez is not None:
                self.bez = old_bez
            if old_rest_bez is not None:
                self.rest_bez = old_rest_bez
            if old_modes is not None:
                self.point_modes = old_modes
            if old_handle_params is not None:
                self.handle_params = old_handle_params

        self.spine_chains = chains
        try:
            self._spine_load_active_chain()
        except Exception:
            pass

        # IMPORTANT: evaluate the mesh exactly like an interactive controller
        # drag.  A normal non-drag _spine_apply() can reconcile bind/weight data
        # before evaluating the changed curve, which leaves the mesh visually
        # stale until the next mouse drag.  We do NOT modify any bind/rest/weight
        # data here; dragging=True only selects the live deformation evaluation
        # path, then the original flag is restored immediately.
        _old_dragging = bool(getattr(self, 'dragging', False))
        try:
            self.dragging = True
            self._spine_apply(context, auto_soft=False)
        finally:
            self.dragging = _old_dragging

        try:
            obj, bm = self.get_obj_bm(context)
            if obj is not None:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        except Exception:
            pass
        try:
            self._spine_save_recall()
        except Exception:
            pass
        try:
            context.area.tag_redraw()
        except Exception:
            pass

        scope = "active" if active_only else f"{done} chain(s)"
        ax = str(axis or 'FREE').upper()
        self.report({'INFO'}, f"Straighten Tube ({scope}, {ax}) — controller curve only")
        return done > 0

    def _spine_set_as_initial(self, context):
        """Edit Place: set active chain's current pose as the Ctrl+R baseline.

        - rest_bez becomes current bez (zero-deform pose)
        - bind offsets recomputed from current mesh
        - bind-rest snapshot rewritten for this chain_id (overwrite mesh_rest)
        """
        if not getattr(self, '_spine_edit_place', False):
            self.report({'INFO'}, "Set as Initial: only in Edit Place")
            return False
        if getattr(self, '_spine_placing_new_chain', False):
            self.report({'INFO'}, "Set as Initial: finish/place chain first (need a bound chain)")
            return False
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            self.report({'WARNING'}, "Set as Initial: no chains")
            return False
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        ac = int(getattr(self, 'active_chain', 0) or 0)
        if not (0 <= ac < len(chains)):
            self.report({'WARNING'}, "Set as Initial: no active chain")
            return False
        ch = chains[ac]
        bez = ch.get('bez') or getattr(self, 'bez', None)
        if not bez or len(bez) < 2:
            self.report({'WARNING'}, "Set as Initial: need at least 2 controllers")
            return False

        self._spine_push_undo(context)
        self._spine_ensure_chain_ids()

        # Current pose becomes rest (Ctrl+R and further deform relative to this)
        ch['bez'] = bez if bez is ch.get('bez') else copy_bezier_points(bez)
        ch['rest_bez'] = copy_bezier_points(ch['bez'])
        if getattr(self, 'point_modes', None) and len(self.point_modes) >= 2:
            ch['modes'] = list(self.point_modes)
        if getattr(self, 'spine_tilt', None) is not None:
            ch['tilt'] = list(self.spine_tilt)
        if getattr(self, 'spine_radius', None) is not None:
            ch['radius'] = list(self.spine_radius)

        # The current controller Influence/Radius values also become the
        # persistent Alt+R reset baseline when Set as Initial State is used.
        # Keep this separate from the actual current radius so later edits do
        # not change the user's chosen initial/reset values.
        try:
            n_ctrl = len(ch.get('bez') or [])
            current_inf = ensure_point_influence(
                n_ctrl, ch.get('point_influence'),
                default=(ch.get('influence') or 0.1),
            )
            ch['point_influence'] = list(current_inf)
            ch['point_influence_default'] = list(current_inf)
        except Exception:
            pass

        obj, bm = self.get_obj_bm(context)
        if bm is not None:
            bm.verts.ensure_lookup_table()
            # Recompute offsets so current mesh is the zero-deform pose
            try:
                self._spine_rebind_chain_from_mesh(bm, ch)
            except Exception:
                try:
                    self._spine_rebind_chain_proximity(context, ch)
                except Exception:
                    pass
            # all_rest tracks current mesh for cancel/prop
            for item in (ch.get('bind') or []):
                vidx = int(item[0])
                if vidx < len(bm.verts):
                    self.all_rest[vidx] = bm.verts[vidx].co.copy()

        # Rewrite first-bind snapshot for this chain only
        cid = ch.get('chain_id')
        try:
            self._spine_merge_bind_rest(
                context,
                force_update_ids={cid} if cid else None,
                overwrite_mesh=True,
            )
        except Exception:
            pass

        # Session rest snapshots (used by some reset paths)
        try:
            self._spine_load_active_chain()
            if getattr(self, 'bez', None):
                self._spine_session_rest_bez = copy_bezier_points(self.bez)
                self._spine_session_rest_tilt = list(getattr(self, 'spine_tilt', []) or [])
                self._spine_session_rest_radius = list(getattr(self, 'spine_radius', []) or [])
                self._spine_session_rest_modes = list(getattr(self, 'point_modes', []) or [])
        except Exception:
            pass

        context.area.tag_redraw()
        self.report({'INFO'}, f"Initial state set for chain {ac + 1} (Ctrl+R uses this)")
        return True

    def _spine_duplicate_active_chain(self, context, mirror=False, axis='X', space='LOCAL', event=None):
        """Duplicate active chain. Shift+D duplicates + grabs; Ctrl+Shift+M
        duplicates + mirrors without entering Grab.
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None:
            return False
        self._spine_push_undo(context)
        tm = getattr(self, 'tool_mode', '')

        # --- Place / Edit Place ---
        if tm == 'SPINE_PLACE':
            pts = [p.copy() for p in (getattr(self, 'spine_points', None) or [])]
            if len(pts) < 2:
                self.report({'WARNING'}, "Need 2+ controllers to duplicate")
                return False
            mirror_fn = None
            if mirror:
                mirror_fn, axis, space = self._spine_mirror_fn(context, obj, axis, space)
                pts = [mirror_fn(p) for p in pts]
            # Archive current into pending list, start working on the copy
            if getattr(self, 'spine_chains_pts', None) is None:
                self.spine_chains_pts = []
            # Keep original as completed pending only when not edit of bound chains
            if not getattr(self, '_spine_edit_place', False) or getattr(self, '_spine_placing_new_chain', False):
                cur = [p.copy() for p in (self.spine_points or [])]
                if len(cur) >= 2:
                    self.spine_chains_pts.append(cur)
                    oids = list(getattr(self, '_spine_origin_ids', None) or list(range(len(cur))))
                    if getattr(self, '_spine_chains_origin_ids', None) is None:
                        self._spine_chains_origin_ids = []
                    self._spine_chains_origin_ids.append(list(oids))
            elif getattr(self, '_spine_edit_place', False) and getattr(self, 'spine_chains', None):
                # Edit place with bound chains: append a new deform chain from duplicated pts
                try:
                    self._spine_store_active_chain()
                except Exception:
                    pass
                chains = list(self.spine_chains)
                ai = int(getattr(self, 'active_chain', 0) or 0)
                src = chains[ai] if 0 <= ai < len(chains) else None
                if src is None:
                    return False
                ch_new = self._spine_copy_chain_dict(src, new_id=f"dup_{len(chains)}_{id(src) & 0xFFFF:x}")
                if mirror and mirror_fn:
                    ch_new['bez'] = self._spine_mirror_bez_points(ch_new.get('bez'), mirror_fn)
                    ch_new['rest_bez'] = self._spine_mirror_bez_points(
                        ch_new.get('rest_bez') or ch_new.get('bez'), mirror_fn,
                    )
                # Clear bind until rebind on Enter
                ch_new['bind'] = []
                chains.append(ch_new)
                self.spine_chains = chains
                self.active_chain = len(chains) - 1
                # Copy first-bind history so Ctrl+R works on the duplicate
                try:
                    self._spine_copy_bind_rest_history(
                        context, src, ch_new,
                        mirror_fn=mirror_fn if mirror else None,
                    )
                except Exception:
                    pass
                # rest_bez = initial state of source chain
                try:
                    data = _vdh_get_bind_rest(obj)
                    by_id = {
                        s.get('chain_id'): s
                        for s in (data.get('chains') or [])
                        if s.get('chain_id')
                    }
                    entry = by_id.get(ch_new.get('chain_id'))
                    if entry and entry.get('bez') and len(entry['bez']) >= 2:
                        ch_new['rest_bez'] = copy_bezier_points(entry['bez'])
                except Exception:
                    pass
                self._spine_load_active_chain()
                self.spine_points = [bp['co'].copy() for bp in (self.bez or [])]
                if mirror:
                    self.report({'INFO'}, f"Mirror-duplicate chain → Edit Place (no Grab)")
                else:
                    self.report({'INFO'}, f"Duplicate chain → Edit Place | move mouse")
                context.area.tag_redraw()
                # Shift+D behaves like Blender Duplicate + Grab.
                # Ctrl+Shift+M is Duplicate + Mirror only and must NOT Grab.
                if not mirror:
                    self._spine_start_grab_active_chain(context, event)
                return True
            self.spine_points = pts
            self._spine_origin_ids = [None] * len(pts)
            self._spine_placing_new_chain = True
            self._spine_last_add_idx = len(pts) - 1
            self.selected = {(i, 'co') for i in range(len(pts))}
            self.active_handle = 0
            self.report({'INFO'}, f"{'Mirror-duplicate' if mirror else 'Duplicate'} chain — move mouse, Enter to bind")
            context.area.tag_redraw()
            # Ctrl+Shift+M must not Grab; Shift+D still behaves like Blender
            # duplicate-and-grab.
            if not mirror:
                self._spine_start_grab_active_chain(context, event)
            return True

        # --- Deform ---
        chains = list(getattr(self, 'spine_chains', None) or [])
        ai = int(getattr(self, 'active_chain', 0) or 0)
        if not chains or not (0 <= ai < len(chains)):
            self.report({'WARNING'}, "No active chain")
            return False
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        src = chains[ai]
        ch_new = self._spine_copy_chain_dict(src, new_id=f"dup_{len(chains)}_{id(src) & 0xFFFF:x}")
        # Preserve handle modes exactly
        ch_new['modes'] = list(src.get('modes') or ['AUTO'] * len(ch_new.get('bez') or []))
        mirror_fn = None
        if mirror:
            mirror_fn, axis, space = self._spine_mirror_fn(context, obj, axis, space)
            ch_new['bez'] = self._spine_mirror_bez_points(ch_new.get('bez'), mirror_fn)
            ch_new['rest_bez'] = self._spine_mirror_bez_points(
                ch_new.get('rest_bez') or ch_new.get('bez'), mirror_fn,
            )
            # Modes stay as on source (AUTO/ALIGNED/FREE)
            ch_new['modes'] = list(src.get('modes') or ch_new['modes'])
        chains.append(ch_new)
        self.spine_chains = chains
        self.active_chain = len(chains) - 1
        # Inherit source first-bind history under the new chain_id
        try:
            self._spine_copy_bind_rest_history(
                context, src, ch_new, mirror_fn=mirror_fn,
            )
        except Exception:
            pass
        # rest_bez = initial state of source chain
        try:
            data = _vdh_get_bind_rest(obj)
            by_id = {
                s.get('chain_id'): s
                for s in (data.get('chains') or [])
                if s.get('chain_id')
            }
            entry = by_id.get(ch_new.get('chain_id'))
            if entry and entry.get('bez') and len(entry['bez']) >= 2:
                ch_new['rest_bez'] = copy_bezier_points(entry['bez'])
        except Exception:
            pass
        self._spine_rebind_chain_proximity(context, ch_new)
        try:
            self._spine_load_active_chain()
        except Exception:
            pass
        try:
            self._spine_apply(context, auto_soft=False)
        except Exception:
            pass
        try:
            # Refresh bind_verts only; history already copied above
            self._spine_merge_bind_rest(context)
        except Exception:
            pass
        label = f"Mirror-dup {axis} ({space})" if mirror else "Duplicate"
        self.report({'INFO'}, f"{label} chain | {len(ch_new.get('bind') or [])} verts  |  move mouse to place")
        context.area.tag_redraw()
        # Shift+D duplicates and immediately Grabs; Ctrl+Shift+M duplicates +
        # mirrors but must leave the new chain stationary.
        if not mirror:
            self._spine_start_grab_active_chain(context, event)
        return True

    def _spine_mirror_active_chain(self, context, axis='X', space='LOCAL', pivot=None, curve_only=True, duplicate=False, event=None):
        """Mirror active chain in place (default), or duplicate+mirror when duplicate=True."""
        try:
            if duplicate:
                return self._spine_duplicate_active_chain(
                    context, mirror=True, axis=axis, space=space, event=event,
                )
            return self._spine_mirror_active_chain_impl(context, axis=axis, space=space)
        except Exception as e:
            try:
                self.report({'ERROR'}, f"Mirror failed: {e}")
            except Exception:
                pass
            return False


    def _spine_recalc_normals(self, context, flip=False):
        """Recalculate face/vertex normals (like Shift+N).
        flip=True: reverse face winding first (needed after geometric mirror).
        Forces viewport refresh so shading updates without Object Mode toggle.
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        if flip:
            try:
                bmesh.ops.reverse_faces(bm, faces=bm.faces)
            except Exception:
                try:
                    for f in bm.faces:
                        f.normal_flip()
                except Exception:
                    pass
        try:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        except Exception:
            pass
        try:
            bm.normal_update()
        except Exception:
            try:
                for f in bm.faces:
                    f.normal_update()
                for v in bm.verts:
                    v.normal_update()
            except Exception:
                pass
        me = obj.data
        # loop_triangles=True forces a fuller edit-mesh push so the viewport
        # rebuilds shaded normals without requiring Object↔Edit toggle.
        try:
            bmesh.update_edit_mesh(me, loop_triangles=True, destructive=False)
        except Exception:
            try:
                bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
            except Exception:
                pass
        try:
            if hasattr(me, 'calc_normals'):
                me.calc_normals()
        except Exception:
            pass
        try:
            me.update()
        except Exception:
            pass
        # Depsgraph + all 3D views
        try:
            context.view_layer.update()
        except Exception:
            pass
        try:
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                        for region in area.regions:
                            region.tag_redraw()
        except Exception:
            pass

    def _spine_mirror_active_chain_impl(self, context, axis='X', space='LOCAL'):
        """In-place mirror of the active chain — keeps handle modes (AUTO/ALIGNED/FREE)."""

        obj, bm = self.get_obj_bm(context)
        if obj is None:
            self.report({'WARNING'}, "No mesh")
            return False
        mirror_fn, axis, space = self._spine_mirror_fn(context, obj, axis, space)
        self._spine_push_undo(context)
        tm = getattr(self, 'tool_mode', '')

        if tm == 'SPINE_PLACE':
            pts = [p.copy() for p in (getattr(self, 'spine_points', None) or [])]
            if len(pts) < 2:
                self.report({'WARNING'}, "Need 2+ controllers")
                return False
            self.spine_points = [mirror_fn(p) for p in pts]
            # Edit place with bez: mirror handles too, keep modes
            if getattr(self, '_spine_edit_place', False) and getattr(self, 'bez', None) and len(self.bez) >= 2:
                modes = list(getattr(self, 'point_modes', None) or ['AUTO'] * len(self.bez))
                self.bez = self._spine_mirror_bez_points(self.bez, mirror_fn)
                self.point_modes = modes
                try:
                    self._spine_store_active_chain()
                except Exception:
                    pass
            self.report({'INFO'}, f"Mirror {axis} ({space}) in place")
            context.area.tag_redraw()
            return True

        # Deform: pure geometric mirror — curve + bound verts together.
        # Avoids frame/tilt twist that comes from "mirror bez then apply vs old rest".
        chains = list(getattr(self, 'spine_chains', None) or [])
        ai = int(getattr(self, 'active_chain', 0) or 0)
        if not chains or not (0 <= ai < len(chains)):
            self.report({'WARNING'}, "No active chain")
            return False
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        ch = chains[ai]
        modes = list(ch.get('modes') or ['AUTO'] * len(ch.get('bez') or []))
        obj, bm = self.get_obj_bm(context)
        if bm is None:
            return False
        bm.verts.ensure_lookup_table()

        # 1) Mirror bound mesh verts (same reflection as the curve)
        mirrored_verts = set()
        for item in (ch.get('bind') or []):
            vidx = int(item[0])
            if vidx in mirrored_verts or vidx >= len(bm.verts):
                continue
            bm.verts[vidx].co = mirror_fn(bm.verts[vidx].co)
            mirrored_verts.add(vidx)

        # 2) Mirror curve (co + handle tips), keep modes
        ch['bez'] = self._spine_mirror_bez_points(ch.get('bez'), mirror_fn)
        ch['rest_bez'] = self._spine_mirror_bez_points(
            ch.get('rest_bez') or ch.get('bez'), mirror_fn,
        )
        ch['modes'] = modes
        n = len(ch.get('bez') or [])
        ch['tilt'] = [0.0] * n
        ch['radius'] = [1.0] * n

        # 3) Rebuild AUTO tips on mirrored curve (ALIGNED/FREE tips already mirrored)
        try:
            self._spine_load_active_chain()
            old_modes = list(getattr(self, 'point_modes', None) or modes)
            self.point_modes = old_modes
            self.rebuild_auto_handles()
            self.point_modes = old_modes
            ch['bez'] = copy_bezier_points(self.bez)
            ch['rest_bez'] = copy_bezier_points(self.bez)
            ch['modes'] = list(old_modes)
        except Exception:
            pass

        # 4) Bind offsets from mirrored mesh vs mirrored rest (no extra deform)
        try:
            self._spine_resync_chain_bind_from_mesh(bm, ch)
        except Exception:
            pass
        self.all_rest = {v.index: v.co.copy() for v in bm.verts}
        self.initial_all_rest = {k: v.copy() for k, v in self.all_rest.items()}
        try:
            self._spine_load_active_chain()
            self._spine_session_rest_bez = copy_bezier_points(self.bez)
            self._spine_session_rest_tilt = list(getattr(self, 'spine_tilt', []) or [])
            self._spine_session_rest_radius = list(getattr(self, 'spine_radius', []) or [])
            self._spine_session_rest_modes = list(getattr(self, 'point_modes', []) or [])
        except Exception:
            pass
        try:
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        except Exception:
            pass
        try:
            # Reflection reverses winding → flip faces then recalc (outward normals)
            self._spine_recalc_normals(context, flip=True)
            self._face_flip_parity = (int(getattr(self, '_face_flip_parity', 0) or 0) + 1) % 2
        except Exception:
            pass
        self.report({'INFO'}, f"Mirror {axis} ({space}) — geometric ({len(mirrored_verts)} verts)")
        context.area.tag_redraw()
        return True

    def _draw_influence_overlay(self, context, obj):
        """Fast, robust Spine influence overlay.

        The overlay is deliberately rendered as a small number of batched 3D
        triangle fans.  We do NOT use GPU POINTS or per-marker draw calls:
        POINTS point-size support differs between Blender/GPU combinations and
        was the reason the markers could disappear.  Marker size is screen-space
        and is completely independent of controller display_scale.
        """
        chains = getattr(self, 'spine_chains', None) or []
        if not chains:
            return

        # A controller is considered selected when its body OR either handle is
        # selected.  This is Spine-only and matches the intended overlay UX.
        selected_ctrls = set()
        for item in (getattr(self, 'selected', None) or set()):
            try:
                if len(item) == 3:
                    ci, idx, part = item
                    if part in {'co', 'hl', 'hr'}:
                        selected_ctrls.add((int(ci), int(idx)))
                elif len(item) == 2:
                    idx, part = item
                    if part in {'co', 'hl', 'hr'}:
                        ci = int(getattr(self, 'active_chain', 0) or 0)
                        selected_ctrls.add((ci, int(idx)))
            except Exception:
                continue
        if not selected_ctrls:
            return

        try:
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
        except Exception:
            return
        try:
            dl = bm.verts.layers.deform.verify()
        except Exception:
            dl = None

        # Cache WEIGHTS only.  Positions are always taken from the live Edit
        # BMesh so markers follow the mesh during Spine dragging.
        cache_key = tuple(sorted(selected_ctrls))
        cached = getattr(self, '_influence_overlay_cache', None)
        if cached is None or cached.get('key') != cache_key:
            combined = {}
            for ci, ctrl_idx in selected_ctrls:
                if not (0 <= ci < len(chains)):
                    continue
                ch = chains[ci]
                bind = ch.get('bind') or []
                bez = ch.get('bez') or []
                n_ctrl = len(bez)
                if not bind or not (0 <= ctrl_idx < n_ctrl):
                    continue

                legacy = float(ch.get('influence') or getattr(self, 'spine_influence', 0.1) or 0.1)
                pinf = ensure_point_influence(n_ctrl, ch.get('point_influence'), default=legacy)
                pfo = ensure_point_inf_falloff(
                    n_ctrl, ch.get('point_inf_falloff'),
                    default=ch.get('inf_falloff') or 'CONSTANT',
                )
                hparams = list(ch.get('handle_params') or [])
                if len(hparams) != n_ctrl:
                    hparams = [i / max(1, n_ctrl - 1) for i in range(n_ctrl)]
                    if hparams:
                        hparams[0], hparams[-1] = 0.0, 1.0

                vg_name = ch.get('vg_name') or ''
                vg = obj.vertex_groups.get(vg_name) if vg_name else None
                gi = vg.index if vg is not None else None

                for item in bind:
                    try:
                        vidx = int(item[0])
                    except Exception:
                        continue
                    if vidx < 0 or vidx >= len(bm.verts):
                        continue

                    gw = 1.0
                    if dl is not None and gi is not None:
                        try:
                            dvert = bm.verts[vidx][dl]
                            gw = float(dvert[gi]) if gi in dvert else 0.0
                        except Exception:
                            gw = 0.0
                        gw = max(0.0, min(1.0, gw))
                    if gw <= 1e-8:
                        continue

                    t = float(item[1])
                    cw = deform_weights(t, hparams)
                    blend = cw[ctrl_idx] if ctrl_idx < len(cw) else 0.0
                    if blend <= 1e-8:
                        continue

                    dist = self._spine_item_radial_dist(item)
                    radius = pinf[ctrl_idx] if ctrl_idx < len(pinf) else legacy
                    falloff = pfo[ctrl_idx] if ctrl_idx < len(pfo) else 'CONSTANT'
                    local_w = blend * spine_influence_weight(dist, radius, falloff) * gw
                    local_w = max(0.0, min(1.0, float(local_w)))
                    if local_w > combined.get(vidx, 0.0):
                        combined[vidx] = local_w

            self._influence_overlay_cache = {'key': cache_key, 'weights': combined}
        else:
            combined = cached.get('weights') or {}

        if not combined:
            return

        # Heat-map colors, quantized into a small number of buckets.  This keeps
        # rendering fast (normally <= 32 GPU batches) while using a very stable
        # shader that exists across Blender versions.
        def weight_color(w):
            w = max(0.0, min(1.0, float(w)))
            if w < 0.5:
                u = w * 2.0
                return (0.12 + 0.15*u, 0.35 + 0.55*u, 1.0 - 0.25*u, 0.60 + 0.35*w)
            u = (w - 0.5) * 2.0
            return (0.27 + 0.73*u, 0.90 - 0.70*u, 0.20*(1.0-u), 0.72 + 0.28*w)

        try:
            rv3d = context.region_data
            mw = obj.matrix_world
            size_px = max(1.0, min(20.0, float(getattr(self, '_influence_overlay_size', 4.0) or 4.0)))
            segs = 8
            buckets = {}

            for vidx, wt in combined.items():
                if not (0 <= int(vidx) < len(bm.verts)):
                    continue
                wpos = mw @ bm.verts[int(vidx)].co

                # Compute a true screen-space radius for EACH marker so zooming
                # and perspective do not make the points vanish or change size.
                try:
                    radius_world = float(self._screen_constant_controller_radius(context, wpos, size_px))
                except Exception:
                    radius_world = 0.055
                radius_world = max(1e-6, abs(radius_world))
                right = (rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))).normalized() * radius_world
                up = (rv3d.view_rotation @ Vector((0.0, 1.0, 0.0))).normalized() * radius_world

                # Quantize weight to 24 stable buckets; visual gradient remains
                # smooth enough while avoiding hundreds/thousands of draw calls.
                bucket = max(0, min(23, int(float(wt) * 24.0)))
                pts = buckets.setdefault(bucket, [])
                col = weight_color((bucket + 0.5) / 24.0)
                ring = []
                for j in range(segs):
                    a = (2.0 * math.pi * j) / segs
                    ring.append(wpos + right * math.cos(a) + up * math.sin(a))
                for j in range(segs):
                    pts.extend((wpos, ring[j], ring[(j + 1) % segs]))

            if not buckets:
                return

            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.blend_set('ALPHA')
            gpu.state.depth_test_set('NONE')
            for bucket, pts in buckets.items():
                if not pts:
                    continue
                batch = batch_for_shader(shader, 'TRIS', {'pos': pts})
                shader.bind()
                shader.uniform_float('color', weight_color((bucket + 0.5) / 24.0))
                batch.draw(shader)
        finally:
            try:
                gpu.state.blend_set('NONE')
            except Exception:
                pass
            try:
                gpu.state.depth_test_set('LESS_EQUAL')
            except Exception:
                pass

    def _spine_session_full_reset(self, context):
        """Wipe live chains/controllers and return to empty Place (like first open)."""
        self.spine_chains = []
        self.spine_chains_pts = []
        self.spine_points = []
        self.spine_bind = []
        self.bez = []
        self.rest_bez = []
        self.point_modes = []
        self.spine_tilt = []
        self.spine_radius = []
        self.handle_params = []
        self._spine_origin_ids = []
        self._spine_chains_origin_ids = []
        self._spine_edit_place = False
        self._spine_placing_new_chain = True
        self._spine_last_add_idx = None
        self.active_chain = 0
        self.selected = set()
        self.active_handle = None
        self.dragging = False
        self.box_selecting = False
        self.tool_mode = 'SPINE_PLACE'
        try:
            context.area.tag_redraw()
        except Exception:
            pass

    def _spine_enter_edit_place(self, context):
        """Alt+Enter from DEFORM: edit chains with full handles, synced to deform data."""
        if getattr(self, 'tool_mode', '') != 'SPINE_DEFORM':
            return False
        try:
            self._spine_store_active_chain()
        except Exception:
            pass
        chains = getattr(self, 'spine_chains', None) or []
        if not chains and getattr(self, 'bez', None) and len(self.bez) >= 2:
            chains = [{
                'bez': self.bez,
                'rest_bez': self.rest_bez,
                'modes': self.point_modes,
                'tilt': getattr(self, 'spine_tilt', None),
                'radius': getattr(self, 'spine_radius', None),
                'handle_params': getattr(self, 'handle_params', None),
                'bind': getattr(self, 'spine_bind', None),
                'influence': getattr(self, 'spine_influence', 0.1),
                'origin_ids': list(range(len(self.bez))),
            }]
            self.spine_chains = chains
        if not chains:
            self.report({'WARNING'}, "No chains to edit")
            return False
        # Ensure every chain has origin_ids + modes
        for ch in chains:
            n = len(ch.get('bez') or [])
            if not ch.get('origin_ids') or len(ch.get('origin_ids') or []) != n:
                ch['origin_ids'] = list(range(n))
            if not ch.get('modes') or len(ch.get('modes') or []) != n:
                ch['modes'] = ['AUTO'] * n
        self._spine_push_undo(context)
        # Bake mesh as current rest so rebind keeps deformation
        obj, bm = self.get_obj_bm(context)
        if bm is not None:
            bm.verts.ensure_lookup_table()
            for v in bm.verts:
                self.all_rest[v.index] = v.co.copy()
        # Keep spine_chains as source of truth — only track NEW place chains in pts lists
        self.spine_chains_pts = []
        self._spine_chains_origin_ids = []
        ac = int(getattr(self, 'active_chain', 0) or 0) % len(chains)
        self.active_chain = ac
        try:
            self._spine_load_active_chain()
        except Exception:
            ch = chains[ac]
            self.bez = ch.get('bez')
            self.point_modes = ch.get('modes') or ['AUTO'] * len(self.bez or [])
            self.spine_points = [bp['co'].copy() for bp in (self.bez or [])]
        self.spine_points = [bp['co'].copy() for bp in (self.bez or [])]
        self._spine_origin_ids = list(chains[ac].get('origin_ids') or list(range(len(self.spine_points))))
        self._spine_edit_place = True
        self._spine_placing_new_chain = False
        self._spine_last_add_idx = None
        self.tool_mode = 'SPINE_PLACE'
        self.selected = set()
        self.active_handle = None
        self.active_bez_part = 'co'
        self.dragging = False
        self.report({'INFO'}, f"Edit Place: {len(chains)} chain(s) — click chain to edit  |  Shift+Enter: new  |  Enter: Rebind")
        context.area.tag_redraw()
        return True

    def _spine_pick_chain_and_segment(self, context, event, pixel_dist=28.0):
        """Pick nearest chain under cursor for insert.
        Returns (key, seg_i, local_point, u) or None.
        key: 'current' | int (pts list) | ('chain', ci)
        Samples the real Bezier (not only co-polyline) so green curve hits work.
        """
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return None
        region = context.region
        rv3d = context.region_data
        mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
        mw = obj.matrix_world
        best = None
        best_d = float(pixel_dist)

        def consider_poly(key, pts):
            nonlocal best, best_d
            if not pts or len(pts) < 2:
                return
            for i in range(len(pts) - 1):
                a = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ pts[i])
                b = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ pts[i + 1])
                if a is None or b is None:
                    continue
                abx, aby = b.x - a.x, b.y - a.y
                lab2 = abx * abx + aby * aby
                if lab2 < 1e-8:
                    u, px, py = 0.0, a.x, a.y
                else:
                    u = max(0.0, min(1.0, ((mx - a.x) * abx + (my - a.y) * aby) / lab2))
                    px = a.x + abx * u
                    py = a.y + aby * u
                # Allow near ends too (was too strict before)
                if u < 0.01 or u > 0.99:
                    continue
                dist = math.hypot(mx - px, my - py)
                if dist < best_d:
                    best_d = dist
                    local = pts[i].lerp(pts[i + 1], u)
                    best = (key, i, local, u)

        def consider_bez(key, bez):
            nonlocal best, best_d
            if not bez or len(bez) < 2:
                return
            # Dense samples along full curve; map t -> segment index by handle_params or equal
            samples = max(64, len(bez) * 24)
            prev_sc = None
            prev_local = None
            prev_t = 0.0
            for s in range(samples + 1):
                t = s / samples
                local = eval_bezier_points(bez, t)
                sc = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ local)
                if sc is None:
                    prev_sc = None
                    prev_local = None
                    prev_t = t
                    continue
                if prev_sc is not None:
                    abx = sc.x - prev_sc.x
                    aby = sc.y - prev_sc.y
                    lab2 = abx * abx + aby * aby
                    if lab2 < 1e-8:
                        u, px, py = 0.0, prev_sc.x, prev_sc.y
                    else:
                        u = max(0.0, min(1.0, ((mx - prev_sc.x) * abx + (my - prev_sc.y) * aby) / lab2))
                        px = prev_sc.x + abx * u
                        py = prev_sc.y + aby * u
                    dist = math.hypot(mx - px, my - py)
                    if dist < best_d:
                        best_d = dist
                        t_hit = prev_t + (t - prev_t) * u
                        local_hit = prev_local.lerp(local, u) if prev_local is not None else local
                        # Map t to controller segment index
                        n = len(bez)
                        # Prefer handle_params if available
                        seg_i = min(n - 2, max(0, int(t_hit * (n - 1))))
                        # Refine: nearest controller interval by arc param
                        if n >= 2:
                            seg_i = min(n - 2, max(0, int(round(t_hit * (n - 1) - 0.5))))
                        best = (key, seg_i, local_hit, t_hit)
                prev_sc = sc
                prev_local = local
                prev_t = t

        placing_new = bool(getattr(self, '_spine_placing_new_chain', False))

        # Current place points (the chain being built)
        cur = getattr(self, 'spine_points', None) or []
        if len(cur) >= 2:
            # While building a NEW chain after Shift+Enter, only use poly points —
            # never the previous active deform bez.
            if (
                getattr(self, '_spine_edit_place', False)
                and not placing_new
                and getattr(self, 'bez', None)
                and len(self.bez) >= 2
            ):
                ac = int(getattr(self, 'active_chain', 0) or 0)
                consider_bez(('chain', ac), self.bez)
            else:
                consider_poly('current', cur)

        for i, pts in enumerate(getattr(self, 'spine_chains_pts', None) or []):
            if len(pts) >= 2:
                consider_poly(i, pts)

        # Existing deform chains only when NOT placing a brand-new chain
        if getattr(self, '_spine_edit_place', False) and not placing_new:
            ac = int(getattr(self, 'active_chain', 0) or 0)
            for ci, ch in enumerate(getattr(self, 'spine_chains', None) or []):
                if ci == ac and getattr(self, 'bez', None) and len(self.bez) >= 2:
                    continue  # already considered live bez
                bez = ch.get('bez') or []
                if len(bez) >= 2:
                    consider_bez(('chain', ci), bez)

        return best

    def _spine_insert_controller_on_line(self, context, event):
        """Shift+MMB in Place/Edit Place: insert controller on the chain under cursor."""
        placing_new = bool(getattr(self, '_spine_placing_new_chain', False))
        hit = self._spine_pick_chain_and_segment(context, event)
        if hit is None:
            return False
        key, seg_i, local, u = hit
        self._spine_push_undo(context)

        # While building a new chain, never switch away to an old deform chain
        if placing_new:
            if key != 'current' and not (isinstance(key, int)):
                # Force insert onto the live place points
                key = 'current'
            elif isinstance(key, int):
                # pending completed place chain — activate that poly chain
                try:
                    self._spine_activate_place_chain(key)
                except Exception:
                    pass
        elif key == 'current':
            pass
        elif isinstance(key, tuple) and len(key) == 2 and key[0] == 'chain':
            self._spine_activate_place_chain(key)
        else:
            try:
                self._spine_activate_place_chain(key)
            except Exception:
                pass

        pts = list(getattr(self, 'spine_points', None) or [])
        if len(pts) < 2:
            return False
        oids = list(getattr(self, '_spine_origin_ids', None) or list(range(len(pts))))
        if len(oids) != len(pts):
            oids = list(range(len(pts)))
        # Clamp insert between first and last controller
        insert_at = max(1, min(int(seg_i) + 1, len(pts)))
        # Snap insert point onto segment between neighbors (stable)
        if 0 <= insert_at - 1 < len(pts) and insert_at <= len(pts):
            # local already on curve from pick; keep it
            co = local.copy()
        else:
            co = local.copy()
        pts.insert(insert_at, co)
        oids.insert(insert_at, None)
        self.spine_points = pts
        self._spine_origin_ids = oids
        self._spine_last_add_idx = insert_at

        # Edit Place on an EXISTING deform chain: insert into live Bezier
        # (skip when placing a brand-new chain — only spine_points matter)
        if getattr(self, '_spine_edit_place', False) and not placing_new:
            # Ensure bez exists from active chain
            if not getattr(self, 'bez', None) or len(self.bez) < 2:
                chains = getattr(self, 'spine_chains', None) or []
                ac = int(getattr(self, 'active_chain', 0) or 0)
                if chains and 0 <= ac < len(chains) and chains[ac].get('bez'):
                    self.bez = chains[ac]['bez']
                    self.point_modes = chains[ac].get('modes') or ['AUTO'] * len(self.bez)
            if getattr(self, 'bez', None) and len(self.bez) >= 2:
                bez = self.bez
                # insert_at must match bez length progression
                if insert_at > len(bez):
                    insert_at = len(bez)
                prev = bez[insert_at - 1] if insert_at - 1 < len(bez) else None
                nxt = bez[insert_at] if insert_at < len(bez) else None
                if prev and nxt:
                    hl = co.lerp(prev['co'], 0.33)
                    hr = co.lerp(nxt['co'], 0.33)
                else:
                    hl = co.copy()
                    hr = co.copy()
                bez.insert(insert_at, {'co': co.copy(), 'hl': hl, 'hr': hr})
                modes = list(getattr(self, 'point_modes', None) or ['AUTO'] * (len(bez) - 1))
                while len(modes) < len(bez) - 1:
                    modes.append('AUTO')
                modes.insert(insert_at, 'AUTO')
                self.point_modes = modes
                pfo = ensure_point_inf_falloff(len(bez), getattr(self, 'point_inf_falloff', None), default='CONSTANT')
                if len(pfo) < len(bez):
                    pfo.insert(insert_at, 'CONSTANT')
                self.point_inf_falloff = pfo[:len(bez)]
                # New controller inherits average influence of neighbors
                pinf = ensure_point_influence(
                    len(bez) - 1, getattr(self, 'point_influence', None),
                    default=float(getattr(self, 'spine_influence', 0.1) or 0.1),
                )
                neigh = []
                if insert_at - 1 < len(pinf):
                    neigh.append(pinf[insert_at - 1])
                if insert_at < len(pinf):
                    neigh.append(pinf[insert_at])
                new_inf = (sum(neigh) / len(neigh)) if neigh else float(getattr(self, 'spine_influence', 0.1) or 0.1)
                pinf.insert(insert_at, max(1e-4, new_inf))
                self.point_influence = pinf[:len(bez)]
                if getattr(self, 'spine_tilt', None) is not None:
                    st = list(self.spine_tilt)
                    while len(st) < len(bez) - 1:
                        st.append(0.0)
                    st.insert(insert_at, 0.0)
                    self.spine_tilt = st
                if getattr(self, 'spine_radius', None) is not None:
                    sr = list(self.spine_radius)
                    while len(sr) < len(bez) - 1:
                        sr.append(1.0)
                    sr.insert(insert_at, 1.0)
                    self.spine_radius = sr
                # Immediately refresh AUTO handles after insertion so the new
                # controller and any AUTO neighbors reflect the updated curve.
                # FREE/ALIGNED handles are left untouched.
                try:
                    self.rebuild_auto_handles(interior=True)
                except Exception:
                    pass
                try:
                    self._spine_store_active_chain()
                except Exception:
                    pass
                # Keep points synced to bez
                self.spine_points = [bp['co'].copy() for bp in self.bez]

        # Controller count/curve state changed: never reuse stale Vertex-Mirror
        # pairings or drag baselines on the next modeling operation.
        self._vertex_mirror_invalidate_state(rebuild=True)
        self.select_only(insert_at, 'co')
        self.report({'INFO'}, f"Controller inserted ({len(self.spine_points)})  |  Enter: Rebind")
        context.area.tag_redraw()
        return True


    def _spine_restore_curve_details(self, ch_new, ch_old):
        """Copy handle positions (hl/hr), modes, tilt, radius from old→new by origin_ids.
        New controllers (origin_id None) keep auto-built handles.
        """
        if not ch_new or not ch_old:
            return
        new_bez = ch_new.get('bez') or []
        old_bez = ch_old.get('bez') or []
        if not new_bez or not old_bez:
            return
        n_new = len(new_bez)
        n_old = len(old_bez)
        new_oids = list(ch_new.get('origin_ids') or list(range(n_new)))
        old_oids = list(ch_old.get('origin_ids') or list(range(n_old)))
        if len(new_oids) != n_new:
            new_oids = list(range(n_new))
        if len(old_oids) != n_old:
            old_oids = list(range(n_old))

        # oid -> old index
        oid_to_old = {}
        for oi, oid in enumerate(old_oids):
            if oid is not None:
                oid_to_old[oid] = oi

        old_modes = list(ch_old.get('modes') or ['AUTO'] * n_old)
        old_tilt = list(ch_old.get('tilt') or [0.0] * n_old)
        old_rad = list(ch_old.get('radius') or [1.0] * n_old)
        new_modes = list(ch_new.get('modes') or ['AUTO'] * n_new)
        new_tilt = list(ch_new.get('tilt') or [0.0] * n_new)
        new_rad = list(ch_new.get('radius') or [1.0] * n_new)
        if len(new_modes) != n_new:
            new_modes = ['AUTO'] * n_new
        if len(new_tilt) != n_new:
            new_tilt = [0.0] * n_new
        if len(new_rad) != n_new:
            new_rad = [1.0] * n_new

        for ni, oid in enumerate(new_oids):
            if oid is None or oid not in oid_to_old:
                continue
            oi = oid_to_old[oid]
            if oi < 0 or oi >= n_old:
                continue
            ob = old_bez[oi]
            nb = new_bez[ni]
            # Preserve handle offsets relative to controller
            hl_off = ob['hl'] - ob['co']
            hr_off = ob['hr'] - ob['co']
            nb['hl'] = nb['co'] + hl_off
            nb['hr'] = nb['co'] + hr_off
            if oi < len(old_modes):
                new_modes[ni] = old_modes[oi]
            if oi < len(old_tilt):
                new_tilt[ni] = float(old_tilt[oi])
            if oi < len(old_rad):
                new_rad[ni] = float(old_rad[oi])

        # Ends: clear unused sides
        if n_new >= 1:
            new_bez[0]['hl'] = new_bez[0]['co'].copy()
            new_bez[-1]['hr'] = new_bez[-1]['co'].copy()

        ch_new['bez'] = new_bez
        ch_new['rest_bez'] = copy_bezier_points(new_bez)
        ch_new['modes'] = new_modes
        ch_new['tilt'] = new_tilt
        ch_new['radius'] = new_rad

    def _spine_rebind_preserve(self, context):
        """Rebind all chains from Place edit, keeping prior vert→chain assignment and
        recomputing offsets from current mesh so deformation does not jump.
        New controllers get influence normalized with neighbors via curve t remap.
        """
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return False
        # Store active curve into spine_chains (list identity / live edits)
        try:
            if getattr(self, '_spine_edit_place', False):
                self._spine_store_active_chain()
                if getattr(self, 'bez', None) and len(self.bez) >= 2:
                    self.spine_points = [bp['co'].copy() for bp in self.bez]
        except Exception:
            pass

        bm.verts.ensure_lookup_table()
        # Capture the pre-Edit-Place recall snapshot before rebuilding chains.
        # This function is also called directly by the Enter handler, so it must
        # determine topology changes locally.  The previous versions referenced
        # `topology_changed` here without defining it, which aborted the rebind
        # after VG ownership was updated and before the final deform/apply pass.
        try:
            _prev_recall = _vdh_load_recall_from_mesh(obj) or {}
        except Exception:
            _prev_recall = {}
        _prev_mesh_snap = _prev_recall.get('mesh_snap') or {}
        _saved_vc = int(_prev_recall.get('vert_count', -1) or -1)
        topology_changed = (_saved_vc >= 0 and _saved_vc != len(bm.verts))
        old_chains = list(getattr(self, 'spine_chains', None) or [])

        # Map vidx -> old chain index
        vert_chain = {}
        for ci, ch in enumerate(old_chains):
            for item in (ch.get('bind') or []):
                vert_chain[item[0]] = ci

        # ---- Build full chain list: keep ALL existing deform chains + any NEW place chains ----
        built = []
        # 1) Every existing deform chain (preserves non-active chains + handles)
        for ch in old_chains:
            bez = ch.get('bez') or []
            if len(bez) < 2:
                continue
            bez_copy = copy_bezier_points(bez)
            modes = list(ch.get('modes') or ['AUTO'] * len(bez_copy))
            while len(modes) < len(bez_copy):
                modes.append('AUTO')
            # If modes look wiped (all missing), default AUTO only for missing slots
            n = len(bez_copy)
            modes = [m if m in ('AUTO', 'ALIGNED', 'FREE') else 'AUTO' for m in modes[:n]]
            while len(modes) < n:
                modes.append('AUTO')
            built.append({
                'bez': bez_copy,
                'rest_bez': copy_bezier_points(bez_copy),  # current shape becomes rest
                'modes': list(modes),
                'tilt': list(ch.get('tilt') or [0.0] * n)[:n] or [0.0] * n,
                'radius': list(ch.get('radius') or [1.0] * n)[:n] or [1.0] * n,
                'handle_params': list(ch.get('handle_params') or []),
                'bind': [],
                'influence': float(ch.get('influence') or 0.1),
                'point_influence': ensure_point_influence(
                    n, ch.get('point_influence'),
                    default=float(ch.get('influence') or 0.1),
                ),
                'point_influence_default': ensure_point_influence(
                    n, ch.get('point_influence_default'),
                    default=float(ch.get('influence') or 0.1),
                ),
                'point_inf_falloff': ensure_point_inf_falloff(n, ch.get('point_inf_falloff'), default='CONSTANT'),
                'inf_falloff': ch.get('inf_falloff') or 'CONSTANT',
                'origin_ids': list(ch.get('origin_ids') or list(range(n))),
                'chain_id': ch.get('chain_id'),
                'in_front': bool(ch.get('in_front', True)),
                '_from_existing': True,  # keep edited handles/modes as-is
            })

        # 2) NEW place chains (Shift+Enter extras stored in spine_chains_pts)
        for pts in (getattr(self, 'spine_chains_pts', None) or []):
            if pts and len(pts) >= 2:
                ch_new = self._spine_build_chain_from_pts([p.copy() for p in pts])
                if ch_new and ch_new.get('bez'):
                    built.append(ch_new)

        # 3) If currently placing a brand-new chain (not yet in spine_chains)
        placing_new = bool(getattr(self, '_spine_placing_new_chain', False))
        if placing_new and len(getattr(self, 'spine_points', []) or []) >= 2:
            ch_new = self._spine_build_chain_from_pts([p.copy() for p in self.spine_points])
            if ch_new and ch_new.get('bez'):
                built.append(ch_new)

        if not built:
            self.report({'WARNING'}, "Need controllers to rebind")
            return False

        # Auto-influence for brand-new chains (same heuristic as first bind)
        edge_lens = [e.calc_length() for e in bm.edges]
        avg_edge = (sum(edge_lens) / len(edge_lens)) if edge_lens else 0.01
        try:
            kd = KDTree(len(bm.verts))
            for v in bm.verts:
                kd.insert(v.co, v.index)
            kd.balance()
        except Exception:
            kd = None
        for ch in built:
            n_ctrl = len(ch.get('bez') or [])
            # Existing chains: keep remembered per-controller radii; only pad length
            if ch.get('_from_existing'):
                legacy = float(ch.get('influence') or 0.1) or 0.1
                ch['point_influence'] = ensure_point_influence(
                    n_ctrl, ch.get('point_influence'), default=legacy,
                )
                ch['point_influence_default'] = ensure_point_influence(
                    n_ctrl, ch.get('point_influence_default'), default=(ch['point_influence'][0] if ch.get('point_influence') else legacy),
                )
                ch['point_inf_falloff'] = ensure_point_inf_falloff(
                    n_ctrl, ch.get('point_inf_falloff'), default='CONSTANT',
                )
                if ch['point_influence']:
                    ch['influence'] = max(ch['point_influence'])
                continue
            # Brand-new chains: auto radius covers nearby surface AND thick volume
            influence = 0.0
            rest = ch.get('rest_bez') or ch.get('bez') or []
            if kd is not None and rest:
                for bp in rest:
                    try:
                        # nearest surface samples
                        for _co, _idx, dist in kd.find_n(bp['co'], min(12, max(1, len(bm.verts)))):
                            influence = max(influence, float(dist))
                        # farthest among a wider sample so fat meshes get enough reach
                        for _co, _idx, dist in kd.find_n(bp['co'], min(64, max(1, len(bm.verts)))):
                            influence = max(influence, float(dist) * 0.85)
                    except Exception:
                        pass
            # Also use max distance from curve samples to bound mesh extents
            if rest and bm is not None:
                try:
                    samples_c = max(16, len(rest) * 8)
                    curve_pts = [eval_bezier_points(rest, s / samples_c) for s in range(samples_c + 1)]
                    # sample subset of verts for speed
                    step = max(1, len(bm.verts) // 200)
                    far = 0.0
                    for vi in range(0, len(bm.verts), step):
                        co = bm.verts[vi].co
                        best = 1e18
                        for p in curve_pts:
                            d = (p - co).length_squared
                            if d < best:
                                best = d
                        far = max(far, math.sqrt(best))
                    if far > 1e-8:
                        influence = max(influence, far * 1.05)
                except Exception:
                    pass
            if influence < 1e-8:
                cos = [bp['co'] for bp in rest]
                length = 0.0
                for i in range(1, len(cos)):
                    length += (cos[i] - cos[i - 1]).length
                influence = max(length * 0.15, avg_edge * 4.0)
            else:
                influence = max(influence * 1.15, avg_edge * 4.0)
            influence = float(influence)
            ch['influence'] = influence
            ch['point_influence'] = [influence] * max(1, n_ctrl)
            ch['point_influence_default'] = [influence] * max(1, n_ctrl)
            ch['point_inf_falloff'] = ensure_point_inf_falloff(
                n_ctrl, ch.get('point_inf_falloff'), default='CONSTANT',
            )

        # Sample each chain
        kd_samples = []
        for ci, ch in enumerate(built):
            samples = max(48, len(ch['rest_bez']) * 24)
            pts = [eval_bezier_points(ch['rest_bez'], s / samples) for s in range(samples + 1)]
            kd_samples.append((ci, samples, pts))

        # Clear binds; reassign all verts to nearest chain within that chain's influence
        for ch in built:
            ch['bind'] = []

        for v in bm.verts:
            vidx = v.index
            best_ci, best_t, best_d = -1, 0.0, 1e18
            for ci, samples, pts in ((k[0], k[1], k[2]) for k in kd_samples):
                for s, p in enumerate(pts):
                    d = (p - v.co).length_squared
                    if d < best_d:
                        best_d = d
                        best_t = s / samples
                        best_ci = ci
            if best_ci < 0:
                continue
            best_d = math.sqrt(best_d)
            ch = built[best_ci]
            inf = float(ch.get('influence') or 0.1)
            # New chains and existing: hard radius = influence (matches falloff)
            # Previously-bound verts may keep a slightly looser margin
            limit = inf
            if vidx in vert_chain:
                limit = max(inf, inf * 1.15)
            if best_d > max(limit, 1e-6):
                continue
            on = eval_bezier_points(ch['rest_bez'], best_t)
            tan = bezier_chain_tangent(ch['rest_bez'], best_t)
            offset = v.co - on
            ch['bind'].append((vidx, best_t, offset.copy(), tan.copy(), float(best_d)))
            self.all_rest[vidx] = v.co.copy()

        # Topology duplicate ownership pass: when Edit Mode created new vertices,
        # a newly placed chain must be allowed to claim the new mesh island even if
        # an older chain's influence radius is larger. Blender copies the old
        # vertex-group membership to duplicated vertices, so the new chain must
        # become the owner of NEW vertices that are actually inside its influence.
        try:
            _prev_ids = set(int(k) for k in (_prev_mesh_snap.keys() if isinstance(_prev_mesh_snap, dict) else []))
        except Exception:
            _prev_ids = set()
        if _prev_ids:
            _new_ids = {int(v.index) for v in bm.verts if int(v.index) not in _prev_ids}
        else:
            _new_ids = set()
        if _new_ids:
            for ci, ch in enumerate(built):
                if ch.get('_from_existing'):
                    continue
                rest = ch.get('rest_bez') or ch.get('bez') or []
                if not rest:
                    continue
                samples = kd_samples[ci][1]
                pts = kd_samples[ci][2]
                inf = float(ch.get('influence') or 0.0)
                if inf <= 1e-8:
                    continue
                claimed = set()
                for vidx in _new_ids:
                    if vidx >= len(bm.verts):
                        continue
                    vv = bm.verts[vidx]
                    best_d2 = 1e30
                    best_t2 = 0.0
                    for ss, pp in enumerate(pts):
                        dd = (pp - vv.co).length_squared
                        if dd < best_d2:
                            best_d2 = dd
                            best_t2 = ss / max(1, samples)
                    if math.sqrt(best_d2) <= inf:
                        claimed.add(vidx)
                if not claimed:
                    continue
                # Remove claimed NEW vertices from every other chain, then bind
                # them to this newly created chain. Existing vertices are untouched.
                for other in built:
                    if other is ch:
                        continue
                    other['bind'] = [it for it in (other.get('bind') or []) if int(it[0]) not in claimed]
                for vidx in claimed:
                    vv = bm.verts[vidx]
                    best_d2 = 1e30
                    best_t2 = 0.0
                    for ss, pp in enumerate(pts):
                        dd = (pp - vv.co).length_squared
                        if dd < best_d2:
                            best_d2 = dd
                            best_t2 = ss / max(1, samples)
                    on = eval_bezier_points(rest, best_t2)
                    tan = bezier_chain_tangent(rest, best_t2)
                    offset = vv.co - on
                    ch['bind'].append((vidx, best_t2, offset.copy(), tan.copy(), float(math.sqrt(best_d2))))
                    self.all_rest[vidx] = vv.co.copy()

        # Second pass: any NEW chain still empty → force-bind nearest verts
        for ci, ch in enumerate(built):
            if ch.get('_from_existing'):
                continue
            if ch.get('bind'):
                continue
            rest = ch.get('rest_bez') or ch.get('bez')
            if not rest or len(rest) < 2:
                continue
            samples, pts = kd_samples[ci][1], kd_samples[ci][2]
            inf = float(ch.get('influence') or avg_edge * 6.0)
            # Expand once if needed (and sync per-controller radii)
            if inf < avg_edge * 2.0:
                inf = avg_edge * 6.0
                ch['influence'] = inf
                n_ctrl = len(ch.get('bez') or [])
                ch['point_influence'] = [inf] * max(1, n_ctrl)
            for v in bm.verts:
                best_t, best_d = 0.0, 1e18
                for s, p in enumerate(pts):
                    d = (p - v.co).length_squared
                    if d < best_d:
                        best_d = d
                        best_t = s / samples
                best_d = math.sqrt(best_d)
                if best_d > inf:
                    continue
                # Don't steal verts already closer to another chain
                steal = True
                for cj, ch2 in enumerate(built):
                    if cj == ci:
                        continue
                    for item in (ch2.get('bind') or []):
                        if item[0] == v.index:
                            # compare distances
                            if len(item) > 4 and float(item[4]) <= best_d * 1.05:
                                steal = False
                            break
                    if not steal:
                        break
                if not steal:
                    continue
                # Remove from other binds
                for ch2 in built:
                    if ch2 is ch:
                        continue
                    ch2['bind'] = [it for it in (ch2.get('bind') or []) if it[0] != v.index]
                on = eval_bezier_points(rest, best_t)
                tan = bezier_chain_tangent(rest, best_t)
                offset = v.co - on
                ch['bind'].append((v.index, best_t, offset.copy(), tan.copy(), float(best_d)))
                self.all_rest[v.index] = v.co.copy()

        # Exact offset pass (no jump)
        for ch in built:
            for i, item in enumerate(list(ch['bind'])):
                vidx, t = item[0], float(item[1])
                if vidx >= len(bm.verts):
                    continue
                on = eval_bezier_points(ch['rest_bez'], t)
                tan = bezier_chain_tangent(ch['rest_bez'], t)
                offset = bm.verts[vidx].co - on
                ch['bind'][i] = (vidx, t, offset.copy(), tan.copy(), self._spine_item_radial_dist((vidx, t, offset, tan, 0.0)))

        # origin_ids: keep what each built chain already has; only fill missing
        for i, ch in enumerate(built):
            n = len(ch['bez'])
            oids = ch.get('origin_ids')
            if not oids or len(oids) != n:
                if i < len(old_chains) and len(old_chains[i].get('origin_ids') or []) == n:
                    ch['origin_ids'] = list(old_chains[i]['origin_ids'])
                else:
                    ch['origin_ids'] = list(range(n))
        # Transfer chain_id from old chains by vert-overlap (stable reset matching)
        old_vert_sets = [
            set(item[0] for item in (och.get('bind') or []))
            for och in old_chains
        ]
        used_old = set()
        for ci, ch in enumerate(built):
            new_verts = set(item[0] for item in (ch.get('bind') or []))
            best_oj, best_ov = -1, 0
            for oj, oset in enumerate(old_vert_sets):
                if oj in used_old:
                    continue
                ov = len(new_verts & oset)
                if ov > best_ov:
                    best_ov, best_oj = ov, oj
            if best_oj >= 0 and best_ov > 0:
                used_old.add(best_oj)
                oid = old_chains[best_oj].get('chain_id')
                if oid:
                    ch['chain_id'] = oid
                elif not ch.get('chain_id'):
                    ch['chain_id'] = f"chain_{best_oj}_{id(ch) & 0xFFFFFF:x}"
                if 'influence' not in ch or not ch.get('influence'):
                    ch['influence'] = float(old_chains[best_oj].get('influence') or 0.0)
            else:
                # brand-new chain (e.g. third finger)
                ch['chain_id'] = f"chain_new_{ci}_{id(ch) & 0xFFFFFF:x}"

        # Existing chains already copied full bez+modes above — do NOT restore from
        # offsets (that can snap handles back toward older AUTO shapes).
        # Only restore details for brand-new chains built from plain points.
        for ci, ch in enumerate(built):
            if ch.get('_from_existing'):
                ch.pop('_from_existing', None)
                continue
            matched = None
            cid = ch.get('chain_id')
            if cid:
                for och in old_chains:
                    if och.get('chain_id') == cid:
                        matched = och
                        break
            if matched is None and ci < len(old_chains):
                matched = old_chains[ci]
            if matched is not None:
                try:
                    self._spine_restore_curve_details(ch, matched)
                except Exception:
                    pass

        self.spine_chains = built
        self.spine_chains_pts = []
        self._spine_chains_origin_ids = []
        self._spine_origin_ids = []
        self._spine_placing_new_chain = False
        self.active_chain = min(int(getattr(self, 'active_chain', 0) or 0), max(0, len(built) - 1))
        self._spine_load_active_chain()
        self._spine_edit_place = False
        self.tool_mode = 'SPINE_DEFORM'

        # Transfer each bound vertex to the BH_Spine group of the chain that deforms it.
        try:
            self._spine_sync_bound_vertex_group_ownership(obj, self.spine_chains)
        except Exception:
            pass
        # Edit Place -> Enter: immediately rebuild Blender's BH_Spine* groups.
        # If Edit Mode duplicated geometry, Blender copied the old groups onto the
        # new vertices. Remove those copied assignments before SYNC seeds the new chain.
        if topology_changed:
            try:
                self._spine_remove_new_topology_weights(obj, _prev_mesh_snap.keys(), self.spine_chains)
            except Exception:
                pass
        # The bind lists are rebuilt above, but without this sync the actual
        # vertex-group deform layer can remain stale until the addon is reopened.
        try:
            self._spine_sync_weight_groups(context, mode='SYNC')
        except Exception:
            pass
        try:
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            obj.data.update()
            context.view_layer.update()
        except Exception:
            pass
        # Return to Deform without selecting/grabbing a controller.
        # Keep the active handle index internally so W and other Spine
        # operations retain exactly the same active-chain behaviour as v43.
        self.selected = set()
        self.active_handle = 0 if self.bez else None
        self.active_bez_part = 'co'
        self.all_rest = {v.index: v.co.copy() for v in bm.verts}
        # Merge bind-rest: keep first-bind snapshot (Reset returns to original Enter state)
        try:
            self._spine_merge_bind_rest(context)
        except Exception:
            pass
        try:
            self._spine_save_recall(context)
        except Exception:
            pass
        # Finalize the confirmed Edit Place state only after ALL bind/VG/rest caches
        # have been updated.  Applying earlier can use the previous session cache,
        # which makes a newly-created second chain appear inert until the tool is
        # activated again.  This final pass uses the exact state that was just saved.
        try:
            self._spine_apply(context, auto_soft=False)
            try:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                obj.data.update()
                context.view_layer.update()
            except Exception:
                pass
        except Exception:
            pass

        # Edit Place -> Enter must always return to a completely idle Deform state.
        # Do NOT leave an active handle behind: Deform's mouse/drag path treats an
        # active handle as a valid grab target even when `selected` is empty.  This
        # was the source of the controller-at-the-end auto-grab after editing a
        # controller's radius/position and pressing Enter.  W/E and other Spine
        # geometry operators use active_chain, not active_handle, so clearing this
        # cannot break them.  A later real click restores active_handle normally.
        self.selected = set()
        self.active_handle = None
        self.active_bez_part = 'co'
        self.dragging = False
        self._pending_click_drag = False
        self._pending_place_drag = False
        self._xform_mode = None
        self._xform_start = None
        self._xform_keys = None
        self.constraint_axis = None

        self.report({'INFO'}, f"Rebind preserve: {len(built)} chain(s)")
        context.area.tag_redraw()
        return True


    def _spine_lock_mesh_selection(self, context):
        """Keep mesh verts/edges/faces deselected so only controllers are selectable."""
        obj, bm = self.get_obj_bm(context)
        if obj is None or bm is None:
            return
        changed = False
        try:
            bm.verts.ensure_lookup_table()
        except Exception:
            pass
        for v in bm.verts:
            if v.select:
                v.select = False
                changed = True
        for e in bm.edges:
            if e.select:
                e.select = False
                changed = True
        for f in bm.faces:
            if f.select:
                f.select = False
                changed = True
        # Clear active element
        try:
            if bm.select_history:
                bm.select_history.clear()
                changed = True
        except Exception:
            pass
        if changed:
            try:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            except Exception:
                pass

    def _spine_start_xform(self, context, event, mode, keys):
        """Start rotate/scale of selected controllers (supports multi-chain keys)."""
        # keys: list of (ci,i,'co') or plain indices on active chain
        norm = []
        ac = int(getattr(self, 'active_chain', 0) or 0)
        for k in keys:
            if isinstance(k, tuple) and len(k) == 3:
                norm.append(k)
            elif isinstance(k, tuple) and len(k) == 2:
                norm.append((ac, k[0], k[1]))
            else:
                norm.append((ac, int(k), 'co'))
        if not norm:
            return

        # Spine-only convenience: starting Rotate/Scale on an AUTO controller
        # promotes that controller to ALIGNED before the transform snapshot is
        # taken. This keeps G/Grab behavior unchanged and only affects R/S.
        if mode in {'ROTATE', 'SCALE'} and getattr(self, 'tool_mode', '') in ('SPINE_DEFORM', 'SPINE_PLACE'):
            chains_for_mode = getattr(self, 'spine_chains', None) or []
            active_ci = int(getattr(self, 'active_chain', 0) or 0)
            for ci, i, part in norm:
                if part != 'co':
                    continue
                if chains_for_mode and 0 <= ci < len(chains_for_mode):
                    bez_ci = chains_for_mode[ci].get('bez') or []
                    modes_ci = list(chains_for_mode[ci].get('modes') or ['AUTO'] * len(bez_ci))
                    if len(modes_ci) != len(bez_ci):
                        modes_ci = (modes_ci + ['AUTO'] * len(bez_ci))[:len(bez_ci)]
                    if 0 <= i < len(modes_ci) and modes_ci[i] == 'AUTO':
                        modes_ci[i] = 'ALIGNED'
                        chains_for_mode[ci]['modes'] = modes_ci
                        if ci == active_ci:
                            self.point_modes = list(modes_ci)
                elif ci == active_ci:
                    pm = getattr(self, 'point_modes', None)
                    if pm is not None and 0 <= i < len(pm) and pm[i] == 'AUTO':
                        pm[i] = 'ALIGNED'

        self._xform_mode = mode
        self.dragging = True
        self._xform_keys = norm
        # Fresh numeric-input state for every transform. Without resetting this,
        # a previous R/S operation could leak its numeric value into the next one.
        self._xform_numeric = ''
        self._xform_numeric_active = False
        # Keep current active if it is among selected keys (do NOT jump to norm[0])
        ac = int(getattr(self, 'active_chain', 0) or 0)
        ah = getattr(self, 'active_handle', None)
        keep = False
        if ah is not None:
            for ci, i, p in norm:
                if ci == ac and int(i) == int(ah):
                    keep = True
                    break
        if not keep:
            # Prefer a key on current active chain, else first key
            pick = None
            for ci, i, p in norm:
                if ci == ac:
                    pick = (ci, i, p)
                    break
            if pick is None:
                pick = norm[0]
            self.active_chain = pick[0]
            self.active_handle = pick[1]
            self.active_bez_part = pick[2] if pick[2] else 'co'
        self.constraint_axis = None
        # Do NOT store_active here — self.bez may be stale and would wipe a chain.
        # Read poses directly from spine_chains.
        chains = getattr(self, 'spine_chains', None) or []
        cos = []
        self._xform_start = {}
        for ci, i, p in norm:
            if chains and 0 <= ci < len(chains):
                bez = chains[ci].get('bez') or []
            else:
                bez = self.bez or []
            if 0 <= i < len(bez):
                bp = bez[i]
                cos.append(bp['co'].copy())
                self._xform_start[(ci, i)] = {
                    'co': bp['co'].copy(),
                    'hl': bp['hl'].copy(),
                    'hr': bp['hr'].copy(),
                }
        if not cos:
            return
        # Pivot from Blender scene setting (same as header Pivot Point)
        pivot_mode = 'MEDIAN_POINT'
        try:
            pivot_mode = str(context.scene.tool_settings.transform_pivot_point or 'MEDIAN_POINT')
        except Exception:
            pivot_mode = 'MEDIAN_POINT'
        self._xform_pivot_mode = pivot_mode

        median = sum(cos, Vector((0, 0, 0))) / float(len(cos))
        center = median.copy()

        if pivot_mode == 'CURSOR':
            try:
                obj_tmp, _ = self.get_obj_bm(context)
                if obj_tmp is not None:
                    center = obj_tmp.matrix_world.inverted() @ context.scene.cursor.location
            except Exception:
                center = median.copy()
        elif pivot_mode == 'ACTIVE_ELEMENT':
            # Pivot = active controller's co (Blender-like). Must be visible as active.
            ac = int(getattr(self, 'active_chain', 0) or 0)
            ah = getattr(self, 'active_handle', None)
            got = False
            # Prefer active if it is among xform keys
            if ah is not None:
                key_act = (ac, int(ah))
                if key_act in self._xform_start:
                    center = self._xform_start[key_act]['co'].copy()
                    got = True
                else:
                    # Active may be on another chain — still use it as pivot
                    if chains and 0 <= ac < len(chains):
                        bez_a = chains[ac].get('bez') or []
                    else:
                        bez_a = self.bez or []
                    if 0 <= int(ah) < len(bez_a):
                        center = bez_a[int(ah)]['co'].copy()
                        got = True
            if not got and norm:
                ci, i, _p = norm[0]
                st0 = self._xform_start.get((ci, i))
                center = st0['co'].copy() if st0 else median.copy()
        elif pivot_mode == 'BOUNDING_BOX_CENTER':
            xs = [c.x for c in cos]
            ys = [c.y for c in cos]
            zs = [c.z for c in cos]
            center = Vector((
                0.5 * (min(xs) + max(xs)),
                0.5 * (min(ys) + max(ys)),
                0.5 * (min(zs) + max(zs)),
            ))
        else:
            # MEDIAN_POINT, INDIVIDUAL_ORIGINS (disabled), and unknown → median
            center = median.copy()
            if pivot_mode == 'INDIVIDUAL_ORIGINS':
                self._xform_pivot_mode = 'MEDIAN_POINT'

        self._xform_center = center
        self._xform_start_mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
        obj, _ = self.get_obj_bm(context)
        if obj is not None:
            region = context.region
            rv3d = context.region_data
            wcen = obj.matrix_world @ center
            sc = view3d_utils.location_3d_to_region_2d(region, rv3d, wcen)
            if sc is not None:
                self._xform_center_2d = Vector((sc.x, sc.y))
                self._xform_start_angle = math.atan2(
                    event.mouse_region_y - sc.y, event.mouse_region_x - sc.x
                )
                self._xform_start_dist = max(1.0, math.hypot(
                    event.mouse_region_x - sc.x, event.mouse_region_y - sc.y
                ))
            else:
                self._xform_center_2d = Vector((event.mouse_region_x, event.mouse_region_y))
                self._xform_start_angle = 0.0
                self._xform_start_dist = 100.0
        rv3d = context.region_data
        self._xform_axis = (rv3d.view_rotation @ Vector((0, 0, 1))).normalized()

    def _spine_update_xform(self, context, event):
        mode = getattr(self, '_xform_mode', None)
        if not mode or not getattr(self, '_xform_start', None):
            return
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return
        center = self._xform_center
        c2d = getattr(self, '_xform_center_2d', None)
        chains = getattr(self, 'spine_chains', None) or []
        individual = getattr(self, '_xform_pivot_mode', '') == 'INDIVIDUAL_ORIGINS'

        # Blender-style transform orientation / axis constraint.  Grab already used
        # _apply_axis_constraint; R/S now use the same orientation basis so X/Y/Z
        # mean the active Blender Transform Orientation rather than hard-coded local axes.
        constraint = getattr(self, 'constraint_axis', None)
        orient = self._transform_orient_matrix(context, obj)
        try:
            orient = orient.to_3x3()
        except Exception:
            orient = Matrix.Identity(3)
        try:
            ax_x = orient.col[0].normalized()
            ax_y = orient.col[1].normalized()
            ax_z = orient.col[2].normalized()
        except Exception:
            ax_x, ax_y, ax_z = Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1))
        axis_world = None
        if constraint in {'X', 'Y', 'Z'}:
            axis_world = {'X': ax_x, 'Y': ax_y, 'Z': ax_z}[constraint]
        elif constraint in {'YZ', 'XZ', 'XY'}:
            # Blender's Shift+Axis transform constraint excludes that axis for
            # scale/translate. For rotation the excluded-axis plane has the same
            # normal axis, so use that normal as the rotation axis.
            axis_world = {'YZ': ax_x, 'XZ': ax_y, 'XY': ax_z}[constraint]
        # Convert the object's local pivot to world for a consistent rotation/scale.
        mw = obj.matrix_world
        mw3 = mw.to_3x3()
        try:
            imw3 = mw3.inverted()
        except Exception:
            imw3 = Matrix.Identity(3)
        if mode == 'ROTATE' and c2d is not None:
            ang = math.atan2(event.mouse_region_y - c2d.y, event.mouse_region_x - c2d.x)
            delta = ang - getattr(self, '_xform_start_angle', 0.0)
            if getattr(self, '_xform_numeric_active', False):
                try:
                    delta = math.radians(float(self._xform_numeric or 0.0))
                except Exception:
                    pass
            # Unconstrained R = view-plane rotation (Blender-like).
            # X/Y/Z = rotation around the corresponding Transform Orientation axis.
            axis_world = axis_world if axis_world is not None else (context.region_data.view_rotation @ Vector((0, 0, 1))).normalized()
            R = Matrix.Rotation(delta, 3, axis_world)
            Rloc = imw3 @ R @ mw3
            for (ci, i), st in self._xform_start.items():
                if chains and 0 <= ci < len(chains):
                    bez = chains[ci].get('bez') or []
                else:
                    bez = self.bez or []
                if i >= len(bez):
                    continue
                pivot = st['co'] if individual else center
                for part in ('co', 'hl', 'hr'):
                    rel = st[part] - pivot
                    bez[i][part] = pivot + Rloc @ rel
        elif mode == 'SCALE' and c2d is not None:
            dist = max(1.0, math.hypot(event.mouse_region_x - c2d.x, event.mouse_region_y - c2d.y))
            factor = dist / max(getattr(self, '_xform_start_dist', 100.0), 1.0)
            if getattr(self, '_xform_numeric_active', False):
                try:
                    factor = float(self._xform_numeric)
                except Exception:
                    pass
            factor = max(0.01, min(100.0, factor))
            for (ci, i), st in self._xform_start.items():
                if chains and 0 <= ci < len(chains):
                    bez = chains[ci].get('bez') or []
                else:
                    bez = self.bez or []
                if i >= len(bez):
                    continue
                pivot = st['co'] if individual else center
                for part in ('co', 'hl', 'hr'):
                    rel = st[part] - pivot
                    if axis_world is None:
                        # Uniform scale.
                        scaled = rel * factor
                    else:
                        if constraint in {'YZ', 'XZ', 'XY'}:
                            # Shift+Axis = scale in the plane perpendicular to the
                            # excluded axis, leaving that axis unchanged.
                            excluded = axis_world * rel.dot(axis_world)
                            plane = rel - excluded
                            scaled = excluded + plane * factor
                        else:
                            # Axis-constrained scale: scale only the component along
                            # the selected Blender orientation axis.
                            par = axis_world * rel.dot(axis_world)
                            perp = rel - par
                            scaled = perp + par * factor
                    bez[i][part] = pivot + scaled
        if chains:
            affected = sorted({ci for (ci, _i) in self._xform_start.keys()})
            prev_ac = int(getattr(self, 'active_chain', 0) or 0)
            for ci in affected:
                try:
                    self.active_chain = ci
                    self._spine_load_active_chain()
                    self.rebuild_auto_handles()
                    self._spine_store_active_chain()
                except Exception:
                    pass
            try:
                self.active_chain = prev_ac
                self._spine_load_active_chain()
            except Exception:
                pass
        else:
            try:
                self.rebuild_auto_handles()
            except Exception:
                pass


    def _spine_finish_place_box_select(self, context, event):
        """Box-select controllers in Place/Edit Place.
        Normal box: controllers (co).  Ctrl+Shift box: handle tips only.
        """
        if not self.box_start or not self.box_end:
            return
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return
        x0, y0 = self.box_start
        x1, y1 = self.box_end
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        region = context.region
        rv3d = context.region_data
        mw = obj.matrix_world
        found = set()
        handles_only = bool(getattr(self, 'box_handles_only', False))
        edit = bool(getattr(self, '_spine_edit_place', False))

        def in_box(co):
            sc = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ co)
            if sc is None:
                return False
            return xmin <= sc.x <= xmax and ymin <= sc.y <= ymax

        # Edit Place with bez: select on active chain (and switch if hits another chain)
        if edit and getattr(self, 'spine_chains', None):
            chains = self.spine_chains
            ac = int(getattr(self, 'active_chain', 0) or 0)
            # Prefer hits on active; if none, activate first other chain with hits
            per_chain = {}
            for ci, ch in enumerate(chains):
                bez = ch.get('bez') or []
                n = len(bez)
                hits = set()
                for i, bp in enumerate(bez):
                    if handles_only:
                        if i == 0:
                            parts = ['hr']
                        elif i == n - 1:
                            parts = ['hl']
                        else:
                            parts = ['hl', 'hr']
                    else:
                        parts = ['co']
                    for part in parts:
                        if part != 'co' and (bp[part] - bp['co']).length < 1e-8:
                            continue
                        if in_box(bp[part]):
                            hits.add((i, part))
                if hits:
                    per_chain[ci] = hits
            use_ci = ac if ac in per_chain else (next(iter(per_chain)) if per_chain else None)
            if use_ci is not None:
                if use_ci != ac:
                    self._spine_activate_place_chain(('chain', use_ci))
                found = per_chain[use_ci]
        else:
            # Points-only place (or active spine_points)
            pts = getattr(self, 'spine_points', None) or []
            if handles_only and getattr(self, 'bez', None):
                bez = self.bez
                n = len(bez)
                for i, bp in enumerate(bez):
                    if i == 0:
                        parts = ['hr']
                    elif i == n - 1:
                        parts = ['hl']
                    else:
                        parts = ['hl', 'hr']
                    for part in parts:
                        if (bp[part] - bp['co']).length < 1e-8:
                            continue
                        if in_box(bp[part]):
                            found.add((i, part))
            else:
                for i, co in enumerate(pts):
                    if in_box(co):
                        found.add((i, 'co'))

        if event.shift:
            self.selected = set(self.selected or set()) | found
        else:
            self.selected = found
        if self.selected:
            item = next(iter(self.selected))
            if len(item) == 3:
                self.active_handle = item[1]
                self.active_bez_part = item[2]
            else:
                self.active_handle = item[0]
                self.active_bez_part = item[1]

    def _spine_finish_box_select(self, context, event):
        """Box-select controllers (all chains) or handle tips (Ctrl+Shift)."""
        obj, _ = self.get_obj_bm(context)
        if obj is None or not self.box_start or not self.box_end:
            return
        x0, y0 = self.box_start
        x1, y1 = self.box_end
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        region = context.region
        rv3d = context.region_data
        mw = obj.matrix_world
        found = set()
        handles_only = bool(getattr(self, 'box_handles_only', False))
        chains = getattr(self, 'spine_chains', None) or []
        if not chains and getattr(self, 'bez', None):
            chains = [{'bez': self.bez}]
        for ci, ch in enumerate(chains):
            bez = ch.get('bez') or []
            n = len(bez)
            for i, bp in enumerate(bez):
                if handles_only:
                    if i == 0:
                        parts = ['hr']
                    elif i == n - 1:
                        parts = ['hl']
                    else:
                        parts = ['hl', 'hr']
                else:
                    parts = ['co']
                for part in parts:
                    if part != 'co' and (bp[part] - bp['co']).length < 1e-8:
                        continue
                    sc = view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ bp[part])
                    if sc is None:
                        continue
                    if xmin <= sc.x <= xmax and ymin <= sc.y <= ymax:
                        found.add((ci, i, part))
        if event.shift:
            self.selected = self._spine_norm_selected() | found
        else:
            self.selected = found
        if self.selected:
            ci, i, p = next(iter(self.selected))
            if getattr(self, 'spine_chains', None):
                try:
                    self._spine_store_active_chain()
                except Exception:
                    pass
            self.active_chain = ci
            self.active_handle = i
            self.active_bez_part = p
            if getattr(self, 'spine_chains', None):
                try:
                    self._spine_load_active_chain()
                except Exception:
                    pass


    def _spine_axis_align_begin(self, context, event):
        """Start one-shot mouse-directed axis alignment for selected handle tips.

        The selected item is the handle (hl/hr), while the controller position
        remains the pivot/origin for the alignment.  This allows Free handles to
        be aligned independently without rotating both sides of the controller.
        """
        if self.tool_mode != 'SPINE_DEFORM':
            return False
        sel = self._spine_norm_selected()
        handles = []
        chains = getattr(self, 'spine_chains', None) or []
        if chains:
            for item in sel:
                try:
                    ci, i, part = item
                except Exception:
                    continue
                if part not in {'hl', 'hr'} or not (0 <= ci < len(chains)):
                    continue
                bez = chains[ci].get('bez') or []
                if 0 <= i < len(bez):
                    handles.append((ci, i, part))
        else:
            bez = getattr(self, 'bez', None) or []
            for item in sel:
                try:
                    i, part = item
                except Exception:
                    continue
                if part in {'hl', 'hr'} and 0 <= i < len(bez):
                    handles.append((0, i, part))

        # A selected controller alone is intentionally NOT enough.  The gesture
        # operates on selected handle tips; the controller is only the pivot.
        if not handles:
            self.report({'INFO'}, "Align Axis: select a handle tip first")
            return False

        region = context.region
        rv3d = context.region_data
        obj, _ = self.get_obj_bm(context)
        if region is None or rv3d is None or obj is None:
            return False

        # Use the controller of the first selected handle as the mouse reference.
        ci0, i0, _part0 = handles[0]
        if chains:
            bp0 = (chains[ci0].get('bez') or [])[i0]
        else:
            bp0 = (getattr(self, 'bez', None) or [])[i0]
        sc = view3d_utils.location_3d_to_region_2d(region, rv3d, obj.matrix_world @ bp0['co'])
        if sc is None:
            sc = Vector((event.mouse_region_x, event.mouse_region_y))

        self._spine_axis_align = True
        self._spine_axis_align_start_mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))
        self._spine_axis_align_origin_2d = Vector((float(sc.x), float(sc.y)))
        self._spine_axis_align_handles = list(dict.fromkeys(handles))
        self._spine_axis_align_done = False
        self._spine_axis_align_threshold = 18.0
        return True

    def _spine_axis_align_finish(self, context, event):
        """Align selected handle vectors to the signed world axis chosen by mouse direction."""
        if not getattr(self, '_spine_axis_align', False) or getattr(self, '_spine_axis_align_done', False):
            return False
        dx = float(event.mouse_region_x) - float(self._spine_axis_align_start_mouse.x)
        dy = float(event.mouse_region_y) - float(self._spine_axis_align_start_mouse.y)
        if dx * dx + dy * dy < float(getattr(self, '_spine_axis_align_threshold', 18.0)) ** 2:
            return False
        region = context.region
        rv3d = context.region_data
        obj, _ = self.get_obj_bm(context)
        if region is None or rv3d is None or obj is None:
            return False

        move = Vector((dx, dy))
        move.normalize()
        axis_defs = (
            ('+X', Vector((1, 0, 0))), ('-X', Vector((-1, 0, 0))),
            ('+Y', Vector((0, 1, 0))), ('-Y', Vector((0, -1, 0))),
            ('+Z', Vector((0, 0, 1))), ('-Z', Vector((0, 0, -1))),
        )
        best = None
        best_score = -1.0
        local_origin = self._spine_axis_align_world_origin(context)
        world0 = obj.matrix_world @ local_origin
        q0 = view3d_utils.location_3d_to_region_2d(region, rv3d, world0)
        if q0 is None:
            q0 = self._spine_axis_align_origin_2d
        for name, axis in axis_defs:
            q1 = view3d_utils.location_3d_to_region_2d(region, rv3d, world0 + axis)
            if q1 is None:
                continue
            av = Vector((q1.x - q0.x, q1.y - q0.y))
            if av.length < 1e-5:
                continue
            av.normalize()
            score = self._bh_safe_dot(move, av)
            if score > best_score:
                best_score = score
                best = (name, axis.copy())
        if best is None:
            return False

        axis_name, world_axis = best
        self._spine_push_undo(context)
        chains = getattr(self, 'spine_chains', None) or []
        mw3 = obj.matrix_world.to_3x3()
        try:
            imw3 = mw3.inverted()
        except Exception:
            imw3 = Matrix.Identity(3)
        local_axis = (imw3 @ world_axis).normalized()
        changed = False
        affected_chains = set()

        for ci, i, part in getattr(self, '_spine_axis_align_handles', []):
            if chains and 0 <= ci < len(chains):
                bez = chains[ci].get('bez') or []
            else:
                bez = getattr(self, 'bez', None) or []
            if not (0 <= i < len(bez)):
                continue
            bp = bez[i]
            co = bp['co'].copy()
            handle = bp[part]
            vec = handle - co
            length = vec.length
            if length < 1e-8:
                continue
            # The controller remains the pivot.  FREE affects only the selected
            # handle; ALIGNED must update the opposite handle immediately too,
            # otherwise the stored ALIGNED state is temporarily inconsistent and
            # Blender only appears to correct it on the next mouse drag.
            bp[part] = co + local_axis * length
            mode_list = chains[ci].get('modes') if chains and 0 <= ci < len(chains) else None
            mode = mode_list[i] if mode_list and 0 <= i < len(mode_list) else getattr(bp, 'mode', 'AUTO')
            if isinstance(mode, str) and mode.upper() == 'ALIGNED':
                # ALIGNED is a stored two-sided relationship.  Do not rely on
                # the later mouse-drag path to repair the opposite tip: write
                # BOTH tips now, using their own original lengths and exact
                # opposite directions from the controller pivot.
                other = 'hr' if part == 'hl' else 'hl'
                other_vec = bp[other] - co
                other_len = other_vec.length
                # The selected handle must follow the mouse-selected signed
                # axis.  The opposite ALIGNED handle stays exactly opposite it.
                # Do not derive the sign from hl/hr: that made one side always
                # point against the mouse direction.
                bp[part] = co + local_axis * length
                if other_len > 1e-8:
                    bp[other] = co - local_axis * other_len
                # Keep the mode explicitly ALIGNED in both the persistent
                # chain state and the active-chain mirror used by point_mode().
                if mode_list is not None and 0 <= i < len(mode_list):
                    mode_list[i] = 'ALIGNED'
                if (not chains) or ci == int(getattr(self, 'active_chain', 0) or 0):
                    try:
                        if len(self.point_modes) != len(bez):
                            self.point_modes = list(mode_list) if mode_list is not None else list(self.point_modes)
                        if 0 <= i < len(self.point_modes):
                            self.point_modes[i] = 'ALIGNED'
                    except Exception:
                        pass
            changed = True
            affected_chains.add(int(ci))

        if changed:
            # Rebuild AUTO neighbors immediately, but do not replace the selected
            # explicit handle that was just aligned.  Save/reload each affected
            # chain so the live overlay updates in the same event.
            try:
                original_chain = int(getattr(self, 'active_chain', 0) or 0)
                for rci in sorted(affected_chains):
                    if not (0 <= rci < len(chains)):
                        continue
                    self.active_chain = rci
                    self._spine_load_active_chain()
                    self.rebuild_auto_handles()
                    self.spine_points = [bp['co'].copy() for bp in (getattr(self, 'bez', None) or [])]
                    self._spine_store_active_chain()
                if 0 <= original_chain < len(chains):
                    self.active_chain = original_chain
                    self._spine_load_active_chain()
            except Exception:
                pass
            self._spine_apply(context)
            self._spine_axis_align_done = True
            self._spine_axis_align = False
            self.report({'INFO'}, f"Handle aligned to {axis_name} axis")
            context.area.tag_redraw()
            return True

        self._spine_axis_align = False
        return False

    def _spine_axis_align_world_origin(self, context):
        chains = getattr(self, 'spine_chains', None) or []
        ci, i = getattr(self, '_spine_axis_align_controllers', [(0, 0)])[0]
        if chains and 0 <= ci < len(chains):
            bez = chains[ci].get('bez') or []
        else:
            bez = getattr(self, 'bez', None) or []
        if 0 <= i < len(bez):
            return bez[i]['co'].copy()
        return Vector((0, 0, 0))

    def _spine_axis_align_local_origin(self, context):
        return self._spine_axis_align_world_origin(context)

    def _vertex_axis_align_begin(self, context, event):
        """Start the same mouse-directed signed-axis handle alignment used by Spine Mode."""
        sel = getattr(self, 'selected', set()) or set()
        handles = []
        for item in sel:
            try:
                i, part = item
            except Exception:
                continue
            if part in {'hl', 'hr'} and 0 <= int(i) < len(getattr(self, 'bez', None) or []):
                handles.append((int(i), part))

        # A selected controller alone is intentionally not enough. The gesture
        # operates on selected handle tips; the controller is only the pivot.
        if not handles:
            self.report({'INFO'}, "Align Axis: select a handle tip first")
            return False

        region = context.region
        rv3d = context.region_data
        obj, _ = self.get_obj_bm(context)
        if region is None or rv3d is None or obj is None:
            return False

        i0, _part0 = handles[0]
        bez = getattr(self, 'bez', None) or []
        if not (0 <= i0 < len(bez)):
            return False
        bp0 = bez[i0]
        sc = view3d_utils.location_3d_to_region_2d(
            region, rv3d, obj.matrix_world @ bp0['co']
        )
        if sc is None:
            sc = Vector((event.mouse_region_x, event.mouse_region_y))

        self._vertex_axis_align = True
        self._vertex_axis_align_start_mouse = Vector((
            float(event.mouse_region_x), float(event.mouse_region_y)
        ))
        self._vertex_axis_align_origin_2d = Vector((float(sc.x), float(sc.y)))
        self._vertex_axis_align_handles = list(dict.fromkeys(handles))
        self._vertex_axis_align_done = False
        self._vertex_axis_align_threshold = 18.0
        return True

    def _vertex_axis_align_finish(self, context, event):
        """Align selected Vertex Mode handle vectors to the signed world axis chosen by mouse direction."""
        if (not getattr(self, '_vertex_axis_align', False)
                or getattr(self, '_vertex_axis_align_done', False)):
            return False

        dx = float(event.mouse_region_x) - float(self._vertex_axis_align_start_mouse.x)
        dy = float(event.mouse_region_y) - float(self._vertex_axis_align_start_mouse.y)
        if dx * dx + dy * dy < float(getattr(self, '_vertex_axis_align_threshold', 18.0)) ** 2:
            return False

        region = context.region
        rv3d = context.region_data
        obj, _ = self.get_obj_bm(context)
        if region is None or rv3d is None or obj is None:
            return False

        move = Vector((dx, dy))
        move.normalize()
        axis_defs = (
            ('+X', Vector((1, 0, 0))), ('-X', Vector((-1, 0, 0))),
            ('+Y', Vector((0, 1, 0))), ('-Y', Vector((0, -1, 0))),
            ('+Z', Vector((0, 0, 1))), ('-Z', Vector((0, 0, -1))),
        )

        # Use the same mouse-to-axis projection logic as Spine Mode, but with
        # the first selected Vertex controller as the pivot/reference.
        i0, _part0 = self._vertex_axis_align_handles[0]
        bez = getattr(self, 'bez', None) or []
        if not (0 <= i0 < len(bez)):
            self._vertex_axis_align = False
            return False
        world0 = obj.matrix_world @ bez[i0]['co']
        q0 = view3d_utils.location_3d_to_region_2d(region, rv3d, world0)
        if q0 is None:
            q0 = self._vertex_axis_align_origin_2d

        best = None
        best_score = -1.0
        for name, axis in axis_defs:
            q1 = view3d_utils.location_3d_to_region_2d(region, rv3d, world0 + axis)
            if q1 is None:
                continue
            av = Vector((q1.x - q0.x, q1.y - q0.y))
            if av.length < 1e-5:
                continue
            av.normalize()
            score = self._bh_safe_dot(move, av)
            if score > best_score:
                best_score = score
                best = (name, axis.copy())
        if best is None:
            return False

        axis_name, world_axis = best
        self.push_undo(context)
        mw3 = obj.matrix_world.to_3x3()
        try:
            imw3 = mw3.inverted()
        except Exception:
            imw3 = Matrix.Identity(3)
        local_axis = (imw3 @ world_axis).normalized()
        changed = False

        for i, part in getattr(self, '_vertex_axis_align_handles', []):
            if not (0 <= i < len(bez)):
                continue
            bp = bez[i]
            co = bp['co'].copy()
            handle = bp[part]
            vec = handle - co
            length = vec.length
            if length < 1e-8:
                continue

            # Same behavior as Spine Mode: the controller stays the pivot and
            # the selected handle keeps its length while following the signed
            # mouse-selected axis. ALIGNED also updates the opposite tip.
            bp[part] = co + local_axis * length
            mode = self.point_mode(i)
            if isinstance(mode, str) and mode.upper() == 'ALIGNED':
                other = 'hr' if part == 'hl' else 'hl'
                other_vec = bp[other] - co
                other_len = other_vec.length
                bp[part] = co + local_axis * length
                if other_len > 1e-8:
                    bp[other] = co - local_axis * other_len
                if 0 <= i < len(self.point_modes):
                    self.point_modes[i] = 'ALIGNED'
            changed = True

        if changed:
            # Keep the same immediate refresh behavior as Spine Mode. AUTO
            # neighbors are rebuilt, while the selected explicit handle is not
            # intentionally changed by a separate mouse-drag operation.
            try:
                self.rebuild_auto_handles()
            except Exception:
                pass
            self.apply_deform(context)
            self._vertex_axis_align_done = True
            self._vertex_axis_align = False
            self._vdh_refresh_edit_normals(context)
            self.report({'INFO'}, f"Handle aligned to {axis_name} axis")
            context.area.tag_redraw()
            return True

        self._vertex_axis_align = False
        return False

    def _modal_spine(self, context, event):
        """Modal handler for SPINE_PLACE and SPINE_DEFORM."""
        self._last_modal_event = event
        # One-shot mouse-directed axis alignment. This is consumed here, before
        # Blender's normal keymap can see Alt+X, and is restricted to Spine Deform.
        if (self.tool_mode == 'SPINE_DEFORM' and event.type == 'X'
                and event.value == 'PRESS' and event.alt and not event.ctrl and not event.shift):
            self._spine_axis_align_begin(context, event)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if getattr(self, '_spine_axis_align', False):
            if event.type == 'MOUSEMOVE':
                if self._spine_axis_align_finish(context, event):
                    return {'RUNNING_MODAL'}
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            # Release Alt cancels only the still-waiting gesture. Once the axis
            # was applied, the gesture has already ended.
            if event.type in {'LEFT_ALT', 'RIGHT_ALT'} and event.value == 'RELEASE':
                self._spine_axis_align = False
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            # Keep the waiting gesture modal so Alt+X cannot fall through to Blender.
            if event.type not in {'MOUSEMOVE'}:
                return {'RUNNING_MODAL'}
        # Alt+Q must finish the active Spine operation before Blender can
        # switch the active mesh.  Otherwise the modal operator can keep
        # running against an object that is no longer its original target.
        if (event.type == 'Q' and event.value == 'PRESS' and event.alt
                and not event.ctrl and not event.shift):
            self.finish(context, cancel=False)
            return {'FINISHED'}

        # Lock mesh vertex selection while tool is active
        try:
            self._spine_lock_mesh_selection(context)
        except Exception:
            pass
        # Any non-wheel event ends the continuous Shift+Scroll session.
        if getattr(self, '_align_prop_session', False):
            self._align_prop_session = False
            self._align_prop_cache = None
            if getattr(self, '_align_prop_kdtree_dirty', False):
                self._rebuild_all_kdtree()
                self._align_prop_kdtree_dirty = False

        # Blender-style numeric input for active Rotate/Scale transforms
        # (shared by Edit Place and Deform).
        if getattr(self, '_xform_mode', None) in {'ROTATE', 'SCALE'} and event.value == 'PRESS':
            if event.type in {'ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE','ZERO','NUMPAD_1','NUMPAD_2','NUMPAD_3','NUMPAD_4','NUMPAD_5','NUMPAD_6','NUMPAD_7','NUMPAD_8','NUMPAD_9','NUMPAD_0','PERIOD','NUMPAD_PERIOD','MINUS','NUMPAD_MINUS'}:
                keymap = {'ONE':'1','TWO':'2','THREE':'3','FOUR':'4','FIVE':'5','SIX':'6','SEVEN':'7','EIGHT':'8','NINE':'9','ZERO':'0','NUMPAD_1':'1','NUMPAD_2':'2','NUMPAD_3':'3','NUMPAD_4':'4','NUMPAD_5':'5','NUMPAD_6':'6','NUMPAD_7':'7','NUMPAD_8':'8','NUMPAD_9':'9','NUMPAD_0':'0','PERIOD':'.','NUMPAD_PERIOD':'.','MINUS':'-','NUMPAD_MINUS':'-'}
                self._xform_numeric = getattr(self, '_xform_numeric', '') + keymap.get(event.type, '')
                self._xform_numeric_active = True
                # Apply the typed value immediately. Blender's transform is live
                # while numeric input is being entered; waiting for MOUSEMOVE made
                # R/S appear to do nothing when the user only typed a value.
                self._spine_update_xform(context, event)
                if self.tool_mode == 'SPINE_DEFORM':
                    self._spine_apply(context)
                else:
                    try:
                        self.spine_points = [bp['co'].copy() for bp in (getattr(self, 'bez', None) or [])]
                        self._spine_store_active_chain()
                    except Exception:
                        pass
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type == 'BACK_SPACE' and getattr(self, '_xform_numeric_active', False):
                self._xform_numeric = self._xform_numeric[:-1]
                if self._xform_numeric:
                    self._spine_update_xform(context, event)
                    if self.tool_mode == 'SPINE_DEFORM':
                        self._spine_apply(context)
                    else:
                        try:
                            self.spine_points = [bp['co'].copy() for bp in (getattr(self, 'bez', None) or [])]
                            self._spine_store_active_chain()
                        except Exception:
                            pass
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # Transform-specific confirm/cancel must run before the normal Spine
        # Enter/Esc handlers, otherwise Enter rebinds the chain or exits Deform.
        if getattr(self, '_xform_mode', None) in {'ROTATE', 'SCALE'}:
            if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                self.dragging = False
                self._xform_mode = None
                self.constraint_axis = None
                self._xform_start = None
                self._xform_keys = None
                self._xform_numeric_active = False
                self._xform_numeric = ''
                try:
                    if self.tool_mode == 'SPINE_DEFORM':
                        self._spine_apply(context)
                    else:
                        self._spine_store_active_chain()
                except Exception:
                    pass
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type == 'ESC' and event.value == 'PRESS':
                try:
                    self._spine_undo(context)
                except Exception:
                    pass
                self.dragging = False
                self._xform_mode = None
                self.constraint_axis = None
                self._xform_start = None
                self._xform_keys = None
                self._xform_numeric_active = False
                self._xform_numeric = ''
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            # X/Y/Z are constraints for both Edit Place and Deform. This must be
            # handled while _xform_mode is active; the old Deform grab-only branch
            # intentionally excluded transform mode, so constraints never fired.
            if (event.type in {'X', 'Y', 'Z'} and event.value == 'PRESS'
                    and not event.ctrl and not event.alt):
                if event.shift:
                    plane = {'X': 'YZ', 'Y': 'XZ', 'Z': 'XY'}[event.type]
                    self.constraint_axis = (None if getattr(self, 'constraint_axis', None) == plane else plane)
                else:
                    self.constraint_axis = (None if getattr(self, 'constraint_axis', None) == event.type else event.type)
                self._spine_update_xform(context, event)
                if self.tool_mode == 'SPINE_DEFORM':
                    self._spine_apply(context)
                else:
                    try:
                        self.spine_points = [bp['co'].copy() for bp in (getattr(self, 'bez', None) or [])]
                        self._spine_store_active_chain()
                    except Exception:
                        pass
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # Confirm / cancel
        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if self.tool_mode == 'SPINE_PLACE':
                if event.alt and not event.shift:
                    # Alt+Enter only meaningful from deform; ignore in place
                    return {'RUNNING_MODAL'}
                if event.shift and not event.alt:
                    # Shift+Enter: close current chain, start next (initial place + edit place)
                    if self._spine_close_chain(context):
                        return {'RUNNING_MODAL'}
                    return {'RUNNING_MODAL'}
                # Enter: bind all / rebind preserve when editing
                if getattr(self, '_spine_edit_place', False):
                    if self._spine_rebind_preserve(context):
                        return {'RUNNING_MODAL'}
                    return {'RUNNING_MODAL'}
                if self._spine_bind(context):
                    return {'RUNNING_MODAL'}
                return {'RUNNING_MODAL'}
            # DEFORM
            if event.alt and not event.shift:
                # Alt+Enter → edit place (add controllers)
                if self._spine_enter_edit_place(context):
                    return {'RUNNING_MODAL'}
                return {'RUNNING_MODAL'}
            # Enter: confirm finish
            self.finish(context, cancel=False)
            return {'FINISHED'}
        if event.type == 'ESC' and event.value == 'PRESS':
            if getattr(self, '_mirror_pending', False):
                self._mirror_pending = False
                self.report({'INFO'}, "Mirror cancelled")
                return {'RUNNING_MODAL'}
            self.finish(context, cancel=True)
            return {'CANCELLED'}

        # Mirror axis confirm (right after Ctrl+M)
        if getattr(self, '_mirror_pending', False) and event.value == 'PRESS' and event.type in {'X', 'Y', 'Z'}:
            self._mirror_pending = False
            try:
                self._spine_mirror_active_chain(context, axis=event.type, pivot=None)
            except Exception as e:
                self.report({'ERROR'}, f"Mirror error: {e}")
            return {'RUNNING_MODAL'}

        # Undo / redo
        if event.type == 'Z' and event.value == 'PRESS' and event.ctrl:
            if event.shift:
                self._spine_redo(context)
            else:
                self._spine_undo(context)
            return {'RUNNING_MODAL'}

        # E: uniformize transverse ring thickness using the average
        # current ring radius. This is intentionally Spine-only; Vertex Mode
        # keeps its existing Shift+R smooth-type behavior.
        if event.type == 'E' and event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            if self.tool_mode == 'SPINE_DEFORM':
                self._spine_push_undo(context)
                self._spine_uniformize_thickness(context, active_only=True)
                return {'RUNNING_MODAL'}

        # Q: circularize transverse edge rings, preserving each ring's
        # own thickness rather than forcing one global tube radius.
        if event.type == 'Q' and event.value == 'PRESS' and not event.ctrl and not event.alt:
            if self.tool_mode == 'SPINE_DEFORM':
                self._spine_push_undo(context)
                self._spine_circularize_rings_to_axis(context, active_only=True)
                return {'RUNNING_MODAL'}

        if event.type == 'W' and event.value == 'PRESS' and not event.ctrl and not event.alt:
            if self.tool_mode == 'SPINE_DEFORM':
                self._spine_push_undo(context)
                self._spine_align_longitudinal_loops_to_axis(context, active_only=True)
                return {'RUNNING_MODAL'}

        # Shift+L: Straighten Tube popup (axis)  |  Ctrl+Shift+L: all chains
        if event.type == 'L' and event.value == 'PRESS' and event.shift and not event.alt:
            if self.tool_mode == 'SPINE_DEFORM':
                all_chains = bool(event.ctrl)
                def draw_straighten_popup(menu, _ctx):
                    layout = menu.layout
                    layout.label(text="Straighten Tube Axis")
                    for ax, label in (
                        ('FREE', 'Free (curve direction)'),
                        ('X', 'X Axis'),
                        ('Y', 'Y Axis'),
                        ('Z', 'Z Axis'),
                    ):
                        op = layout.operator("mesh.vdh_straighten_tube", text=label)
                        op.axis = ax
                        op.active_only = not all_chains
                        op.circularize = True
                        op.even_spacing = True
                title = "Straighten Tube (All)" if all_chains else "Straighten Tube"
                context.window_manager.popup_menu(draw_straighten_popup, title=title)
                return {'RUNNING_MODAL'}
        if event.type == 'L' and event.value == 'PRESS' and not event.ctrl and not event.alt:
            if self.tool_mode == 'SPINE_DEFORM':
                chains = getattr(self, 'spine_chains', None) or []
                ci = int(getattr(self, 'active_chain', 0) or 0)
                hit = self.pick_handle(context, event, pixel_dist=40.0)
                if hit is not None and len(hit) == 3:
                    ci = hit[0]
                if chains and 0 <= ci < len(chains):
                    n = len(chains[ci].get('bez') or [])
                    keys = {(ci, i, 'co') for i in range(n)}
                    self.selected = self._spine_norm_selected()
                    if event.shift:
                        self.selected |= keys
                    else:
                        self.selected = keys
                    try:
                        self._spine_store_active_chain()
                    except Exception:
                        pass
                    self.active_chain = ci
                    try:
                        self._spine_load_active_chain()
                    except Exception:
                        pass
                    self.active_handle = 0 if n else None
                    self.active_bez_part = 'co'
                elif getattr(self, 'bez', None):
                    n = len(self.bez)
                    keys = {(0, i, 'co') for i in range(n)}
                    self.selected = (self._spine_norm_selected() | keys) if event.shift else keys
                    self.active_handle = 0
                    self.active_bez_part = 'co'
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # Alt+R: Reset influence radius of selected Spine controllers to their
        # bind-time defaults. This is intentionally different from Ctrl+R, which
        # resets the controller curve pose.
        if (event.type == 'R' and event.value == 'PRESS' and event.alt
                and not event.ctrl and not event.shift
                and self.tool_mode == 'SPINE_DEFORM'):
            self._spine_reset_influence_radius(context)
            return {'RUNNING_MODAL'}

        # Ctrl+R: Reset active chain | Ctrl+Alt+R: Reset all
        if event.type == 'R' and event.value == 'PRESS' and event.ctrl and not event.shift:
            if self.tool_mode == 'SPINE_DEFORM':
                if event.alt:
                    self._spine_reset_controllers(context, active_only=False)
                else:
                    self._spine_reset_controllers(context, active_only=True)
            return {'RUNNING_MODAL'}

        # P: Place in Volume — toggle snap to VOLUME / restore previous
        if event.type == 'P' and event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            self._spine_toggle_place_in_volume(context)
            return {'RUNNING_MODAL'}

        # Spine Deform shortcuts:
        #   [ / ]          = controller / handle display size
        #   Shift + [ / ]  = influence radius (original behavior)
        #   Alt + [ / ]    = Influence Overlay marker size
        # Keep these three functions strictly separate.
        if event.type in {'LEFT_BRACKET', 'RIGHT_BRACKET'} and event.value == 'PRESS':
            if getattr(self, '_attr_mode', None) and not event.shift and not event.ctrl and not event.alt:
                step = -1 if event.type == 'LEFT_BRACKET' else 1
                self._spine_cycle_attr_interp(context, step)
                return {'RUNNING_MODAL'}
            # Alt+[ / ] = overlay marker size ONLY.
            if (event.alt and not event.ctrl and not event.shift
                    and getattr(self, 'tool_mode', None) == 'SPINE_DEFORM'
                    and bool(getattr(self, '_show_influence', False))):
                marker_size = float(getattr(self, '_influence_overlay_size', 4.0) or 4.0)
                marker_size *= 0.8 if event.type == 'LEFT_BRACKET' else 1.25
                self._influence_overlay_size = max(1.0, min(20.0, marker_size))
                context.area.tag_redraw()
                self.report({'INFO'}, f"Weight marker size: {self._influence_overlay_size:.1f}px")
                return {'RUNNING_MODAL'}
            # Shift+[ / ] = influence radius ONLY. This remains active even
            # while the Influence Overlay is visible.
            if (event.shift and not event.ctrl and not event.alt
                    and getattr(self, 'tool_mode', None) == 'SPINE_DEFORM'):
                delta = -1.0 if event.type == 'LEFT_BRACKET' else 1.0
                self._spine_adjust_influence(context, delta)
                self._influence_overlay_cache = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            ds = float(getattr(self, 'display_scale', 1.0) or 1.0)
            if event.type == 'LEFT_BRACKET':
                ds *= 0.8
            else:
                ds *= 1.25
            self.display_scale = max(0.05, min(5.0, ds))
            context.area.tag_redraw()
            self.report({'INFO'}, f"Handle size: {self.display_scale:.2f}x")
            return {'RUNNING_MODAL'}

        # I: toggle influence overlay (group × envelope on all chains)
        if event.type == 'I' and event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            self._show_influence = not bool(getattr(self, '_show_influence', False))
            # Rebuild the overlay data on the next draw after toggling.
            self._influence_overlay_cache = None
            if self._show_influence:
                try:
                    self._spine_ensure_vg_names()
                except Exception:
                    pass
                n_bound = 0
                for ch in (getattr(self, 'spine_chains', None) or []):
                    n_bound += len(ch.get('bind') or [])
                if n_bound <= 0:
                    self.report({'INFO'}, "Weight overlay: ON  — bind first (Enter) to see weights")
                else:
                    self.report({'INFO'}, f"Weight overlay: ON  ({n_bound} verts)")
            else:
                self.report({'INFO'}, "Weight overlay: OFF")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # N: toggle In Front (active chain)  |  Shift+N: all chains
        if event.type == 'N' and event.value == 'PRESS' and not event.ctrl and not event.alt:
            self._spine_toggle_in_front(context, all_chains=bool(event.shift))
            return {'RUNNING_MODAL'}

        # Shift+F: influence falloff popup (select type)
        if event.type == 'F' and event.value == 'PRESS' and event.shift and not event.ctrl and not event.alt:
            if self.tool_mode == 'SPINE_DEFORM':
                def draw_falloff_popup(menu, _ctx):
                    layout = menu.layout
                    for mid in _INFLUENCE_FALLOFF_ORDER:
                        label = mid.replace('_', ' ').title()
                        op = layout.operator("mesh.vdh_influence_falloff", text=label)
                        op.mode = mid
                context.window_manager.popup_menu(draw_falloff_popup, title="Influence Falloff")
            return {'RUNNING_MODAL'}


        # Alt+Wheel: resize the Influence Overlay markers in Spine Deform.
        # This is a display-only control; it does not alter Radius/weights and
        # therefore does not create an Undo step.  Keep it separate from the
        # Shift+Wheel Radius shortcut below.
        if (getattr(self, 'tool_mode', None) == 'SPINE_DEFORM'
                and event.alt and not event.ctrl and not event.shift
                and event.value != 'RELEASE'):
            et = event.type
            if et in {'WHEELUPMOUSE', 'WHEELINMOUSE'}:
                marker_size = float(getattr(self, '_influence_overlay_size', 4.0) or 4.0)
                self._influence_overlay_size = max(1.0, min(20.0, marker_size * 1.18))
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if et in {'WHEELDOWNMOUSE', 'WHEELOUTMOUSE'}:
                marker_size = float(getattr(self, '_influence_overlay_size', 4.0) or 4.0)
                self._influence_overlay_size = max(1.0, min(20.0, marker_size / 1.18))
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # Shift+Wheel: influence radius in Spine Deform only.
        # OS/trackpad may report vertical wheel as WHEELIN/OUT.
        if (getattr(self, 'tool_mode', None) == 'SPINE_DEFORM'
                and event.shift and not event.ctrl and not event.alt
                and event.value != 'RELEASE'):
            et = event.type
            if et in {'WHEELUPMOUSE', 'WHEELINMOUSE'}:
                self._spine_adjust_influence(context, 1.0)
                self._influence_overlay_cache = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if et in {'WHEELDOWNMOUSE', 'WHEELOUTMOUSE'}:
                self._spine_adjust_influence(context, -1.0)
                self._influence_overlay_cache = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # M: Set as Initial State — Edit Place only (rewrites Ctrl+R baseline)
        if event.type == 'M' and event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            if (
                self.tool_mode == 'SPINE_PLACE'
                and getattr(self, '_spine_edit_place', False)
            ):
                def draw_initial_popup(menu, _ctx):
                    layout = menu.layout
                    layout.label(text="Use current chain pose as reset baseline")
                    layout.operator("mesh.vdh_set_initial", text="Set as Initial State", icon='FILE_TICK')
                context.window_manager.popup_menu(draw_initial_popup, title="Initial State")
            return {'RUNNING_MODAL'}

        # Ctrl+M: Mirror in place (all spine modes)
        # Ctrl+Shift+M: Duplicate + Mirror — Spine Place / Edit Place
        if event.type == 'M' and event.value == 'PRESS' and event.ctrl and not event.alt:
            dup = bool(event.shift)
            if dup:
                # Duplicate + Mirror is available in Spine Place as well as Edit Place.
                # This is important while building a new chain: Ctrl+Shift+M must
                # create a mirrored copy, not fall back to an in-place mirror.
                if self.tool_mode != 'SPINE_PLACE':
                    return {'RUNNING_MODAL'}
            title = "Duplicate + Mirror" if dup else "Mirror (in place)"
            def draw_mirror_popup(menu, _ctx):
                col = menu.layout.column(align=True)
                for space, label in (
                    ('LOCAL', 'Local'),
                    ('CURSOR', 'Cursor'),
                    ('WORLD', 'World'),
                ):
                    for ax in ('X', 'Y', 'Z'):
                        op = col.operator("mesh.vdh_mirror_chain", text=f"{label} {ax}")
                        op.axis = ax
                        op.space = space
                        op.duplicate = dup
            context.window_manager.popup_menu(draw_mirror_popup, title=title)
            return {'RUNNING_MODAL'}

        # Shift+D: Duplicate — Edit Place only
        if event.type == 'D' and event.value == 'PRESS' and event.shift and not event.ctrl and not event.alt:
            if (
                self.tool_mode == 'SPINE_PLACE'
                and getattr(self, '_spine_edit_place', False)
            ):
                self._spine_duplicate_active_chain(context, mirror=False, event=event)
            return {'RUNNING_MODAL'}


        # Allow UI (snap / proportional / headers) — mouse may be over any region
        if self._spine_event_in_ui(context, event):
            return {'PASS_THROUGH'}
        region = context.region
        if region is not None and region.type in {
            'HEADER', 'TOOL_HEADER', 'TOOLS', 'UI', 'HUD', 'NAVIGATION_BAR',
        }:
            return {'PASS_THROUGH'}

        # --- Placement mode ---
        if self.tool_mode == 'SPINE_PLACE':
            # Ctrl+Alt+C: Clear cache + wipe all chains (Place/Edit only)
            if event.type == 'C' and event.value == 'PRESS' and event.ctrl and event.alt and not event.shift:
                def draw_cache_popup(menu, _ctx):
                    layout = menu.layout
                    layout.label(text="Clear Spine Cache")
                    layout.label(text="Removes cache and all chains")
                    op = layout.operator("mesh.vdh_clear_cache", text="This Object Only")
                    op.scope = 'THIS'
                    op = layout.operator("mesh.vdh_clear_cache", text="All Meshes")
                    op.scope = 'ALL'
                context.window_manager.popup_menu(draw_cache_popup, title="Clear Spine Cache")
                return {'RUNNING_MODAL'}

            # V: handle type popup (Edit Place — same as Deform)
            if event.type == 'V' and event.value == 'PRESS' and not event.ctrl and not event.alt:
                if getattr(self, 'bez', None) and len(self.bez) >= 2:
                    def draw_handle_popup(menu, _context):
                        layout = menu.layout
                        for mid, name in (
                            ('AUTO', 'Automatic'),
                            ('ALIGNED', 'Aligned'),
                            ('FREE', 'Free'),
                        ):
                            op = layout.operator("mesh.vdh_handle_type", text=name)
                            op.mode = mid
                    context.window_manager.popup_menu(draw_handle_popup, title="Handle Type")
                return {'RUNNING_MODAL'}

            # L: select all controllers on active chain
            if event.type == 'L' and event.value == 'PRESS' and not event.ctrl and not event.alt:
                pts = getattr(self, 'spine_points', []) or []
                if pts:
                    if event.shift:
                        self.selected |= {(i, 'co') for i in range(len(pts))}
                    else:
                        self.selected = {(i, 'co') for i in range(len(pts))}
                    self.active_handle = 0
                    self.active_bez_part = 'co'
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # R/S axis constraint: after starting Rotate/Scale, X/Y/Z (and Shift+X/Y/Z)
            # behave like Blender transform constraints. A repeated key toggles the constraint.
            if (getattr(self, '_xform_mode', None) in {'ROTATE', 'SCALE'}
                    and event.type in {'X', 'Y', 'Z'} and event.value == 'PRESS'
                    and not event.ctrl and not event.alt):
                if event.shift:
                    plane = {'X': 'YZ', 'Y': 'XZ', 'Z': 'XY'}[event.type]
                    self.constraint_axis = None if getattr(self, 'constraint_axis', None) == plane else plane
                else:
                    self.constraint_axis = None if getattr(self, 'constraint_axis', None) == event.type else event.type
                self._spine_update_xform(context, event)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # G-grab axis constraints in Edit Place: X/Y/Z (and Shift+X/Y/Z)
            # must behave like Blender Grab constraints.  This has to run BEFORE
            # the delete handler below, otherwise X would be interpreted as delete.
            if (getattr(self, '_spine_edit_place', False) and self.dragging
                    and event.type in {'X', 'Y', 'Z'} and event.value == 'PRESS'
                    and not event.ctrl and not event.alt):
                if event.shift:
                    plane = {'X': 'YZ', 'Y': 'XZ', 'Z': 'XY'}[event.type]
                    self.constraint_axis = (
                        None if getattr(self, 'constraint_axis', None) == plane else plane
                    )
                else:
                    self.constraint_axis = (
                        None if getattr(self, 'constraint_axis', None) == event.type
                        else event.type
                    )
                # Re-evaluate the current mouse position immediately under the new
                # constraint, exactly like Blender G -> X/Y/Z.
                self._spine_update_place_drag(context, event)
                try:
                    if getattr(self, 'bez', None):
                        if getattr(self, 'active_bez_part', 'co') == 'co':
                            self.rebuild_auto_handles()
                        self.spine_points = [bp['co'].copy() for bp in self.bez]
                        self._spine_store_active_chain()
                        if getattr(self, '_mirror_drag', None):
                            self._spine_sync_mirror_chain_live(context)
                except Exception:
                    pass
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Delete selected controllers (and sync bez in Edit Place).
            # While R/S transform is active, X/Y/Z are Blender-style axis constraints,
            # so they must NOT be treated as delete commands.
            if (event.type in {'DEL', 'X', 'BACK_SPACE'} and event.value == 'PRESS'
                    and not event.ctrl and not event.alt
                    and not getattr(self, '_xform_mode', None)):
                if event.type == 'X' and event.shift:
                    return {'PASS_THROUGH'}
                sel = []
                for item in (self.selected or set()):
                    if len(item) == 3:
                        _ci, i, p = item
                    else:
                        i, p = item[0], item[1]
                    if p == 'co':
                        sel.append(i)
                sel = sorted(set(sel), reverse=True)
                if not sel and self.active_handle is not None and getattr(self, 'active_bez_part', 'co') == 'co':
                    sel = [self.active_handle]
                if sel and (len(getattr(self, 'spine_points', []) or []) > 0 or (getattr(self, 'bez', None) and len(self.bez) > 0)):
                    self._spine_push_undo(context)
                    oids = list(getattr(self, '_spine_origin_ids', None) or [])
                    for hit in sel:
                        if 0 <= hit < len(getattr(self, 'spine_points', []) or []):
                            self.spine_points.pop(hit)
                        if hit < len(oids):
                            oids.pop(hit)
                        # Remove from live bez + modes in Edit Place
                        if getattr(self, 'bez', None) and 0 <= hit < len(self.bez):
                            self.bez.pop(hit)
                            if getattr(self, 'point_modes', None) and hit < len(self.point_modes):
                                self.point_modes.pop(hit)
                            if getattr(self, 'spine_tilt', None) and hit < len(self.spine_tilt):
                                self.spine_tilt.pop(hit)
                            if getattr(self, 'spine_radius', None) and hit < len(self.spine_radius):
                                self.spine_radius.pop(hit)
                    self._spine_origin_ids = oids
                    # Immediately refresh any remaining AUTO handles after deletion.
                    # FREE/ALIGNED handles stay exactly as they are.
                    try:
                        if getattr(self, 'bez', None):
                            self.rebuild_auto_handles(interior=True)
                    except Exception:
                        pass
                    # Full chain delete → remove chain, activate previous/last
                    pruned = False
                    if (len(getattr(self, 'bez', None) or []) < 2
                            and len(getattr(self, 'spine_points', None) or []) < 2):
                        try:
                            pruned = bool(self._spine_prune_empty_active_chain(context))
                        except Exception:
                            pruned = False
                    if not pruned:
                        try:
                            self._spine_store_active_chain()
                        except Exception:
                            pass
                        self.selected = set()
                        self.active_handle = None
                        self.active_bez_part = 'co'
                        self._spine_last_add_idx = (
                            len(self.spine_points) - 1 if self.spine_points else None
                        )
                    # Controller count/curve state changed: invalidate all Vertex-Mirror
                    # caches so the next G/R/L/S/F/Align operation rebuilds from the
                    # current mesh/rest state instead of using the pre-delete pairing.
                    self._vertex_mirror_invalidate_state(rebuild=True)
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}


            # Fresh place = initial tool open OR after Shift+Enter new chain
            placing_fresh = (
                not getattr(self, '_spine_edit_place', False)
                or getattr(self, '_spine_placing_new_chain', False)
            )

            # R / S: rotate or scale selected controllers in Edit Place.
            # Keep G on the existing grab path; R/S use the same Blender-style
            # pivot/orientation/axis machinery as Deform mode.
            if (getattr(self, '_spine_edit_place', False)
                    and event.type in {'R', 'S'} and event.value == 'PRESS'
                    and not event.ctrl and not event.alt):
                self.selected = self._spine_norm_selected()
                keys = []
                for k in (self.selected or set()):
                    if len(k) == 3:
                        ci, i, part = int(k[0]), int(k[1]), k[2]
                    else:
                        ci, i, part = int(getattr(self, 'active_chain', 0) or 0), int(k[0]), k[1]
                    if part == 'co':
                        keys.append((ci, i, 'co'))
                if not keys and self.active_handle is not None:
                    keys = [(int(getattr(self, 'active_chain', 0) or 0),
                             int(self.active_handle), 'co')]
                if keys:
                    self._spine_push_undo(context)
                    self._spine_start_xform(
                        context, event, 'ROTATE' if event.type == 'R' else 'SCALE', keys
                    )
                    return {'RUNNING_MODAL'}
                return {'RUNNING_MODAL'}

            # R/S axis constraint is handled here too, because Edit Place has its
            # own modal branch before the shared Deform transform handler.
            if (getattr(self, '_xform_mode', None) in {'ROTATE', 'SCALE'}
                    and event.type in {'X', 'Y', 'Z'} and event.value == 'PRESS'
                    and not event.ctrl and not event.alt):
                if event.shift:
                    plane = {'X': 'YZ', 'Y': 'XZ', 'Z': 'XY'}[event.type]
                    self.constraint_axis = (None if getattr(self, 'constraint_axis', None) == plane
                                            else plane)
                else:
                    self.constraint_axis = (None if getattr(self, 'constraint_axis', None) == event.type
                                            else event.type)
                self._spine_update_xform(context, event)
                try:
                    if getattr(self, 'bez', None):
                        self.rebuild_auto_handles()
                        self.spine_points = [bp['co'].copy() for bp in self.bez]
                        self._spine_store_active_chain()
                        if getattr(self, '_mirror_drag', None):
                            self._spine_sync_mirror_chain_live(context)
                except Exception:
                    pass
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # --- LMB ---
            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                # Confirm ongoing drag: keep selection (Blender-like)
                if self.dragging:
                    self.dragging = False
                    self._pending_place_drag = False
                    self.constraint_axis = None
                    return {'RUNNING_MODAL'}
                if event.alt:
                    # Edit Place: Alt+click on handle tip → FREE (same as Deform)
                    if getattr(self, '_spine_edit_place', False):
                        ph = self._spine_pick_place_handle(context, event, pixel_dist=18.0)
                        if ph is not None:
                            key, hit, part = ph[0], ph[1], ph[2]
                            if key != 'current':
                                self._spine_activate_place_chain(key)
                            if part in ('hl', 'hr') and getattr(self, 'bez', None) and 0 <= hit < len(self.bez):
                                self._spine_push_undo(context)
                                n = len(self.bez)
                                if not hasattr(self, 'point_modes') or len(self.point_modes) != n:
                                    self.point_modes = ['AUTO'] * n
                                self.point_modes[hit] = 'FREE'
                                self.select_only(hit, part)
                                try:
                                    self._spine_store_active_chain()
                                except Exception:
                                    pass
                                self.report({'INFO'}, f"Handle {hit}: FREE")
                                context.area.tag_redraw()
                                return {'RUNNING_MODAL'}
                    # Alt+click on controller → remove
                    other = self._spine_pick_any_chain_controller(context, event, pixel_dist=22.0)
                    if other is not None:
                        key, hit = other
                        if key != 'current':
                            self._spine_activate_place_chain(key)
                        if 0 <= hit < len(self.spine_points):
                            self._spine_push_undo(context)
                            self.spine_points.pop(hit)
                            oids = list(getattr(self, '_spine_origin_ids', None) or [])
                            if hit < len(oids):
                                oids.pop(hit)
                            self._spine_origin_ids = oids
                            if getattr(self, 'bez', None) and 0 <= hit < len(self.bez):
                                self.bez.pop(hit)
                                if getattr(self, 'point_modes', None) and hit < len(self.point_modes):
                                    self.point_modes.pop(hit)
                                if getattr(self, 'spine_tilt', None) and hit < len(self.spine_tilt):
                                    self.spine_tilt.pop(hit)
                                if getattr(self, 'spine_radius', None) and hit < len(self.spine_radius):
                                    self.spine_radius.pop(hit)
                            # Immediately refresh remaining AUTO handles after deletion.
                            # FREE/ALIGNED handles remain untouched.
                            try:
                                if getattr(self, 'bez', None):
                                    self.rebuild_auto_handles(interior=True)
                            except Exception:
                                pass
                            pruned = False
                            if (len(getattr(self, 'bez', None) or []) < 2
                                    and len(getattr(self, 'spine_points', None) or []) < 2):
                                try:
                                    pruned = bool(self._spine_prune_empty_active_chain(context))
                                except Exception:
                                    pruned = False
                            if not pruned:
                                try:
                                    self._spine_store_active_chain()
                                except Exception:
                                    pass
                                self.selected = set()
                                self.active_handle = None
                            context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                # --- Fresh place (first open / Shift+Enter): click = add, only pick OWN points ---
                if placing_fresh:
                    if getattr(self, '_spine_edit_place', False):
                        ph = self._spine_pick_place_handle(context, event, pixel_dist=20.0)
                        if ph is not None:
                            key, hit, part = ph[0], ph[1], ph[2]
                            if key != 'current':
                                self._spine_activate_place_chain(key)
                            if event.shift:
                                self.select_toggle(hit, part)
                            else:
                                self.select_only(hit, part)
                            # Click = select; drag after small move
                            self._pending_place_drag = True
                            self._pending_place_idx = hit
                            self._pending_place_part = part
                            self._pending_place_xy = (event.mouse_region_x, event.mouse_region_y)
                            self.dragging = False
                            context.area.tag_redraw()
                            return {'RUNNING_MODAL'}
                    hit = self._spine_pick_controller(context, event, pixel_dist=20.0)
                    if hit is not None:
                        if event.shift:
                            self.select_toggle(hit, 'co')
                        else:
                            self.select_only(hit, 'co')
                        self._pending_place_drag = True
                        self._pending_place_idx = hit
                        self._pending_place_part = 'co'
                        self._pending_place_xy = (event.mouse_region_x, event.mouse_region_y)
                        self.dragging = False
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    # Empty space → add controller immediately (any count)
                    if not event.shift:
                        local = self._spine_mouse_local(context, event)
                        if local is not None:
                            self._spine_push_undo(context)
                            self.spine_points.append(local.copy())
                            oids = list(getattr(self, '_spine_origin_ids', None) or [])
                            oids.append(None if getattr(self, '_spine_edit_place', False) else len(oids))
                            self._spine_origin_ids = oids
                            self._spine_last_add_idx = len(self.spine_points) - 1
                            self.select_only(self._spine_last_add_idx, 'co')
                            context.area.tag_redraw()
                            return {'RUNNING_MODAL'}
                    # shift+empty → box select
                    self.dragging = False
                    self._pending_place_drag = False
                    self.box_selecting = True
                    self.box_start = (event.mouse_region_x, event.mouse_region_y)
                    self.box_end = self.box_start
                    self.box_handles_only = bool(event.ctrl and event.shift)
                    return {'RUNNING_MODAL'}

                # --- Edit existing chain(s): click = select, move = drag ---
                if getattr(self, '_spine_edit_place', False):
                    ph = self._spine_pick_place_handle(context, event, pixel_dist=20.0)
                    if ph is not None:
                        key, hit, part = ph[0], ph[1], ph[2]
                        if key != 'current':
                            self._spine_activate_place_chain(key)
                        if event.shift:
                            self.select_toggle(hit, part)
                        else:
                            self.select_only(hit, part)
                        self._pending_place_drag = True
                        self._pending_place_idx = hit
                        self._pending_place_part = part
                        self._pending_place_xy = (event.mouse_region_x, event.mouse_region_y)
                        self.dragging = False
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                other = self._spine_pick_any_chain_controller(context, event, pixel_dist=24.0)
                if other is not None:
                    key, hit = other
                    if key != 'current':
                        self._spine_activate_place_chain(key)
                        context.area.tag_redraw()
                    if event.shift:
                        self.select_toggle(hit, 'co')
                    else:
                        self.select_only(hit, 'co')
                    self._pending_place_drag = True
                    self._pending_place_idx = hit
                    self._pending_place_part = 'co'
                    self._pending_place_xy = (event.mouse_region_x, event.mouse_region_y)
                    self.dragging = False
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                if event.shift and len(self.spine_points) >= 2:
                    seg = self._spine_pick_segment(context, event, pixel_dist=14.0)
                    if seg is not None:
                        seg_i, local_on = seg
                        self._spine_push_undo(context)
                        insert_at = seg_i + 1
                        self.spine_points.insert(insert_at, local_on)
                        oids = list(getattr(self, '_spine_origin_ids', None) or [])
                        if len(oids) != len(self.spine_points) - 1:
                            oids = list(range(len(self.spine_points) - 1))
                        oids.insert(insert_at, None)
                        self._spine_origin_ids = oids
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}

                self.dragging = False
                self.box_selecting = True
                self.box_start = (event.mouse_region_x, event.mouse_region_y)
                self.box_end = self.box_start
                # Ctrl+Shift+drag = handle tips only (same as Deform)
                self.box_handles_only = bool(event.ctrl and event.shift)
                return {'RUNNING_MODAL'}

            # Live Rotate/Scale update for Edit Place. This must run before the
            # normal grab/pending-drag path so R/S actually owns the mouse move.
            if (getattr(self, '_xform_mode', None) in {'ROTATE', 'SCALE'}
                    and event.type == 'MOUSEMOVE'):
                self._spine_update_xform(context, event)
                try:
                    if getattr(self, 'bez', None):
                        self.spine_points = [bp['co'].copy() for bp in self.bez]
                        self._spine_store_active_chain()
                        # Keep AUTO handles live after transform.
                        self.rebuild_auto_handles()
                        self._spine_store_active_chain()
                except Exception:
                    pass
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Pending click→drag in Place / Edit Place
            if getattr(self, '_pending_place_drag', False):
                if event.type == 'MOUSEMOVE':
                    x0, y0 = getattr(self, '_pending_place_xy', (0, 0))
                    dx = event.mouse_region_x - x0
                    dy = event.mouse_region_y - y0
                    if dx * dx + dy * dy >= 9:
                        self._pending_place_drag = False
                        self._spine_push_undo(context)
                        self._spine_start_place_drag(
                            context, event,
                            getattr(self, '_pending_place_idx', 0),
                            part=getattr(self, '_pending_place_part', 'co'),
                        )
                        self._spine_update_place_drag(context, event)
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                    self._pending_place_drag = False
                    self.dragging = False
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                self._pending_place_drag = False
                if getattr(self, '_xform_mode', None) in {'ROTATE', 'SCALE'}:
                    self._xform_mode = None
                    self.constraint_axis = None
                    self._xform_start = None
                    self._xform_keys = None
                    try:
                        self._spine_store_active_chain()
                    except Exception:
                        pass
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if getattr(self, 'box_selecting', False):
                    self.box_end = (event.mouse_region_x, event.mouse_region_y)
                    x0, y0 = self.box_start or (0, 0)
                    x1, y1 = self.box_end
                    self.box_selecting = False
                    if abs(x1 - x0) < 4 and abs(y1 - y0) < 4:
                        self.selected = set()
                        self.active_handle = None
                        # Fresh place already adds on PRESS; here only for safety
                        placing_fresh = (
                            not getattr(self, '_spine_edit_place', False)
                            or getattr(self, '_spine_placing_new_chain', False)
                        )
                        if placing_fresh and not event.shift:
                            local = self._spine_mouse_local(context, event)
                            if local is not None:
                                self._spine_push_undo(context)
                                self.spine_points.append(local.copy())
                                oids = list(getattr(self, '_spine_origin_ids', None) or [])
                                oids.append(None if getattr(self, '_spine_edit_place', False) else len(oids))
                                self._spine_origin_ids = oids
                                self._spine_last_add_idx = len(self.spine_points) - 1
                                self.select_only(self._spine_last_add_idx, 'co')
                    else:
                        self._spine_finish_place_box_select(context, event)
                    self.dragging = False
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                self.dragging = False
                return {'RUNNING_MODAL'}

            if getattr(self, 'box_selecting', False):
                self.dragging = False
                if event.type == 'MOUSEMOVE':
                    self.box_end = (event.mouse_region_x, event.mouse_region_y)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                    self.box_selecting = False
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            if event.type == 'MOUSEMOVE' and self.dragging and self.active_handle is not None:
                self._spine_update_place_drag(context, event)
                # Edit Place must keep AUTO handles synchronized with the live
                # controller position.  Do this at the modal boundary as well as
                # inside the drag routine, so the active chain cannot display stale
                # tips after a controller move / mirror update.
                try:
                    if getattr(self, '_spine_edit_place', False) and getattr(self, 'bez', None):
                        if getattr(self, 'active_bez_part', 'co') == 'co':
                            self.rebuild_auto_handles()
                        self.spine_points = [bp['co'].copy() for bp in self.bez]
                        self._spine_store_active_chain()
                        if getattr(self, '_mirror_drag', None):
                            self._spine_sync_mirror_chain_live(context)
                except Exception:
                    pass
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # G: grab selected (controllers or handle tips)
            if event.type == 'G' and event.value == 'PRESS' and not event.ctrl and not event.alt:
                part = getattr(self, 'active_bez_part', 'co') or 'co'
                idx = self.active_handle
                if idx is None and self.selected:
                    item = next(iter(self.selected))
                    if len(item) == 3:
                        idx, part = item[1], item[2]
                    else:
                        idx, part = item[0], item[1]
                if idx is not None and (
                    (getattr(self, 'spine_points', None) and 0 <= idx < len(self.spine_points))
                    or (getattr(self, 'bez', None) and 0 <= idx < len(self.bez))
                ):
                    self.active_handle = idx
                    self.active_bez_part = part
                    self._spine_push_undo(context)
                    self._spine_start_place_drag(context, event, idx, part=part)
                return {'RUNNING_MODAL'}

            # A: select all | Alt+A: deselect
            if event.type == 'A' and event.value == 'PRESS' and not event.ctrl:
                if event.alt:
                    self.selected = set()
                    self.active_handle = None
                else:
                    self.selected = {(i, 'co') for i in range(len(self.spine_points))}
                    if self.spine_points:
                        self.active_handle = 0
                        self.active_bez_part = 'co'
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'RIGHTMOUSE' and event.value == 'PRESS' and (
                self.dragging or getattr(self, '_xform_mode', None)
            ):
                self._spine_undo(context)
                self.dragging = False
                self._xform_mode = None
                self._pending_click_drag = False
                self._pending_place_drag = False
                self.constraint_axis = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Shift+MMB on active/any chain line → insert controller (edit place)
            if event.type == 'MIDDLEMOUSE' and event.value == 'PRESS' and event.shift and not event.alt:
                # Wider hit radius so insert is reliable along the green curve
                if self._spine_insert_controller_on_line(context, event):
                    return {'RUNNING_MODAL'}
                self.report({'INFO'}, "Shift+MMB: hover the green curve to insert")
                return {'RUNNING_MODAL'}

            # Middle mouse etc → pass for navigation
            return {'PASS_THROUGH'}

        # --- Deform mode ---
        if self.tool_mode == 'SPINE_DEFORM':
            # --- Attr modes (Tilt / Shrink) must run BEFORE LMB pick/drag ---
            if getattr(self, '_attr_mode', None):
                if event.type == 'MOUSEMOVE':
                    if self._attr_start_mouse is None or not self._attr_start_values:
                        return {'RUNNING_MODAL'}
                    dx = event.mouse_region_x - self._attr_start_mouse
                    if self._attr_mode == 'TILT':
                        delta = dx * 0.01
                        for i, base in self._attr_start_values.items():
                            if 0 <= i < len(self.spine_tilt):
                                self.spine_tilt[i] = base + delta
                    elif self._attr_mode == 'RADIUS':
                        factor = max(0.05, 1.0 + dx * 0.005)
                        for i, base in self._attr_start_values.items():
                            if 0 <= i < len(self.spine_radius):
                                self.spine_radius[i] = max(0.05, base * factor)
                    self._spine_apply(context)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                    self._attr_mode = None
                    self._attr_start_values = None
                    self._attr_start_mouse = None
                    return {'RUNNING_MODAL'}
                if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                    self._spine_undo(context)
                    self._attr_mode = None
                    self._attr_start_values = None
                    self._attr_start_mouse = None
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                # Block other keys while adjusting (except we'll allow starting a new attr below)
                if event.value == 'PRESS' and event.type not in {
                    'T', 'A', 'LEFT_CTRL', 'RIGHT_CTRL', 'LEFT_ALT', 'RIGHT_ALT',
                    'LEFT_SHIFT', 'RIGHT_SHIFT',
                }:
                    return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                # Confirm ongoing grab/rotate/scale: keep selection (Blender-like)
                if self.dragging or getattr(self, '_xform_mode', None):
                    self.dragging = False
                    self._xform_mode = None
                    self._pending_click_drag = False
                    self.constraint_axis = None
                    return {'RUNNING_MODAL'}
                # Pick controller or Bezier handle tip (generous hit radius)
                hit = self.pick_handle(context, event, pixel_dist=28.0)
                if hit is not None:
                    self.box_selecting = False
                    self._pending_click_drag = False
                    if len(hit) == 3:
                        ci, idx, part = hit
                    else:
                        ci = int(getattr(self, 'active_chain', 0) or 0)
                        idx, part = hit[0], hit[1]
                    if event.shift:
                        self.select_toggle(ci, idx, part)
                        # If still selected after toggle → this is the new active
                        if self._sel_has(idx, part, chain_idx=ci):
                            self.active_chain = ci
                            self.active_handle = idx
                            self.active_bez_part = part
                    else:
                        self.select_only(ci, idx, part)
                        self.active_chain = ci
                        self.active_handle = idx
                        self.active_bez_part = part
                    # Click = select only; drag starts after small mouse move
                    self._pending_click_drag = True
                    self._pending_drag_idx = idx
                    self._pending_drag_part = part
                    self._pending_drag_ci = ci
                    self._pending_drag_xy = (event.mouse_region_x, event.mouse_region_y)
                    self.dragging = False
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                # Empty click: box-select (never start a drag)
                self.dragging = False
                self._pending_click_drag = False
                self._xform_mode = None
                self.box_selecting = True
                self.box_start = (event.mouse_region_x, event.mouse_region_y)
                self.box_end = self.box_start
                self.box_handles_only = bool(event.ctrl and event.shift)
                return {'RUNNING_MODAL'}

            # Pending click→drag: select already done; start drag after threshold
            if getattr(self, '_pending_click_drag', False):
                if event.type == 'MOUSEMOVE':
                    x0, y0 = getattr(self, '_pending_drag_xy', (0, 0))
                    dx = event.mouse_region_x - x0
                    dy = event.mouse_region_y - y0
                    if dx * dx + dy * dy >= 9:  # ~3 px
                        self._pending_click_drag = False
                        self._spine_push_undo(context)
                        self.start_drag(
                            context, event,
                            getattr(self, '_pending_drag_idx', 0),
                            getattr(self, '_pending_drag_part', 'co'),
                        )
                        self.update_drag(context, event)
                        self._spine_apply(context)
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                    # Pure click: keep selection, no move
                    self._pending_click_drag = False
                    self.dragging = False
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            # Box select (must handle before generic LMB release)
            if getattr(self, 'box_selecting', False):
                self.dragging = False
                self._xform_mode = None
                if event.type == 'MOUSEMOVE':
                    self.box_end = (event.mouse_region_x, event.mouse_region_y)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                    x0, y0 = self.box_start or (0, 0)
                    x1, y1 = event.mouse_region_x, event.mouse_region_y
                    self.box_end = (x1, y1)
                    self.box_selecting = False
                    self.dragging = False
                    if abs(x1 - x0) < 5 and abs(y1 - y0) < 5:
                        if not event.shift:
                            self.selected = set()
                            self.active_handle = None
                    else:
                        self._spine_finish_box_select(context, event)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                    self.box_selecting = False
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                self.dragging = False
                self._pending_click_drag = False
                if getattr(self, '_xform_mode', None):
                    self._xform_mode = None
                return {'RUNNING_MODAL'}

            if event.type == 'MOUSEMOVE' and self.dragging and self.active_handle is not None:
                if getattr(self, '_xform_mode', None) in {'ROTATE', 'SCALE'}:
                    self._spine_update_xform(context, event)
                else:
                    self.update_drag(context, event)
                self._spine_apply(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Axis constraints while dragging (grab only)
            if self.dragging and not getattr(self, '_xform_mode', None) and event.value == 'PRESS' and event.type in {'X', 'Y', 'Z'}:
                if event.shift:
                    plane = {'X': 'YZ', 'Y': 'XZ', 'Z': 'XY'}[event.type]
                    self.constraint_axis = None if getattr(self, 'constraint_axis', None) == plane else plane
                else:
                    self.constraint_axis = None if getattr(self, 'constraint_axis', None) == event.type else event.type
                self.update_drag(context, event)
                self._spine_apply(context)
                return {'RUNNING_MODAL'}

            # A: select all controllers (all chains) | Alt+A: deselect
            if event.type == 'A' and event.value == 'PRESS' and not event.ctrl:
                if event.alt:
                    self.selected = set()
                    self.active_handle = None
                else:
                    chains = getattr(self, 'spine_chains', None) or []
                    # Keep active chain data safe — do NOT switch active_chain on Select All
                    try:
                        self._spine_store_active_chain()
                    except Exception:
                        pass
                    if chains:
                        keys = set()
                        for ci, ch in enumerate(chains):
                            for i in range(len(ch.get('bez') or [])):
                                keys.add((ci, i, 'co'))
                        self.selected = keys
                    else:
                        self.selected = {(0, i, 'co') for i in range(len(self.bez or []))}
                    # Keep current active_chain; just ensure an active handle exists on it
                    ac = int(getattr(self, 'active_chain', 0) or 0)
                    if chains and 0 <= ac < len(chains) and (chains[ac].get('bez') or []):
                        self.active_handle = 0
                        self.active_bez_part = 'co'
                    elif self.selected:
                        # fallback only if active chain empty
                        for ci, i, p in self.selected:
                            try:
                                self._spine_store_active_chain()
                            except Exception:
                                pass
                            self.active_chain = ci
                            self.active_handle = i
                            self.active_bez_part = p
                            try:
                                self._spine_load_active_chain()
                            except Exception:
                                pass
                            break
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Ctrl+T = Tilt | Alt+T = Remove Tilt
            # Ctrl+A = Shrink/Inflate | Ctrl+Alt+A = Remove Shrink/Inflate
            if event.type == 'T' and event.value == 'PRESS' and event.ctrl and not event.alt:
                # Commit previous attr mode if any (no undo rollback)
                self._attr_mode = None
                self._attr_start_values = None
                self._attr_mode = 'TILT'
                self._attr_start_mouse = event.mouse_region_x
                idxs = self.selected_point_indices() or (
                    {self.active_handle} if self.active_handle is not None else set(range(len(self.bez)))
                )
                if not hasattr(self, 'spine_tilt') or len(self.spine_tilt) != len(self.bez):
                    self.spine_tilt = [0.0] * len(self.bez)
                self._attr_start_values = {i: self.spine_tilt[i] for i in idxs if i is not None}
                self._spine_push_undo(context)
                self.report({'INFO'}, "Tilt: move mouse  |  LMB confirm  |  RMB cancel")
                return {'RUNNING_MODAL'}

            if event.type == 'A' and event.value == 'PRESS' and event.ctrl and not event.shift:
                if event.alt:
                    # Ctrl+Alt+A = Remove Shrink/Inflate (reset radius to 1)
                    self._attr_mode = None
                    self._attr_start_values = None
                    self._spine_push_undo(context)
                    if not hasattr(self, 'spine_radius') or len(self.spine_radius) != len(self.bez):
                        self.spine_radius = [1.0] * len(self.bez)
                    idxs = self.selected_point_indices() or set(range(len(self.bez)))
                    for i in idxs:
                        if i is not None and 0 <= i < len(self.spine_radius):
                            self.spine_radius[i] = 1.0
                    try:
                        self._spine_clear_attr_layer_group(context.object, 'BH_ATTR_Inflate')
                    except Exception:
                        pass
                    self._spine_apply(context)
                    context.area.tag_redraw()
                    self.report({'INFO'}, "Shrink/Inflate removed")
                    return {'RUNNING_MODAL'}
                # Ctrl+A = Shrink/Inflate
                self._attr_mode = None
                self._attr_start_values = None
                self._attr_mode = 'RADIUS'
                self._attr_start_mouse = event.mouse_region_x
                idxs = self.selected_point_indices() or (
                    {self.active_handle} if self.active_handle is not None else set(range(len(self.bez)))
                )
                if not hasattr(self, 'spine_radius') or len(self.spine_radius) != len(self.bez):
                    self.spine_radius = [1.0] * len(self.bez)
                self._attr_start_values = {i: self.spine_radius[i] for i in idxs if i is not None}
                self._spine_push_undo(context)
                self.report({'INFO'}, "Shrink/Inflate: move mouse  |  LMB confirm  |  RMB cancel")
                return {'RUNNING_MODAL'}

            if event.type == 'T' and event.value == 'PRESS' and event.alt and not event.ctrl:
                self._attr_mode = None
                self._attr_start_values = None
                self._spine_push_undo(context)
                if not hasattr(self, 'spine_tilt') or len(self.spine_tilt) != len(self.bez):
                    self.spine_tilt = [0.0] * len(self.bez)
                idxs = self.selected_point_indices() or set(range(len(self.bez)))
                for i in idxs:
                    if i is not None and 0 <= i < len(self.spine_tilt):
                        self.spine_tilt[i] = 0.0
                try:
                    self._spine_clear_attr_layer_group(context.object, 'BH_ATTR_Tilt')
                except Exception:
                    pass
                self._spine_apply(context)
                context.area.tag_redraw()
                self.report({'INFO'}, "Tilt removed")
                return {'RUNNING_MODAL'}

            # G grab / R rotate / S scale selected controllers (multi-chain)
            if event.type in {'G', 'R', 'S'} and event.value == 'PRESS' and not event.ctrl and not event.alt:
                self.selected = self._spine_norm_selected()
                keys = [k for k in self.selected if (k[2] if len(k) == 3 else k[1]) == 'co']
                if not keys and self.active_handle is not None:
                    ac = int(getattr(self, 'active_chain', 0) or 0)
                    keys = [(ac, self.active_handle, 'co')]
                if not keys:
                    return {'RUNNING_MODAL'}
                self._spine_push_undo(context)
                if event.type == 'G':
                    self._xform_mode = None
                    part = getattr(self, 'active_bez_part', 'co')
                    hi = self.active_handle if self.active_handle is not None else (keys[0][1] if len(keys[0]) == 3 else keys[0][0])
                    self.start_drag(context, event, hi, part)
                elif event.type == 'R':
                    self._spine_start_xform(context, event, 'ROTATE', keys)
                else:
                    self._spine_start_xform(context, event, 'SCALE', keys)
                return {'RUNNING_MODAL'}

            if event.type == 'RIGHTMOUSE' and event.value == 'PRESS' and (
                self.dragging or getattr(self, '_xform_mode', None)
            ):
                self._spine_undo(context)
                self.dragging = False
                self._xform_mode = None
                self._pending_click_drag = False
                self.constraint_axis = None
                self._xform_start = None
                self._xform_keys = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # V handle type (reuse vertex menu)
            if event.type == 'V' and event.value == 'PRESS' and not event.ctrl and not event.alt:
                def draw_handle_popup(menu, _context):
                    layout = menu.layout
                    for mid, name, _desc in (
                        ('AUTO', 'Automatic', ''),
                        ('ALIGNED', 'Aligned', ''),
                        ('FREE', 'Free', ''),
                    ):
                        op = layout.operator("mesh.vdh_handle_type", text=name)
                        op.mode = mid
                context.window_manager.popup_menu(draw_handle_popup, title="Handle Type")
                return {'RUNNING_MODAL'}

            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    def modal(self, context, event):
        # Left Edit Mode (e.g. Tab → Object) → auto-confirm like pressing Enter
        if context.mode != 'EDIT_MESH':
            try:
                self.finish(context, cancel=False)
            except Exception:
                try:
                    self.finish(context, cancel=True)
                except Exception:
                    pass
            return {'FINISHED'}

        obj, bm = self.get_obj_bm(context)
        if obj is None:
            self.finish(context, cancel=True)
            return {'CANCELLED'}
        # Mesh-selection locking is only needed by Spine modes.
        # In Vertex mode the original selection is already fixed at invoke time;
        # scanning every vertex/edge/face on every mouse event is extremely costly
        # on dense meshes and was the main source of proportional-drag lag.

        # Spine modes have their own modal path
        if getattr(self, 'tool_mode', 'VERTEX') in ('SPINE_PLACE', 'SPINE_DEFORM'):
            try:
                return self._modal_spine(context, event)
            except Exception as e:
                # Keep tool alive — only report the error
                try:
                    self.report({'ERROR'}, f"Spine error: {e}")
                except Exception:
                    pass
                try:
                    context.area.tag_redraw()
                except Exception:
                    pass
                return {'RUNNING_MODAL'}

        # Vertex mode: do not scan the whole BMesh on every modal event.
        # Mesh selection is locked by the tool and mesh-edit selection operations
        # are blocked while the modal operator is active.

        # UI unlocked like Spine: snap, proportional, headers, N-panel...
        if self._spine_event_in_ui(context, event):
            return {'PASS_THROUGH'}
        region = context.region
        if region is not None and region.type in {
            'HEADER', 'TOOL_HEADER', 'TOOLS', 'UI', 'HUD', 'NAVIGATION_BAR',
        }:
            return {'PASS_THROUGH'}

        # Track Ctrl for temporary snap (Blender-style)
        self._ctrl_snap = bool(event.ctrl)

        # Detect prop toggled from UI (header) not only via O key
        prop_now = bool(context.tool_settings.use_proportional_edit)
        prop_was = getattr(self, '_prop_was_on', None)
        if prop_was is not None and prop_was != prop_now:
            self._sync_prop_toggle(context, was_on=prop_was)
        self._prop_was_on = prop_now

        # Vertex Mode: the same Alt+X mouse-direction handle alignment gesture
        # as Spine Mode. This branch is Vertex-only; Spine Mode is untouched.
        if (event.type == 'X' and event.value == 'PRESS' and event.alt
                and not event.ctrl and not event.shift):
            self._vertex_axis_align_begin(context, event)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if getattr(self, '_vertex_axis_align', False):
            if event.type == 'MOUSEMOVE':
                if self._vertex_axis_align_finish(context, event):
                    return {'RUNNING_MODAL'}
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type in {'LEFT_ALT', 'RIGHT_ALT'} and event.value == 'RELEASE':
                self._vertex_axis_align = False
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type != 'MOUSEMOVE':
                return {'RUNNING_MODAL'}

        # Scroll handling:
        # - Shift + Scroll: pull selected verts toward blue curve (align)
        # - While dragging + proportional ON: change prop size
        # - Otherwise: let Blender zoom (PASS_THROUGH)
        wheel_up = event.type in {'WHEELUPMOUSE', 'WHEELINMOUSE', 'PAGE_UP'}
        wheel_down = event.type in {'WHEELDOWNMOUSE', 'WHEELOUTMOUSE', 'PAGE_DOWN'}
        if (wheel_up or wheel_down) and event.value == 'PRESS':
            if event.shift and event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE'}:
                # Shift+Ctrl+Wheel: align selected verts toward the blue curve.
                # Keep one proportional neighborhood cache across repeated wheel ticks.
                self._align_prop_session = True
                amount = 0.15 if wheel_up else -0.15
                self.align_selection_to_curve(context, amount)
                self._vdh_refresh_edit_normals(context)
                return {'RUNNING_MODAL'}
            if getattr(self, '_align_prop_session', False):
                self._align_prop_session = False
                self._align_prop_cache = None
                if getattr(self, '_align_prop_kdtree_dirty', False):
                    self._rebuild_all_kdtree()
                    self._align_prop_kdtree_dirty = False
            if (
                getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX'
                and self.dragging
                and context.tool_settings.use_proportional_edit
            ):
                factor = (1.0 / 1.1) if wheel_up else 1.1
                ts = context.tool_settings
                size = max(1e-4, float(ts.proportional_size or self.prop_size or 1.0) * factor)
                ts.proportional_size = size
                self.prop_size = size
                self.update_drag(context, event)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            # Not dragging (or prop off) -> normal zoom
            return {'PASS_THROUGH'}

        # Confirm / cancel
        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            self.finish(context, cancel=False)
            return {'FINISHED'}
        if event.type == 'ESC' and event.value == 'PRESS':
            self.finish(context, cancel=True)
            return {'CANCELLED'}

        # Ctrl+Z / Ctrl+Shift+Z -> our handle undo (keeps mesh + handles together)
        if event.type == 'Z' and event.value == 'PRESS' and event.ctrl:
            if event.shift:
                self.do_redo(context)
            else:
                self.do_undo(context)
            return {'RUNNING_MODAL'}

        # Axis lock while dragging (G then X/Y/Z, or Shift+X/Y/Z for plane)
        if self.dragging and event.value == 'PRESS' and event.type in {'X', 'Y', 'Z'}:
            if event.shift:
                # Shift+X -> YZ plane, Shift+Y -> XZ, Shift+Z -> XY
                plane = {'X': 'YZ', 'Y': 'XZ', 'Z': 'XY'}[event.type]
                if getattr(self, "constraint_axis", None) == plane:
                    self.constraint_axis = None
                else:
                    self.constraint_axis = plane
            else:
                if getattr(self, "constraint_axis", None) == event.type:
                    self.constraint_axis = None
                else:
                    self.constraint_axis = event.type
            self.update_drag(context, event)
            return {'RUNNING_MODAL'}

        # V cycles handle type (like Curve) — must not reach mesh rip/V-menu
        if event.type == 'V' and event.value == 'PRESS':
            self.cycle_handle_mode(context)
            return {'RUNNING_MODAL'}

        # Vertex Mirror is fully synchronized with Blender's mesh mirror flags; no custom M toggle.

        # O toggles proportional; Shift+O cycles falloff type (like Blender)
        if event.type == 'O' and event.value == 'PRESS' and not event.ctrl and not event.alt:
            ts = context.tool_settings
            if event.shift:
                # Cycle falloff: Smooth -> Sphere -> Root -> Inverse Square -> Sharp -> Linear -> Constant
                order = (
                    'SMOOTH', 'SPHERE', 'ROOT', 'INVERSE_SQUARE',
                    'SHARP', 'LINEAR', 'CONSTANT',
                )
                cur = getattr(ts, 'proportional_edit_falloff', 'SMOOTH')
                try:
                    i = order.index(cur)
                except ValueError:
                    i = 0
                nxt = order[(i + 1) % len(order)]
                ts.proportional_edit_falloff = nxt
                self.prop_falloff = nxt
                self.report({'INFO'}, f"Proportional Falloff: {nxt.replace('_', ' ').title()}")
            else:
                was_on = ts.use_proportional_edit
                ts.use_proportional_edit = not ts.use_proportional_edit
                self._sync_prop_toggle(context, was_on=was_on)
                self._prop_was_on = ts.use_proportional_edit
                state = "ON" if ts.use_proportional_edit else "OFF"
                self.report({'INFO'}, f"Proportional Editing: {state}")
            if ts.proportional_size <= 0:
                ts.proportional_size = 1.0
            self.prop_size = ts.proportional_size
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # S / R / L : Space / Relax / Straighten selection (while tool active)
        if event.type == 'S' and event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            self.space_selection(context)
            return {'RUNNING_MODAL'}
        if event.type == 'R' and event.value == 'PRESS' and not event.ctrl and not event.alt:
            if event.shift:
                # Small popup to pick smooth type, then runs relax
                def draw_smooth_popup(menu, _ctx):
                    layout = menu.layout
                    layout.label(text="Smooth Type")
                    for mid, name, tip in VDH_SMOOTH_ITEMS:
                        op = layout.operator("mesh.vdh_smooth_choice", text=name)
                        op.mode = mid
                context.window_manager.popup_menu(draw_smooth_popup, title="Smooth Type")
            else:
                self.relax_selection(context)
            # R only changes handle type: AUTO -> ALIGNED. Keep handle positions and sizes untouched.
            if hasattr(self, 'point_modes'):
                self.point_modes = ['ALIGNED' if m == 'AUTO' else m for m in self.point_modes]
                try:
                    if hasattr(self, 'chain_index') and self.chain_index is not None:
                        self.chains[self.chain_index]['modes'] = list(self.point_modes)
                except Exception:
                    pass
            self._vdh_refresh_edit_normals(context)
            return {'RUNNING_MODAL'}
        if event.type == 'L' and event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            self.straighten_selection(context)
            # L only changes handle type: AUTO -> ALIGNED. Keep handle positions and sizes untouched.
            if hasattr(self, 'point_modes'):
                self.point_modes = ['ALIGNED' if m == 'AUTO' else m for m in self.point_modes]
                try:
                    if hasattr(self, 'chain_index') and self.chain_index is not None:
                        self.chains[self.chain_index]['modes'] = list(self.point_modes)
                except Exception:
                    pass
            self._vdh_refresh_edit_normals(context)
            return {'RUNNING_MODAL'}
        if event.type == 'F' and event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            self.set_flow_selection(context)

            # Set Flow is a modeling operation: freeze current automatic handle
            # solutions so the first controller move does not rebuild AUTO
            # handles from the previous state and cause a jump.
            if hasattr(self, "point_modes"):
                self.point_modes = ["ALIGNED" if m == "AUTO" else m for m in self.point_modes]
            try:
                if hasattr(self, "chain_index") and self.chain_index is not None:
                    ch = self.chains[self.chain_index]
                    ch["modes"] = list(self.point_modes)
            except Exception:
                pass

            # Set Flow changes the actual mesh positions. Re-bake the current
            # vertex rest state so the first deformation starts from the new
            # flow result instead of jumping back to the pre-flow rest pose.
            try:
                obj, bm = self.get_obj_bm(context)
                if obj is not None and bm is not None:
                    self.rest_local = [bm.verts[i].co.copy() for i in self.vert_indices if i < len(bm.verts)]
                    self.all_rest = {v.index: v.co.copy() for v in bm.verts}
                    self.initial_rest_local = [p.copy() for p in self.rest_local]
                    self.initial_all_rest = {k: v.copy() for k, v in self.all_rest.items()}
                    self._all_kdtree_dirty = True
                    self._prop_nearest = None
                    self._prop_nearest_key = None
            except Exception:
                pass
            self._vdh_refresh_edit_normals(context)
            return {'RUNNING_MODAL'}

        # [ ] resize on-screen controllers + handles (both modes use display_scale)
        if event.type in {'LEFT_BRACKET', 'RIGHT_BRACKET'} and event.value == 'PRESS':
            ds = float(getattr(self, 'display_scale', 1.0) or 1.0)
            if event.type == 'LEFT_BRACKET':
                ds *= 0.8
            else:
                ds *= 1.25
            self.display_scale = max(0.05, min(5.0, ds))
            try:
                _vdh_set_vertex_display_scale(context.object, self.display_scale)
            except Exception:
                pass
            context.area.tag_redraw()
            self.report({'INFO'}, f"Handle size: {self.display_scale:.2f}x")
            return {'RUNNING_MODAL'}

        # Block mesh edit ops; selection stays locked (O is allowed above)
        # S, R, L, F handled above; V handled earlier
        blocked = {
            'B', 'C', 'W', 'U', 'I', 'J', 'K', 'P', 'N', 'M',
            'E', 'X', 'Y', 'Z', 'DEL', 'BACK_SPACE',
            'TAB', 'H',
            'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE',
            'D', 'T', 'Q',
        }
        if event.type in blocked and event.value == 'PRESS':
            return {'RUNNING_MODAL'}

        # A: select all controllers (co only, not handle tips)
        if event.type == 'A' and event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            self.selected = {(i, 'co') for i in range(len(self.bez))}
            if self.bez:
                self.active_handle = 0
                self.active_bez_part = 'co'
            context.area.tag_redraw()
            self.report({'INFO'}, f"Selected {len(self.selected)} controllers")
            return {'RUNNING_MODAL'}

        if event.type == 'G' and event.value == 'PRESS':
            # Prefer selected controllers; fallback to active
            idxs = self.selected_point_indices()
            if idxs:
                # Use active if in selection, else first selected
                if self.active_handle in idxs:
                    idx = self.active_handle
                    part = getattr(self, 'active_bez_part', 'co')
                    if not self._sel_has(idx, part):
                        part = 'co'
                else:
                    idx = next(iter(idxs))
                    part = 'co'
                self.push_undo()
                self.start_drag(context, event, idx, part)
            elif self.active_handle is not None:
                self.push_undo()
                part = getattr(self, 'active_bez_part', 'co')
                self.start_drag(context, event, self.active_handle, part)
            return {'RUNNING_MODAL'}

        # Header / tool header / sidebar / N-panel → allow UI (shading, snap, prop)
        # Keep mesh tools & gizmos locked by not passing empty viewport clicks to Blender
        region = context.region
        if region is not None and region.type in {
            'HEADER', 'TOOL_HEADER', 'TOOLS', 'UI', 'HUD', 'NAVIGATION_BAR',
        }:
            return {'PASS_THROUGH'}

        # Mouse: handle pick / box select controllers
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self.dragging:
                self.dragging = False
                return {'RUNNING_MODAL'}
            if region is None or region.type != 'WINDOW':
                return {'PASS_THROUGH'}
            hit = self.pick_handle(context, event, pixel_dist=28.0)
            if hit is not None:
                if len(hit) == 3:
                    _ci, idx, part = hit
                else:
                    idx, part = hit
                if event.alt and not event.shift:
                    if not hasattr(self, 'point_modes') or len(self.point_modes) != len(self.bez):
                        self.point_modes = ['AUTO'] * len(self.bez)
                    self.point_modes[idx] = 'FREE'
                    self.select_only(idx, part)
                    self.report({'INFO'}, "Handle type: Free")
                    context.area.tag_redraw()
                    self.push_undo()
                    self.start_drag(context, event, idx, part)
                    return {'RUNNING_MODAL'}
                if event.shift:
                    was_selected = self._sel_has(idx, part)
                    self.select_toggle(idx, part)
                    if was_selected:
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                else:
                    self.select_only(idx, part)
                self.push_undo()
                self.start_drag(context, event, idx, part)
                return {'RUNNING_MODAL'}
            # Empty viewport click → start box select
            self.box_selecting = True
            self.box_start = (event.mouse_region_x, event.mouse_region_y)
            self.box_end = self.box_start
            # Ctrl+Shift: box-select handle tips only; else controllers (co) only
            self.box_handles_only = bool(event.ctrl and event.shift)
            if not event.shift and not self.box_handles_only:
                self.selected = set()
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.dragging:
                self.dragging = False
                self._bake_prop_after_drag(context)
                self._vertex_mirror_drag_source_base = {}
                self._vertex_mirror_drag_target_base = {}
                self._vertex_mirror_drag_axis = 'OFF'
                self._vertex_mirror_drag_axis_items = ()
                self._vertex_mirror_drag_source_sides = {}
                self._vertex_mirror_drag_pairs = None
                return {'RUNNING_MODAL'}
            if self.box_selecting:
                self.box_end = (event.mouse_region_x, event.mouse_region_y)
                self._finish_box_select(context, event)
                self.box_selecting = False
                self.box_start = self.box_end = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            if self.dragging:
                self.update_drag(context, event)
                return {'RUNNING_MODAL'}
            if self.box_selecting:
                self.box_end = (event.mouse_region_x, event.mouse_region_y)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        # RMB cancels current drag (restore last undo snap)
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS' and self.dragging:
            if self.undo_stack:
                self.restore_handles(self.undo_stack.pop())
                self.apply_deform(context)
            self.dragging = False
            self._vertex_mirror_drag_source_base = {}
            self._vertex_mirror_drag_target_base = {}
            self._vertex_mirror_drag_axis = 'OFF'
            self._vertex_mirror_drag_axis_items = ()
            self._vertex_mirror_drag_source_sides = {}
            self._vertex_mirror_drag_pairs = None
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Shift + Middle: add handle on curve only
        if event.type == 'MIDDLEMOUSE' and event.value == 'PRESS' and event.shift and not event.alt:
            picked = self.pick_curve_local(context, event)
            if picked is not None:
                t, local_p = picked
                self.push_undo()
                insert_at = len(self.bez)
                for i in range(len(self.bez) - 1):
                    t0 = i / (len(self.bez) - 1)
                    t1 = (i + 1) / (len(self.bez) - 1)
                    if t0 <= t <= t1:
                        insert_at = i + 1
                        break
                # Exact position + tangent of blue curve at t (before insert)
                cur_p = eval_bezier_points(self.bez, t)
                rest_p = eval_bezier_points(self.rest_bez, t)
                tan = bezier_chain_tangent(self.bez, t)
                tan_rest = bezier_chain_tangent(self.rest_bez, t)

                cos_old = [p['co'].copy() for p in self.bez]
                if insert_at <= 0:
                    ndist = (cos_old[0] - cur_p).length if cos_old else 1.0
                elif insert_at >= len(cos_old):
                    ndist = (cos_old[-1] - cur_p).length if cos_old else 1.0
                else:
                    d0 = (cur_p - cos_old[insert_at - 1]).length
                    d1 = (cos_old[insert_at] - cur_p).length
                    ndist = max(0.5 * (d0 + d1), 1e-4)

                # Short handle length so new controller doesn't overlap neighbors
                # ~1/4 of distance to nearer neighbor
                nearer = ndist
                if insert_at > 0 and insert_at < len(cos_old):
                    nearer = min(
                        (cur_p - cos_old[insert_at - 1]).length,
                        (cos_old[insert_at] - cur_p).length,
                    )
                L = max(nearer * 0.2, min(nearer * 0.28, ndist * 0.3))
                L_rest = L

                new_bp = {
                    'co': cur_p.copy(),
                    'hl': cur_p - tan * L,
                    'hr': cur_p + tan * L,
                }
                new_rbp = {
                    'co': rest_p.copy(),
                    'hl': rest_p - tan_rest * L_rest,
                    'hr': rest_p + tan_rest * L_rest,
                }

                self.bez.insert(insert_at, new_bp)
                self.rest_bez.insert(insert_at, new_rbp)

                self.handle_params.insert(insert_at, float(t))
                self.handle_params[0] = 0.0
                self.handle_params[-1] = 1.0
                if not hasattr(self, 'point_modes'):
                    self.point_modes = ['AUTO'] * len(self.bez)
                else:
                    self.point_modes.insert(insert_at, 'AUTO')

                # A newly inserted controller and its immediate neighbors are FREE.
                # Do not rebuild their handles: preserving the current handle tips
                # prevents the insertion itself from changing the curve/deformation.
                for ni in (insert_at - 1, insert_at, insert_at + 1):
                    if 0 <= ni < len(self.point_modes):
                        self.point_modes[ni] = 'FREE'

                try:
                    # Only rebuild AUTO points so FREE/ALIGNED stay
                    self.rebuild_auto_handles(interior=True)
                except Exception:
                    pass
                if hasattr(self, 'chains') and getattr(self, 'chain_index', None) is not None:
                    try:
                        self.chains[self.chain_index]['modes'] = list(self.point_modes)
                    except Exception:
                        pass

                # Re-parameterize verts onto the REST curve (preserves existing deformation)
                self._reparam_verts_on_curve()
                # Apply so mesh follows the (still deformed) bez with updated params
                self.apply_deform(context)

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            # not on curve -> let Blender use Shift+MMB if mapped; usually not pan
            return {'PASS_THROUGH'}

        # Alt + Shift + Middle: remove handle ONLY if cursor is on a handle;
        # otherwise PASS_THROUGH so Alt+MMB remains available for navigation
        if event.type == 'MIDDLEMOUSE' and event.value == 'PRESS' and event.alt and event.shift:
            hit = self.pick_handle(context, event)
            if hit is not None and len(self.bez) > 2:
                self.push_undo()
                hit = hit[0]  # index only
                self.bez.pop(hit)
                self.rest_bez.pop(hit)
                if hit < len(self.handle_params):
                    self.handle_params.pop(hit)
                if self.handle_params:
                    self.handle_params[0] = 0.0
                    self.handle_params[-1] = 1.0
                if hasattr(self, 'point_modes') and hit < len(self.point_modes):
                    self.point_modes.pop(hit)
                # Rebuild handles independently so existing deformation is preserved
                cos = [p['co'].copy() for p in self.bez]
                rcos = [p['co'].copy() for p in self.rest_bez]
                self.bez = make_bezier_points(cos, poly=self.rest_local, center=self._arc_center, normal=self._arc_normal)
                self.rest_bez = make_bezier_points(rcos, poly=self.rest_local, center=self._arc_center, normal=self._arc_normal)
                # keep point_modes length in sync
                while len(self.point_modes) > len(self.bez):
                    self.point_modes.pop()
                while len(self.point_modes) < len(self.bez):
                    self.point_modes.append('AUTO')
                self.rebuild_auto_handles()
                # re-apply end look-at to rest_bez as well
                n = len(self.rest_bez)
                if n >= 2 and self.point_mode(0) == 'AUTO':
                    # simple: copy end handles direction from bez scaled to rest lengths
                    pass  # rest ends already set by make_bezier_points
                self._reparam_verts_on_curve()
                if self.active_handle is not None:
                    if self.active_handle == hit:
                        self.active_handle = None
                    elif self.active_handle > hit:
                        self.active_handle -= 1
                self.apply_deform(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        # Allow navigation (orbit / pan / zoom)
        return {'PASS_THROUGH'}


    def _draw_spine_text(self, context):
        mode = getattr(self, 'tool_mode', 'SPINE_PLACE')
        n = len(getattr(self, 'spine_points', []))
        vol = "On" if getattr(self, '_place_in_volume', False) else "Off"
        if mode == 'SPINE_PLACE':
            edit = bool(getattr(self, '_spine_edit_place', False))
            placing_new = bool(getattr(self, '_spine_placing_new_chain', False))
            n_chains = len(getattr(self, 'spine_chains', None) or [])
            if edit and not placing_new:
                lines = [
                    "Edit Place",
                    "Shift+MMB: Insert controller  |  Shift+Enter: New chain  |  Enter: Rebind all",
                    f"M: Set Initial  |  Shift+D: Dup  |  Ctrl+M: Mirror  |  Ctrl+Shift+M: Dup + Mirror  |  P: Volume ({vol})",
                    "Ctrl+Alt+C: Clear Cache",
                    "N: In Front  |  Shift+N: All In Front",
                ]
            elif edit and placing_new:
                lines = [
                    "Edit Place — New chain",
                    "Click: Add  |  Shift+Enter: Close & another  |  Enter: Rebind all",
                    f"P: Place in Volume ({vol})  |  Esc: Cancel",
                ]
            else:
                lines = [
                    "Spine Place  |  [ / ]: Resize handles",
                    f"Enter: Bind/Rebind  |  Shift+Enter: New chain  |  N: In Front  |  Shift+N: All In Front",
                    f"Ctrl+M: Mirror  |  P: Place in Volume ({vol})",
                    "Shift+[ ] / Shift+Wheel: Influence  |  Alt+[ ]: Overlay Size  |  Shift+F: Falloff  |  I: Overlay  |  Esc: Cancel",
                ]
        else:
            # Controller count from active chain bez
            bez = getattr(self, 'bez', None) or []
            n_ctrl = len(bez) if bez else n
            # Falloff label only when exactly one controller (co) is selected
            fo_label = None
            fo_key = None
            sel_cos = []
            ac = int(getattr(self, 'active_chain', 0) or 0)
            for item in (getattr(self, 'selected', None) or set()):
                if len(item) == 3:
                    ci, i, p = item
                    if p == 'co':
                        sel_cos.append((ci, i))
                elif len(item) == 2:
                    i, p = item
                    if p == 'co':
                        sel_cos.append((ac, i))
            if len(sel_cos) == 1:
                ci, i = sel_cos[0]
                chains = getattr(self, 'spine_chains', None) or []
                pfo = None
                if chains and 0 <= ci < len(chains):
                    ch = chains[ci]
                    pfo = ch.get('point_inf_falloff')
                    nb = len(ch.get('bez') or [])
                else:
                    nb = n_ctrl
                    pfo = getattr(self, 'point_inf_falloff', None)
                if pfo is None:
                    pfo = []
                if 0 <= i < max(nb, len(pfo)):
                    fo = 'CONSTANT'
                    if i < len(pfo) and pfo[i]:
                        fo = str(pfo[i])
                    elif getattr(self, 'point_inf_falloff', None) and i < len(self.point_inf_falloff):
                        fo = str(self.point_inf_falloff[i])
                    fo_label = fo.replace('_', ' ').title()
                    fo_key = str(fo).upper().replace(' ', '_')
            attr = str(getattr(self, '_attr_mode', '') or '').upper()
            if attr in ('TILT', 'RADIUS'):
                kind = "Tilt" if attr == 'TILT' else "Shrink/Inflate"
                interp = str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH')
                interp_label = interp.replace('_', ' ').title()
                lines = [
                    f"{kind}  |  Falloff: {interp_label}",
                    "[ / ]: Cycle falloff (Smooth / Sphere / Linear / Sharp / Constant)",
                    "Move mouse to adjust  |  LMB: Confirm  |  RMB / Esc: Cancel",
                ]
            else:
                head = "Spine Deform  |  [ / ]: Resize handles  |  Alt+X: Align handle to axis by mouse direction"
                if fo_label:
                    head += f"  |  Falloff: {fo_label}"
                lines = [
                    head,
                    f"Ctrl+T: Tilt  |  Alt+T: Remove Tilt  |  Ctrl+A: Shrink/Inflate  |  Ctrl+Alt+A: Remove Shrink/Inflate",
                    "G/R/S: Transform (Blender pivot)  |  Ctrl+R: Reset  |  Ctrl+Alt+R: Reset All  |  Alt+R: Reset Radius  |  Shift+L: Straighten",
                    "W: Align Longitudinal Loops  |  Q: Circularize Rings  |  E: Uniform Thickness",
                    "Shift+[ ] / Shift+Wheel: Influence  |  Alt+[ ]: Overlay Size  |  Shift+F: Falloff  |  I: Overlay",
                    "N: In Front  |  Shift+N: All In Front  |  Ctrl+M: Mirror",
                    "Alt+Enter: Edit place  |  Enter: Confirm  |  Esc: Cancel",
                ]
        # Deform HUD uses blue; Edit Place remains green.
        hud_col = (0.35, 0.75, 1.0, 1.0) if mode != 'SPINE_PLACE' else (0.45, 1.0, 0.55, 1.0)
        font_id = 0
        try:
            blf.size(font_id, 14)
        except TypeError:
            blf.size(font_id, 14, 72)
        region = context.region
        line_h = 18
        base_y = 28
        vol_on = bool(getattr(self, '_place_in_volume', False))
        for i, line in enumerate(reversed(lines)):
            y = base_y + i * line_h
            # Split line so On/Off can be colored differently
            marker = "Volume ("
            if marker in line:
                pre, rest = line.split(marker, 1)
                if rest.startswith("On"):
                    state, post = "On", rest[2:]
                elif rest.startswith("Off"):
                    state, post = "Off", rest[3:]
                else:
                    state, post = "", rest

                # Edit Place stays green; keys and descriptions use different
                # green shades.
                key_col = (0.45, 1.0, 0.55, 1.0)
                fn_col = (0.60, 0.88, 0.68, 1.0)

                def _parts(text_part):
                    parts = []
                    for seg_i, seg in enumerate(text_part.split(' | ')):
                        if seg_i:
                            parts.append(('  |  ', hud_col))
                        if ': ' in seg:
                            key_part, fn_part = seg.split(': ', 1)
                            parts.append((key_part + ':', key_col))
                            parts.append((' ' + fn_part, fn_col))
                        else:
                            parts.append((seg, key_col))
                    return parts

                spans = _parts(pre)
                spans.append((marker, fn_col))
                if state:
                    spans.append((state, (0.15, 1.0, 0.25, 1.0) if vol_on
                                  else (1.0, 0.15, 0.12, 1.0)))
                if post.startswith(")"):
                    spans.append((")", fn_col))
                    remainder = post[1:]
                else:
                    remainder = post
                if remainder:
                    spans.extend(_parts(remainder))

                try:
                    tw = sum(blf.dimensions(font_id, txt)[0] for txt, _col in spans)
                except Exception:
                    tw = sum(len(txt) * 7 for txt, _col in spans)
                x = (region.width - tw) * 0.5

                sx = x
                for txt, col in spans:
                    blf.position(font_id, sx, y, 0)
                    blf.color(font_id, *col)
                    blf.draw(font_id, txt)
                    try:
                        sx += blf.dimensions(font_id, txt)[0]
                    except Exception:
                        sx += len(txt) * 7
            elif "Falloff: " in line:
                # Color falloff type per kind
                pre, fo_part = line.split("Falloff: ", 1)
                pre = pre + "Falloff: "
                # fo_part is only the type name (end of line)
                fo_name = fo_part.strip()
                fo_colors = {
                    'CONSTANT': (1.0, 0.25, 0.22, 1.0),        # red
                    'SMOOTH': (0.2, 1.0, 0.45, 1.0),          # green
                    'SPHERE': (0.80, 0.35, 1.0, 1.0),            # violet
                    'ROOT': (0.70, 0.55, 1.0, 1.0),            # violet
                    'INVERSE_SQUARE': (1.0, 0.85, 0.25, 1.0),  # yellow
                    'SHARP': (1.0, 0.55, 0.20, 1.0),           # orange
                    'LINEAR': (1.0, 0.40, 0.75, 1.0),          # magenta
                }
                key = fo_name.upper().replace(' ', '_')
                fo_col = fo_colors.get(key, (1.0, 1.0, 0.4, 1.0))
                try:
                    w_pre = blf.dimensions(font_id, pre)[0]
                    w_fo = blf.dimensions(font_id, fo_name)[0]
                    tw = w_pre + w_fo
                except Exception:
                    w_pre = len(pre) * 7
                    w_fo = len(fo_name) * 7
                    tw = w_pre + w_fo
                x = (region.width - tw) * 0.5
                # shadow
                blf.position(font_id, x + 1, y - 1, 0)
                blf.color(font_id, 0.0, 0.0, 0.0, 0.75)
                blf.draw(font_id, pre)
                blf.position(font_id, x + 1 + w_pre, y - 1, 0)
                blf.draw(font_id, fo_name)
                # main
                blf.position(font_id, x, y, 0)
                blf.color(font_id, *hud_col)
                blf.draw(font_id, pre)
                blf.position(font_id, x + w_pre, y, 0)
                blf.color(font_id, *fo_col)
                blf.draw(font_id, fo_name)
            else:
                # Shortcut/function separation: keep shortcut labels in the existing
                # HUD color, while rendering the actual function/description brighter
                # for faster scanning.  Protected Falloff and Volume On/Off branches
                # above are intentionally left untouched.
                # Edit Place: make the shortcut keys clearly brighter/different
                # from their function descriptions.
                fn_col = (0.52, 0.76, 0.90, 1.0) if mode != 'SPINE_PLACE' else (0.78, 1.0, 0.82, 1.0)

                # Build colored spans.  A segment like ``G/R/S: Transform`` becomes
                # shortcut=HUD color and function=brighter color.  Segments without a
                # colon (titles such as ``Edit Place``) remain in the original color.
                spans = []
                for seg_i, seg in enumerate(line.split(' | ')):
                    if seg_i:
                        spans.append(('  |  ', hud_col))
                    if ': ' in seg:
                        key_part, fn_part = seg.split(': ', 1)
                        spans.append((key_part + ':', hud_col))
                        spans.append((' ' + fn_part, fn_col))
                    else:
                        spans.append((seg, hud_col))

                try:
                    tw = sum(blf.dimensions(font_id, txt)[0] for txt, _col in spans)
                except Exception:
                    tw = sum(len(txt) * 7 for txt, _col in spans)
                x = (region.width - tw) * 0.5

                # Main colored spans only — no black duplicate/shadow.
                sx = x
                for txt, col in spans:
                    blf.position(font_id, sx, y, 0)
                    blf.color(font_id, *col)
                    blf.draw(font_id, txt)
                    try:
                        sx += blf.dimensions(font_id, txt)[0]
                    except Exception:
                        sx += len(txt) * 7

    def draw_text_callback(self, context):
        """Brief help at bottom-center of the viewport + box select rect."""
        if context.area is None or context.region is None:
            return
        if context.area.type != 'VIEW_3D':
            return

        # Spine mode help + box select rect
        if getattr(self, 'tool_mode', 'VERTEX') in ('SPINE_PLACE', 'SPINE_DEFORM'):
            if getattr(self, 'box_selecting', False) and self.box_start and self.box_end:
                x0, y0 = self.box_start
                x1, y1 = self.box_end
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                gpu.state.blend_set('ALPHA')
                coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": coords})
                shader.bind()
                shader.uniform_float("color", (0.3, 0.9, 0.45, 0.95))
                gpu.state.line_width_set(1.5)
                batch.draw(shader)
                fill = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                batch_f = batch_for_shader(shader, 'TRI_FAN', {"pos": fill})
                shader.uniform_float("color", (0.2, 0.8, 0.35, 0.12))
                batch_f.draw(shader)
                gpu.state.blend_set('NONE')
            self._draw_spine_text(context)
            return

        # Draw box-select rectangle
        if getattr(self, 'box_selecting', False) and self.box_start and self.box_end:
            x0, y0 = self.box_start
            x1, y1 = self.box_end
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.blend_set('ALPHA')
            coords = [
                (x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0),
            ]
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": coords})
            shader.bind()
            if self.box_handles_only:
                shader.uniform_float("color", (1.0, 0.7, 0.2, 0.95))
            else:
                shader.uniform_float("color", (0.3, 0.75, 1.0, 0.95))
            gpu.state.line_width_set(1.5)
            batch.draw(shader)
            # fill
            fill = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            batch_f = batch_for_shader(shader, 'TRI_FAN', {"pos": fill})
            if self.box_handles_only:
                shader.uniform_float("color", (1.0, 0.6, 0.1, 0.12))
            else:
                shader.uniform_float("color", (0.2, 0.6, 1.0, 0.12))
            batch_f.draw(shader)
            gpu.state.blend_set('NONE')

        prop_on = context.tool_settings.use_proportional_edit
        prop_state = "On" if prop_on else "Off"
        prop_suffix = "  (O: Toggle  |  Shift+O: Falloff type)"

        # Vertex Mode HUD — keep every original help line visible.
        lines = [
            "Shift+MMB: Add controller  |  Alt+Shift+MMB: Remove controller  |  V: Handle type",
            "Alt+X+Mouse Direction: Align handle to mouse direction  |  R: Relax  |  L: Line  |  S: Space  |  F: Set Flow",
            "Shift+R: Smooth type  |  Shift+Scroll: Align to curve  |  Proportional Editing: ",
        ]

        font_id = 0
        try:
            blf.size(font_id, 14)
        except TypeError:
            blf.size(font_id, 14, 72)

        region = context.region
        line_h = 18
        base_y = 28

        # Vertex Mode: blue shortcut labels, softer blue descriptions.
        key_col = (0.35, 0.75, 1.0, 1.0)
        fn_col = (0.52, 0.76, 0.90, 1.0)

        def colored_parts(text_part):
            parts = []
            for seg_i, seg in enumerate(text_part.split(' | ')):
                if seg_i:
                    parts.append(('  |  ', key_col))
                if ': ' in seg:
                    key_part, fn_part = seg.split(': ', 1)
                    parts.append((key_part + ':', key_col))
                    parts.append((' ' + fn_part, fn_col))
                else:
                    parts.append((seg, key_col))
            return parts

        for i, line in enumerate(reversed(lines)):
            y = base_y + i * line_h
            spans = colored_parts(line)

            # Insert proportional-editing state and suffix after the label.
            if i == 0:
                spans.append((prop_state, (0.35, 1.0, 0.45, 1.0) if prop_on
                              else (1.0, 0.55, 0.4, 1.0)))
                spans.extend(colored_parts(prop_suffix))

            try:
                total_w = sum(blf.dimensions(font_id, txt)[0] for txt, _col in spans)
            except Exception:
                total_w = sum(len(txt) * 7 for txt, _col in spans)

            sx = (region.width - total_w) * 0.5

            # No black duplicate/shadow.
            for txt, col in spans:
                blf.position(font_id, sx, y, 0)
                blf.color(font_id, *col)
                blf.draw(font_id, txt)
                try:
                    sx += blf.dimensions(font_id, txt)[0]
                except Exception:
                    sx += len(txt) * 7



    def _draw_spine_deform_chains(self, context, obj):
        """Draw every bound chain (Bezier + controllers), highlight multi-select."""
        mw = obj.matrix_world
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
        rv3d = context.region_data
        segs = 18
        ds = self._effective_display_scale()
        ac = int(getattr(self, 'active_chain', 0) or 0)
        chains = getattr(self, 'spine_chains', None) or []

        def draw_dot(world, size, color):
            right = (rv3d.view_rotation @ Vector((1, 0, 0))).normalized() * size
            up = (rv3d.view_rotation @ Vector((0, 1, 0))).normalized() * size
            fan = [world]
            for s in range(segs + 1):
                a = (2.0 * math.pi * s) / segs
                fan.append(world + right * math.cos(a) + up * math.sin(a))
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": fan})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        for ci, ch in enumerate(chains):
            bez = ch.get('bez') or []
            if not bez:
                continue
            # Per-chain In Front (depth test)
            if bool(ch.get('in_front', True)):
                gpu.state.depth_test_set('NONE')
            else:
                gpu.state.depth_test_set('LESS_EQUAL')
            samples = 48
            pts = []
            for s in range(samples + 1):
                t = s / samples
                pts.append(mw @ eval_bezier_points(bez, t))
            is_active = (ci == ac)
            line_col = (0.3, 1.0, 0.55, 1.0) if is_active else (0.2, 0.65, 0.4, 0.75)
            if len(pts) >= 2:
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
                shader.bind()
                shader.uniform_float("color", (line_col[0], line_col[1], line_col[2], 0.3))
                gpu.state.line_width_set(9.0)
                batch.draw(shader)
                shader.uniform_float("color", line_col)
                gpu.state.line_width_set(2.8)
                batch.draw(shader)
            # controllers + handles
            n = len(bez)
            for i, bp in enumerate(bez):
                wco = mw @ bp['co']
                sel = self._sel_has(i, 'co', chain_idx=ci)
                ah = getattr(self, 'active_handle', None)
                # Active = last selected controller on active chain (any part → whole controller)
                act = is_active and ah is not None and int(i) == int(ah)
                if act:
                    draw_dot(wco, self._screen_constant_controller_radius(context, wco, 9.0 * ds), (0.15, 0.45, 1.0, 1.0))
                elif sel:
                    draw_dot(wco, self._screen_constant_controller_radius(context, wco, 7.0 * ds), (1.0, 1.0, 0.25, 1.0))
                elif is_active:
                    draw_dot(wco, self._screen_constant_controller_radius(context, wco, 7.0 * ds), (0.25, 0.95, 0.5, 0.95))
                else:
                    draw_dot(wco, self._screen_constant_controller_radius(context, wco, 7.0 * ds), (0.2, 0.7, 0.4, 0.75))
                # handle tips
                for part in ('hl', 'hr'):
                    if i == 0 and part == 'hl':
                        continue
                    if i == n - 1 and part == 'hr':
                        continue
                    if (bp[part] - bp['co']).length < 1e-8:
                        continue
                    wtip = mw @ bp[part]
                    tip_sel = self._sel_has(i, part, chain_idx=ci)
                    mode = modes[i] if i < len(modes) else 'AUTO'
                    if mode == 'FREE':
                        tip_col = (1.0, 0.4, 0.35, 0.85)
                        tip_sel_col = (1.0, 0.18, 0.12, 1.0)
                    elif mode == 'ALIGNED':
                        tip_col = (1.0, 0.75, 0.3, 0.85)
                        tip_sel_col = (1.0, 0.52, 0.08, 1.0)
                    else:
                        tip_col = (0.4, 0.9, 0.55, 0.75)
                        tip_sel_col = (0.12, 0.75, 0.35, 1.0)
                    draw_dot(wtip, self._screen_constant_controller_radius(context, wtip, 6.0 * ds), tip_sel_col if tip_sel else tip_col)
                    # arm
                    arm = [wco, wtip]
                    batch = batch_for_shader(shader, 'LINES', {"pos": arm})
                    shader.bind()
                    shader.uniform_float("color", (0.3, 0.6, 0.9, 0.6))
                    gpu.state.line_width_set(1.5)
                    batch.draw(shader)

        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)


    def _draw_spine_edit_place_bez(self, context, obj):
        """Edit Place: draw ALL deform chains with handles (green), active brighter."""
        mw = obj.matrix_world
        chains = getattr(self, 'spine_chains', None) or []
        ac = int(getattr(self, 'active_chain', 0) or 0)
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
        rv3d = context.region_data
        ds = self._effective_display_scale()
        segs = 18
        samples = 48

        def draw_dot(world, size, color):
            right = (rv3d.view_rotation @ Vector((1, 0, 0))).normalized() * size
            up = (rv3d.view_rotation @ Vector((0, 1, 0))).normalized() * size
            fan = [world]
            for s in range(segs + 1):
                a = (2.0 * math.pi * s) / segs
                fan.append(world + right * math.cos(a) + up * math.sin(a))
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": fan})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        def draw_one_bez(bez, modes, is_active, in_front=True):
            if not bez or len(bez) < 2:
                return
            if in_front:
                gpu.state.depth_test_set('NONE')
            else:
                gpu.state.depth_test_set('LESS_EQUAL')
            pts = [mw @ eval_bezier_points(bez, s / samples) for s in range(samples + 1)]
            line_col = (0.3, 1.0, 0.5, 1.0) if is_active else (0.2, 0.7, 0.4, 0.7)
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
            shader.bind()
            shader.uniform_float("color", (line_col[0], line_col[1], line_col[2], 0.3))
            gpu.state.line_width_set(9.0 * min(ds, 2.0))
            batch.draw(shader)
            shader.uniform_float("color", line_col)
            gpu.state.line_width_set(3.0 * min(ds, 2.0))
            batch.draw(shader)
            n = len(bez)
            modes = modes or ['AUTO'] * n
            for i, bp in enumerate(bez):
                wco = mw @ bp['co']
                mode = modes[i] if i < len(modes) else 'AUTO'
                tips = []
                if i == 0:
                    tips = [('hr', bp['hr'])]
                elif i == n - 1:
                    tips = [('hl', bp['hl'])]
                else:
                    tips = [('hl', bp['hl']), ('hr', bp['hr'])]
                for part, tip in tips:
                    if (tip - bp['co']).length < 1e-8:
                        continue
                    wtip = mw @ tip
                    batch = batch_for_shader(shader, 'LINES', {"pos": [wco, wtip]})
                    shader.bind()
                    if mode == 'FREE':
                        shader.uniform_float("color", (1.0, 0.4, 0.35, 0.85))
                    elif mode == 'ALIGNED':
                        shader.uniform_float("color", (1.0, 0.75, 0.3, 0.85))
                    else:
                        shader.uniform_float("color", (0.4, 0.9, 0.55, 0.75))
                    gpu.state.line_width_set(1.4)
                    batch.draw(shader)
                    tip_sel = is_active and self._sel_has(i, part)
                    tip_act = is_active and (i == getattr(self, 'active_handle', None) and getattr(self, 'active_bez_part', 'co') == part)
                    col = (1.0, 1.0, 0.3, 1.0) if (tip_sel or tip_act) else (0.45, 0.95, 0.55, 0.9)
                    draw_dot(wtip, self._screen_constant_controller_radius(context, wtip, 4.5 * ds), col)
                co_sel = is_active and self._sel_has(i, 'co')
                ah = getattr(self, 'active_handle', None)
                co_act = is_active and ah is not None and int(i) == int(ah)
                if co_act:
                    draw_dot(wco, self._screen_constant_controller_radius(context, wco, 9.0 * ds), (0.15, 0.45, 1.0, 1.0))
                elif co_sel:
                    draw_dot(wco, self._screen_constant_controller_radius(context, wco, 7.0 * ds), (1.0, 1.0, 0.3, 1.0))
                else:
                    col = (0.25, 0.95, 0.45, 0.95) if is_active else (0.2, 0.75, 0.4, 0.8)
                    draw_dot(wco, self._screen_constant_controller_radius(context, wco, 7.0 * ds), col)

        # Draw non-active first, active on top
        for ci, ch in enumerate(chains):
            if ci == ac:
                continue
            draw_one_bez(ch.get('bez'), ch.get('modes'), False, bool(ch.get('in_front', True)))
        placing_new = bool(getattr(self, '_spine_placing_new_chain', False))
        live = getattr(self, 'bez', None) or []
        live_ok = len(live) >= 2
        if 0 <= ac < len(chains):
            ch = chains[ac]
            in_front = bool(ch.get('in_front', True))
            # Prefer live bez only when it is a real curve (not cleared after Shift+Enter)
            if live_ok and not placing_new:
                bez = live
                modes = getattr(self, 'point_modes', None) or ch.get('modes')
                draw_one_bez(bez, modes, True, in_front)
            else:
                # Show stored chain (keeps previous chains visible while placing new)
                draw_one_bez(ch.get('bez'), ch.get('modes'), not placing_new, in_front)
        elif live_ok:
            draw_one_bez(
                live, getattr(self, 'point_modes', None), True,
                bool(getattr(self, 'spine_in_front', True)),
            )

        # New place chains (points only, dim)
        for ch_pts in (getattr(self, 'spine_chains_pts', None) or []):
            if not ch_pts or len(ch_pts) < 2:
                continue
            pts = [mw @ p for p in ch_pts]
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
            shader.bind()
            shader.uniform_float("color", (0.15, 0.55, 0.3, 0.5))
            gpu.state.line_width_set(4.0)
            batch.draw(shader)

        # If placing new chain, also show current spine_points
        if getattr(self, '_spine_placing_new_chain', False):
            pts = [mw @ p for p in (getattr(self, 'spine_points', None) or [])]
            if len(pts) >= 2:
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
                shader.bind()
                shader.uniform_float("color", (0.3, 1.0, 0.5, 1.0))
                gpu.state.line_width_set(3.5)
                batch.draw(shader)
            for i, wco in enumerate(pts):
                col = (1.0, 1.0, 0.3, 1.0) if self._sel_has(i, 'co') else (0.25, 0.95, 0.45, 0.95)
                draw_dot(wco, self._screen_constant_controller_radius(context, wco, 7.0 * ds), col)

        gpu.state.depth_mask_set(True)
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)

    def _draw_spine(self, context, obj):
        """Draw spine polyline + controller dots (including completed chains)."""
        mw = obj.matrix_world
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
        rv3d = context.region_data
        segs = 18
        ds = self._effective_display_scale()

        def draw_dot(world, size, color):
            right = (rv3d.view_rotation @ Vector((1, 0, 0))).normalized() * size
            up = (rv3d.view_rotation @ Vector((0, 1, 0))).normalized() * size
            fan = [world]
            for s in range(segs + 1):
                a = (2.0 * math.pi * s) / segs
                fan.append(world + right * math.cos(a) + up * math.sin(a))
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": fan})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        def draw_chain_pts(pts_local, line_col, dot_col, active=False):
            pts = [mw @ p for p in pts_local]
            if len(pts) >= 2:
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
                shader.bind()
                shader.uniform_float("color", (line_col[0], line_col[1], line_col[2], 0.35))
                gpu.state.line_width_set(10.0)
                batch.draw(shader)
                shader.uniform_float("color", line_col)
                gpu.state.line_width_set(3.5)
                batch.draw(shader)
            for i, wco in enumerate(pts):
                col = dot_col
                if active:
                    is_sel = self._sel_has(i, 'co', chain_idx=int(getattr(self, 'active_chain', 0) or 0))
                    is_act = (i == getattr(self, 'active_handle', None))
                    if is_act or is_sel:
                        col = (1.0, 1.0, 0.3, 1.0)
                draw_dot(wco, self._screen_constant_controller_radius(context, wco, 7.0 * ds), col)

        for ch_pts in (getattr(self, 'spine_chains_pts', None) or []):
            if ch_pts:
                draw_chain_pts(ch_pts, (0.2, 0.7, 0.4, 0.85), (0.2, 0.75, 0.4, 0.8), active=False)
        cur = getattr(self, 'spine_points', []) or []
        if cur:
            draw_chain_pts(cur, (0.3, 1.0, 0.5, 1.0), (0.25, 0.95, 0.45, 0.95), active=True)

        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)

    def _screen_constant_controller_radius(self, context, world, pixels=7.0):
        """Return a world-space radius that keeps controller dots visually
        constant in screen pixels while zooming in/out.

        The viewport projection is sampled at the controller position, so this
        also behaves correctly in perspective views where apparent size depends
        on camera distance.
        """
        try:
            region = context.region
            rv3d = context.region_data
            if region is None or rv3d is None:
                return 0.055
            p0 = view3d_utils.location_3d_to_region_2d(region, rv3d, world)
            if p0 is None:
                return 0.055
            right = (rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))).normalized()
            # One world unit along the viewport's horizontal direction.
            p1 = view3d_utils.location_3d_to_region_2d(region, rv3d, world + right)
            if p1 is None:
                return 0.055
            px_per_world = math.hypot(p1.x - p0.x, p1.y - p0.y)
            if px_per_world <= 1e-8:
                return 0.055
            return max(1e-5, float(pixels) / px_per_world)
        except Exception:
            return 0.055

    def _effective_display_scale(self):
        """Spine uses full size; Vertex controllers are ~2 steps smaller."""
        ds = max(0.05, min(5.0, float(getattr(self, 'display_scale', 1.0) or 1.0)))
        if getattr(self, 'tool_mode', 'VERTEX') == 'VERTEX':
            ds *= 0.75
        return ds

    def draw_callback(self, context):
        obj, _ = self.get_obj_bm(context)
        if obj is None:
            return

        if (getattr(self, 'tool_mode', None) == 'SPINE_DEFORM'
                and getattr(self, '_show_influence', False)
                and getattr(self, 'spine_chains', None)):
            try:
                self._draw_influence_overlay(context, obj)
            except Exception:
                pass
        # Spine place / Edit Place: always green (never blue deform colors)
        if getattr(self, 'tool_mode', 'VERTEX') == 'SPINE_PLACE':
            # Edit Place: always draw all deform chains (even while placing a new one)
            if getattr(self, '_spine_edit_place', False):
                self._draw_spine_edit_place_bez(context, obj)
            else:
                self._draw_spine(context, obj)
            return
        if getattr(self, 'tool_mode', 'VERTEX') == 'SPINE_DEFORM':
            # Use original Bezier draw (blue curve + warm controllers) below
            if not getattr(self, 'bez', None):
                self._draw_spine(context, obj)
                return

        mw = obj.matrix_world
        pts = []
        samples = 64
        if not getattr(self, 'bez', None):
            return
        for s in range(samples + 1):
            t = s / samples
            pts.append(mw @ eval_bezier_points(self.bez, t))

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)

        ds = self._effective_display_scale()
        # Other chains (non-active) as dimmer curves
        if getattr(self, 'tool_mode', '') == 'SPINE_DEFORM' and getattr(self, 'spine_chains', None):
            ac = int(getattr(self, 'active_chain', 0) or 0)
            for ci, ch in enumerate(self.spine_chains):
                if ci == ac:
                    continue
                bez = ch.get('bez')
                if not bez or len(bez) < 2:
                    continue
                if bool(ch.get('in_front', True)):
                    gpu.state.depth_test_set('NONE')
                else:
                    gpu.state.depth_test_set('LESS_EQUAL')
                opts = [mw @ eval_bezier_points(bez, s / samples) for s in range(samples + 1)]
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": opts})
                shader.bind()
                shader.uniform_float("color", (0.15, 0.35, 0.7, 0.35))
                gpu.state.line_width_set(6.0 * min(ds, 2.0))
                batch.draw(shader)
                shader.uniform_float("color", (0.25, 0.5, 0.9, 0.75))
                gpu.state.line_width_set(3.0 * min(ds, 2.0))
                batch.draw(shader)
                for bi, bp in enumerate(bez):
                    wco = mw @ bp['co']
                    # Keep non-active controllers in the same screen-space size
                    # system as the active chain.
                    radius = self._screen_constant_controller_radius(context, wco, 7.0 * ds)
                    right = (context.region_data.view_rotation @ Vector((1, 0, 0))).normalized() * radius
                    up = (context.region_data.view_rotation @ Vector((0, 1, 0))).normalized() * radius
                    fan = [wco]
                    for s in range(13):
                        a = (2.0 * math.pi * s) / 12
                        fan.append(wco + right * math.cos(a) + up * math.sin(a))
                    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": fan})
                    shader.bind()
                    if self._sel_has(bi, 'co', chain_idx=ci):
                        shader.uniform_float("color", (1.0, 0.9, 0.2, 0.95))
                    else:
                        shader.uniform_float("color", (0.3, 0.55, 0.95, 0.7))
                    batch.draw(shader)

        # Active chain depth
        _ac_front = True
        if getattr(self, 'tool_mode', '') == 'SPINE_DEFORM' and getattr(self, 'spine_chains', None):
            _ac = int(getattr(self, 'active_chain', 0) or 0)
            if 0 <= _ac < len(self.spine_chains):
                _ac_front = bool(self.spine_chains[_ac].get('in_front', True))
            else:
                _ac_front = bool(getattr(self, 'spine_in_front', True))
        else:
            _ac_front = bool(getattr(self, 'spine_in_front', True))
        gpu.state.depth_test_set('NONE' if _ac_front else 'LESS_EQUAL')

        if len(pts) >= 2:
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
            shader.bind()
            shader.uniform_float("color", (0.1, 0.4, 1.0, 0.4))
            gpu.state.line_width_set(10.0 * min(ds, 2.0))
            batch.draw(shader)
            shader.uniform_float("color", (0.2, 0.7, 1.0, 1.0))
            gpu.state.line_width_set(5.0 * min(ds, 2.0))
            batch.draw(shader)

        rv3d = context.region_data
        segs = 18
        n = len(self.bez)
        active_part = getattr(self, 'active_bez_part', 'co')

        def draw_dot(world, size, color):
            right = (rv3d.view_rotation @ Vector((1, 0, 0))).normalized() * size
            up = (rv3d.view_rotation @ Vector((0, 1, 0))).normalized() * size
            fan = [world]
            for s in range(segs + 1):
                a = (2.0 * math.pi * s) / segs
                fan.append(world + right * math.cos(a) + up * math.sin(a))
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": fan})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        for i, bp in enumerate(self.bez):
            wco = mw @ bp['co']
            # handle lines + tips (OUT only on ends)
            tips = []
            if i == 0:
                tips = [('hr', bp['hr'])]
            elif i == n - 1:
                tips = [('hl', bp['hl'])]
            else:
                tips = [('hl', bp['hl']), ('hr', bp['hr'])]

            mode = self.point_mode(i)
            if mode == 'AUTO':
                line_col = (0.45, 0.75, 1.0, 0.85)
                tip_col = (0.55, 0.8, 1.0, 0.95)
            elif mode == 'ALIGNED':
                line_col = (1.0, 0.7, 0.35, 0.85)
                tip_col = (1.0, 0.75, 0.4, 0.95)
            else:
                line_col = (1.0, 0.35, 0.3, 0.85)
                tip_col = (1.0, 0.4, 0.35, 0.95)

            # Controllers always keep the AUTO controller colors.
            # Handle line/tip colors above remain mode-dependent and untouched.
            co_col = (0.15, 0.55, 1.0, 0.9)
            co_sel = (0.5, 1.0, 1.0, 0.95)

            for part, tip in tips:
                if (tip - bp['co']).length < 1e-8:
                    continue
                wtip = mw @ tip
                batch = batch_for_shader(shader, 'LINES', {"pos": [wco, wtip]})
                shader.bind()
                shader.uniform_float("color", line_col)
                ds = self._effective_display_scale()
                gpu.state.line_width_set(1.5 * min(ds, 2.5))
                batch.draw(shader)
                is_sel = self._sel_has(i, part)
                is_act_tip = (
                    self.active_handle is not None
                    and int(i) == int(self.active_handle)
                    and active_part == part
                )
                # Handle tips keep their mode color when selected/active; selection
                # is shown by a stronger/darker version of that same color.
                if mode == 'FREE':
                    tip_sel_col = (1.0, 0.18, 0.12, 1.0)
                elif mode == 'ALIGNED':
                    tip_sel_col = (1.0, 0.52, 0.08, 1.0)
                else:
                    tip_sel_col = (0.25, 0.62, 1.0, 1.0)
                if is_act_tip or is_sel:
                    draw_dot(wtip, self._screen_constant_controller_radius(context, wtip, 6.0 * ds), tip_sel_col)
                else:
                    draw_dot(wtip, self._screen_constant_controller_radius(context, wtip, 6.0 * ds), tip_col)

            # Controller: active = solid strong blue (no ring); selected = original co_sel; else co_col
            ac_draw = int(getattr(self, 'active_chain', 0) or 0)
            is_sel = self._sel_has(i, 'co', chain_idx=ac_draw)
            is_act = (
                self.active_handle is not None
                and int(i) == int(self.active_handle)
            )
            ds = self._effective_display_scale()
            if is_act:
                draw_dot(wco, self._screen_constant_controller_radius(context, wco, 9.0 * ds), (0.15, 0.45, 1.0, 1.0))
            elif is_sel:
                draw_dot(wco, self._screen_constant_controller_radius(context, wco, 9.0 * ds), co_sel)
            else:
                draw_dot(wco, self._screen_constant_controller_radius(context, wco, 9.0 * ds), co_col)

        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)

        try:
            self._draw_vertex_prop_circle(context, obj)
        except Exception:
            pass

    def _draw_vertex_prop_circle(self, context, obj):
        """Vertex Mode: Blender-sized proportional circle around the drag pivot."""
        if getattr(self, 'tool_mode', 'VERTEX') != 'VERTEX':
            return
        if not getattr(self, 'dragging', False):
            return
        ts = getattr(context, 'tool_settings', None)
        if ts is None or not getattr(ts, 'use_proportional_edit', False):
            return
        radius = max(float(getattr(ts, 'proportional_size', 0.0) or self.prop_size or 0.0), 1e-8)
        self.prop_size = radius
        center_local = getattr(self, '_xform_center', None)
        bez = getattr(self, 'bez', None) or []
        if center_local is None:
            idx = getattr(self, 'active_handle', None)
            if idx is not None and 0 <= int(idx) < len(bez):
                center_local = bez[int(idx)]['co']
            elif getattr(self, 'drag_start_co', None) is not None:
                center_local = self.drag_start_co
            else:
                return
        rv3d = context.region_data
        if rv3d is None or obj is None:
            return
        mw = obj.matrix_world
        view_right_w = (rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))).normalized()
        view_up_w = (rv3d.view_rotation @ Vector((0.0, 1.0, 0.0))).normalized()
        try:
            inv3 = mw.to_3x3().inverted()
        except Exception:
            inv3 = Matrix.Identity(3)
        right_l = inv3 @ view_right_w
        up_l = inv3 @ view_up_w
        if right_l.length < 1e-8 or up_l.length < 1e-8:
            right_l = Vector((1.0, 0.0, 0.0))
            up_l = Vector((0.0, 1.0, 0.0))
        else:
            right_l.normalize()
            up_l.normalize()
        segs = 64
        pts = []
        for i in range(segs + 1):
            a = (2.0 * math.pi * i) / segs
            local = (
                center_local
                + right_l * (math.cos(a) * radius)
                + up_l * (math.sin(a) * radius)
            )
            pts.append(mw @ local)
        # Native prop circle is a muted gray ring (light on dark themes, dark on light).
        try:
            theme_3d = context.preferences.themes[0].view_3d
            bg = theme_3d.space.back
            luma = 0.2126 * float(bg[0]) + 0.7152 * float(bg[1]) + 0.0722 * float(bg[2])
            if luma < 0.45:
                col = (0.36, 0.36, 0.36, 0.96)
            else:
                col = (0.20, 0.20, 0.20, 0.96)
        except Exception:
            col = (0.36, 0.36, 0.36, 0.96)
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
        shader.bind()
        shader.uniform_float("color", col)
        gpu.state.line_width_set(3.0)
        batch.draw(shader)
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)

    def execute(self, context):
        return self.invoke(context, None)


classes = (
    VDH_Preferences,
    MESH_OT_vdh_smooth_choice,
    MESH_OT_vdh_handle_type,
    MESH_OT_vdh_influence_falloff,
    MESH_OT_vdh_straighten_tube,
    MESH_OT_vdh_set_initial,
    MESH_OT_vdh_clear_cache,
    MESH_OT_vdh_mirror_chain,
    MESH_OT_vertex_deform_handles,
)
addon_keymaps = []


# Spine-only Weight Paint transfer/normalization.
#
# Desired Spine behavior: when painting one BH_Spine* group, that group owns
# the painted amount and the other BH_Spine* groups give up the same amount.
# Thus painting a vertex to 1.0 in the active Spine group makes all other
# Spine groups 0.0 at that vertex.  Partial paint keeps the other groups in
# their previous proportions.
_bh_wp_pending_obj = None
_bh_wp_last_update = 0.0
_bh_wp_timer_running = False
_bh_wp_busy = False
_bh_wp_ignore_until = 0.0
_bh_wp_snapshots = {}
_bh_wp_state_obj_key = None
_bh_wp_state_active = -1


def _vdh_spine_weightpaint_snapshot(obj):
    """Cache the complete previous Spine-weight state for the active group.

    Keeping all Spine groups in the snapshot is important: when the active
    group's paint changes, the redistribution must be based on the *previous*
    non-active weights, not whatever Blender (or a previous pass) happens to
    have in those groups at the moment the timer runs.
    """
    if obj is None or getattr(obj, 'type', None) != 'MESH':
        return False
    if getattr(obj, 'mode', None) != 'WEIGHT_PAINT':
        return False
    try:
        active = int(obj.vertex_groups.active_index)
        if active < 0 or active >= len(obj.vertex_groups):
            return False
        av = obj.vertex_groups[active]
        if not _vdh_is_spine_vg_name(av.name):
            return False

        spine_groups = [vg for vg in obj.vertex_groups
                        if _vdh_is_spine_vg_name(vg.name)]
        if not spine_groups:
            return False

        # One mesh pass per snapshot, but only when entering/switching group
        # or after a completed paint transfer. This is deliberately not done
        # on every depsgraph update.
        arrays = {}
        n = len(obj.data.vertices)
        for vg in spine_groups:
            arr = [0.0] * n
            for v in obj.data.vertices:
                try:
                    arr[v.index] = max(0.0, min(1.0, float(vg.weight(v.index))))
                except Exception:
                    pass
            arrays[int(vg.index)] = arr

        _bh_wp_snapshots[obj.as_pointer()] = (active, arrays)
        return True
    except Exception:
        return False


def _vdh_spine_weightpaint_transfer(obj):
    """Make Spine Weight Paint exclusive while preserving other-group ratios.

    If the active group changes from A_old to A_new, the old non-active Spine
    weights are rescaled so that their new sum is 1-A_new. Examples:

      A=.6, B=.4  -> paint A=.7  -> A=.7, B=.3
      A=.6, B=.3, C=.1 -> paint A=1 -> A=1, B=0, C=0
      A=.6, B=.2, C=.2 -> paint A=.3 -> A=.3, B=.35, C=.35

    If there were no previous non-active Spine weights, there is no meaningful
    source from which to manufacture the remaining weight, so the other groups
    stay zero and the active group keeps the painted value.
    """
    global _bh_wp_busy
    if _bh_wp_busy or obj is None or getattr(obj, 'type', None) != 'MESH':
        return False
    if getattr(obj, 'mode', None) != 'WEIGHT_PAINT':
        return False

    try:
        active = int(obj.vertex_groups.active_index)
        if active < 0 or active >= len(obj.vertex_groups):
            return False
        av = obj.vertex_groups[active]
        if not _vdh_is_spine_vg_name(av.name):
            return False
        vg_indices = [int(vg.index) for vg in obj.vertex_groups
                      if _vdh_is_spine_vg_name(vg.name) and int(vg.index) != active]
    except Exception:
        return False

    if not vg_indices:
        _vdh_spine_weightpaint_snapshot(obj)
        return False

    key = obj.as_pointer()
    snap = _bh_wp_snapshots.get(key)
    if not snap or int(snap[0]) != active or not isinstance(snap[1], dict):
        _vdh_spine_weightpaint_snapshot(obj)
        return False

    old_active = snap[1].get(active)
    if old_active is None:
        _vdh_spine_weightpaint_snapshot(obj)
        return False

    _bh_wp_busy = True
    changed = False
    try:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        dl = bm.verts.layers.deform.verify()
        eps = 1e-7

        # The redistribution is based on the cached *old* state of every
        # non-active Spine group. This makes the result independent of whether
        # Blender's own Weight Paint normalization has already modified them.
        old_other = {
            gi: snap[1].get(gi, []) for gi in vg_indices
        }

        for v in bm.verts:
            vi = v.index
            try:
                new_a = max(0.0, min(1.0, float(v[dl].get(active, 0.0))))
            except Exception:
                new_a = 0.0
            if vi >= len(old_active):
                continue
            old_a = float(old_active[vi])
            if abs(new_a - old_a) <= eps:
                continue

            old_values = []
            old_other_total = 0.0
            for gi in vg_indices:
                arr = old_other.get(gi) or []
                ow = float(arr[vi]) if vi < len(arr) else 0.0
                ow = max(0.0, min(1.0, ow))
                old_values.append((gi, ow))
                old_other_total += ow

            target_other = max(0.0, min(1.0, 1.0 - new_a))

            if old_other_total > eps:
                scale = target_other / old_other_total
                for gi, ow in old_values:
                    nw = max(0.0, min(1.0, ow * scale))
                    try:
                        if nw <= eps:
                            if gi in v[dl]:
                                v[dl][gi] = 0.0
                        else:
                            v[dl][gi] = nw
                    except Exception:
                        pass
            else:
                # No previous competing Spine weight exists. Do not invent a
                # destination group; clear stale competitors if any exist.
                for gi in vg_indices:
                    try:
                        if gi in v[dl]:
                            v[dl][gi] = 0.0
                    except Exception:
                        pass
            changed = True

        if changed:
            bm.to_mesh(obj.data)
            obj.data.update()
            # Notify the Spine evaluator that Blender Weight Paint changed the
            # runtime source weights. The operator will invalidate its _fw
            # caches before the next Radius/controller evaluation.
            try:
                obj.data['_bh_spine_weightpaint_dirty'] = True
            except Exception:
                pass
        bm.free()
        return changed
    except Exception:
        try:
            bm.free()
        except Exception:
            pass
        return False
    finally:
        _bh_wp_busy = False


def _vdh_spine_weightpaint_transfer_timer():
    global _bh_wp_timer_running, _bh_wp_pending_obj, _bh_wp_last_update
    global _bh_wp_ignore_until
    obj = _bh_wp_pending_obj
    if obj is None:
        _bh_wp_timer_running = False
        return None
    now = time.monotonic()
    if now - _bh_wp_last_update < 0.10:
        return 0.04
    _bh_wp_pending_obj = None
    _bh_wp_timer_running = False
    try:
        if getattr(obj, 'mode', None) == 'WEIGHT_PAINT':
            _vdh_spine_weightpaint_transfer(obj)
            # New state becomes the baseline for the next stroke.
            _vdh_spine_weightpaint_snapshot(obj)
            _bh_wp_ignore_until = time.monotonic() + 0.15
    except Exception:
        pass
    return None


def _vdh_spine_weightpaint_auto_normalize(scene=None, depsgraph=None):
    """Lightweight Spine-only watcher; actual mesh work is debounced."""
    global _bh_wp_pending_obj, _bh_wp_last_update, _bh_wp_timer_running
    global _bh_wp_state_obj_key, _bh_wp_state_active
    try:
        obj = bpy.context.object
        if obj is None or getattr(obj, 'type', None) != 'MESH':
            return
        if getattr(obj, 'mode', None) != 'WEIGHT_PAINT':
            return
        if not any(_vdh_is_spine_vg_name(vg.name) for vg in obj.vertex_groups):
            return

        active = int(getattr(obj.vertex_groups, 'active_index', -1))
        if active < 0 or active >= len(obj.vertex_groups):
            return
        if not _vdh_is_spine_vg_name(obj.vertex_groups[active].name):
            return

        key = obj.as_pointer()
        snap = _bh_wp_snapshots.get(key)
        # Establish a baseline on entering Weight Paint / changing the active
        # Spine group. Do this before arming a transfer for the new group.
        if (snap is None or snap[0] != active or
                len(obj.data.vertices) != len(next(iter(snap[1].values()), []))):
            _vdh_spine_weightpaint_snapshot(obj)
            _bh_wp_state_obj_key = key
            _bh_wp_state_active = active
            return

        if time.monotonic() < _bh_wp_ignore_until:
            return

        _bh_wp_state_obj_key = key
        _bh_wp_state_active = active
        _bh_wp_pending_obj = obj
        _bh_wp_last_update = time.monotonic()
        if not _bh_wp_timer_running:
            _bh_wp_timer_running = True
            try:
                bpy.app.timers.register(_vdh_spine_weightpaint_transfer_timer, first_interval=0.04)
            except Exception:
                _bh_wp_timer_running = False
    except Exception:
        pass


def draw_vd_header_button(self, context):
    """BH (Blue Handles) button on the Tool Header in Edit Mode."""
    if context.mode != 'EDIT_MESH':
        return
    layout = self.layout
    layout.separator()
    row = layout.row(align=True)
    row.operator(
        "mesh.vertex_deform_handles",
        text="BH",
        icon='CURVE_BEZCURVE',
    )


def _bh_spine_geometry_layer_wrapper(self, method, context, *args, **kwargs):
    """Central final-deformation layer for Spine geometry operations.

    E/W/Q are pure geometry edits.  Tilt and Shrink/Inflate are kept
    outside those operations: the mesh is evaluated once with both attributes
    neutralized, the geometry operation updates Rest/Bind, then the saved
    attributes are restored and applied exactly once to the new base geometry.
    """
    chains = getattr(self, 'spine_chains', None) or []
    if getattr(self, 'tool_mode', '') != 'SPINE_DEFORM' or not chains:
        return method(self, context, *args, **kwargs)

    saved = []
    for ch in chains:
        saved.append((
            ch,
            list(ch.get('tilt') or []),
            list(ch.get('radius') or []),
            str(ch.get('attr_interp') or 'SMOOTH').upper(),
        ))

    saved_props = (
        list(getattr(self, 'spine_tilt', []) or []),
        list(getattr(self, 'spine_radius', []) or []),
        str(getattr(self, 'spine_attr_interp', 'SMOOTH') or 'SMOOTH').upper(),
        getattr(self, '_xform_mode', None),
    )

    try:
        # BASE LAYER: hide final attributes from the geometry operation.
        self._xform_mode = 'GEOMETRY_LAYER'
        for ch, _, _, _ in saved:
            n = len(ch.get('bez') or [])
            ch['tilt'] = [0.0] * n
            ch['radius'] = [1.0] * n
            ch['_fw'] = None
            ch['_sw_key'] = None
            ch['_soft_w'] = None
        self._influence_overlay_cache = None
        self._spine_apply(context, auto_soft=False)

        # PURE GEOMETRY: E/W/Q may rebind, but only against base mesh.
        return method(self, context, *args, **kwargs)
    finally:
        # Restore the persistent attribute layer regardless of operation result.
        for ch, tilt, radius, attr_interp in saved:
            ch['tilt'] = list(tilt)
            ch['radius'] = list(radius)
            ch['attr_interp'] = attr_interp
            ch['_fw'] = None
            ch['_sw_key'] = None
            ch['_soft_w'] = None
        self._influence_overlay_cache = None
        self._xform_mode = saved_props[3]

        try:
            self._spine_load_active_chain()
        except Exception:
            self.spine_tilt = list(saved_props[0])
            self.spine_radius = list(saved_props[1])
            self.spine_attr_interp = saved_props[2]

        # FINAL LAYER: apply the user's Tilt + Shrink/Inflate once.
        try:
            self._spine_apply(context, auto_soft=False)
        except Exception:
            pass


def _bh_wrap_spine_geometry_methods():
    """Wrap Spine geometry tools with the centralized final layer."""
    cls = MESH_OT_vertex_deform_handles
    names = (
        '_spine_align_rings_to_axis',
        '_spine_align_longitudinal_loops_to_axis',
        '_spine_circularize_rings_to_axis',
        '_spine_uniformize_thickness',
    )
    for name in names:
        original = getattr(cls, name, None)
        if original is None or getattr(original, '_bh_final_layer_wrapped', False):
            continue
        def make_wrapper(fn):
            def wrapped(self, context, *args, **kwargs):
                return _bh_spine_geometry_layer_wrapper(self, fn, context, *args, **kwargs)
            wrapped._bh_final_layer_wrapped = True
            wrapped.__name__ = getattr(fn, '__name__', 'spine_geometry_operation')
            wrapped.__doc__ = getattr(fn, '__doc__', None)
            return wrapped
        setattr(cls, name, make_wrapper(original))


_bh_wrap_spine_geometry_methods()
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    try:
        if _vdh_spine_weightpaint_auto_normalize not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(_vdh_spine_weightpaint_auto_normalize)
    except Exception:
        pass
    bpy.types.VIEW3D_HT_tool_header.append(draw_vd_header_button)
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Mesh', space_type='EMPTY')
        kmi = km.keymap_items.new(
            "mesh.vertex_deform_handles",
            type='D',
            value='PRESS',
            ctrl=True, shift=True, alt=True,
        )
        addon_keymaps.append((km, kmi))


def unregister():
    try:
        if _vdh_spine_weightpaint_auto_normalize in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(_vdh_spine_weightpaint_auto_normalize)
    except Exception:
        pass
    try:
        bpy.types.VIEW3D_HT_tool_header.remove(draw_vd_header_button)
    except Exception:
        pass
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
