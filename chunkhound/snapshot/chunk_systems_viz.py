# ruff: noqa: E101, E501

"""Self-contained HTML visualization for snapshot chunk-systems adjacency graph.

This renderer is intentionally dependency-free and deterministic. Emission is gated
behind CLI flags so the baseline snapshot output remains unchanged.
"""

from __future__ import annotations

import json


def _safe_json_for_html(obj: object) -> str:
    # Prevent accidentally terminating the <script> tag.
    # Escaping "<" is sufficient because it covers "</script" and friends.
    raw = json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return raw.replace("<", "\\u003c")


def render_chunk_systems_viz_html(
    *,
    adjacency_payload: dict[str, object],
    system_metrics: dict[int, dict[str, float | int | None]] | None = None,
    chunk_system_labels: dict[int, dict[str, object]] | None = None,
    system_groups_payload: dict[str, object] | None = None,
    system_group_labels_payload: dict[str, object] | None = None,
) -> str:
    data = {
        "adjacency": adjacency_payload,
        "system_metrics": system_metrics or {},
        "chunk_system_labels": chunk_system_labels or {},
        "system_groups": system_groups_payload or {},
        "system_group_labels": system_group_labels_payload or {},
    }
    data_json = _safe_json_for_html(data)

    # NOTE: Keep everything in one file for easy sharing and offline viewing.
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ChunkHound Snapshot: Chunk Systems</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: rgba(255,255,255,0.08);
      --panel2: rgba(255,255,255,0.06);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --accent: #8ee3ff;
      --warn: #ffb86b;
      --edge: rgba(255,255,255,0.25);
      --edge-hi: rgba(142,227,255,0.85);
      --node: rgba(255,255,255,0.88);
      --node-sel: rgba(255,184,107,0.95);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(1000px 700px at 10% 15%, rgba(142,227,255,0.16), rgba(0,0,0,0)),
        radial-gradient(900px 650px at 85% 20%, rgba(255,184,107,0.14), rgba(0,0,0,0)),
        linear-gradient(180deg, #070a14, #0b1020 55%, #070a14);
      height: 100vh;
      overflow: hidden;
    }}
    .wrap {{
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 14px;
      padding: 14px;
      height: 100vh;
    }}
    .card {{
      border: 1px solid rgba(255,255,255,0.10);
      background: var(--panel);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      min-height: 0;
    }}
    .graph {{
      position: relative;
      padding: 0;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
      opacity: 1;
    }}
    .loading {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(7,10,20,0.65);
      backdrop-filter: blur(8px);
      font-family: var(--mono);
      font-size: 12px;
      color: rgba(255,255,255,0.75);
      letter-spacing: 0.2px;
      z-index: 2;
    }}
    .hud {{
      position: absolute;
      left: 12px;
      top: 12px;
      right: 12px;
      display: flex;
      gap: 10px;
      align-items: center;
      background: rgba(7,10,20,0.55);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 12px;
      padding: 10px;
      backdrop-filter: blur(10px);
    }}
    .hud button {{
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.06);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      cursor: pointer;
      font-size: 12px;
      line-height: 1;
      white-space: nowrap;
    }}
    .hud button:hover {{
      border-color: rgba(142,227,255,0.35);
    }}
    .hud input[type="text"] {{
      flex: 1;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.06);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      outline: none;
    }}
    .hud .ctrl {{
      display: flex;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .hud .readout {{
      font-family: var(--mono);
      font-size: 12px;
      color: rgba(255,255,255,0.70);
      padding: 0 6px;
      white-space: nowrap;
    }}
	    .hud input[type="range"] {{
	      width: 160px;
	    }}
	    .hud select {{
	      border: 1px solid rgba(255,255,255,0.16);
	      background: rgba(255,255,255,0.06);
	      color: rgba(255,255,255,0.88);
	      border-radius: 10px;
	      padding: 6px 10px;
	      font-size: 12px;
	      font-family: var(--mono);
	      outline: none;
	    }}
	    .hud select:focus {{
	      border-color: rgba(142,227,255,0.55);
	      box-shadow: 0 0 0 3px rgba(142,227,255,0.12);
	    }}
    .hud label {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      user-select: none;
    }}
    .side {{
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
    }}
    .side header {{
      padding: 14px 14px 10px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.05);
    }}
    .side header .title {{
      font-size: 14px;
      letter-spacing: 0.2px;
      color: var(--text);
      margin: 0 0 6px 0;
    }}
    .side header .meta {{
      font-size: 12px;
      color: var(--muted);
      font-family: var(--mono);
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .panel {{
      padding: 14px;
      overflow: auto;
    }}
    .pill {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      font-size: 12px;
      color: var(--muted);
      margin-right: 8px;
      margin-bottom: 8px;
    }}
    .pill b {{
      color: var(--text);
      font-weight: 600;
    }}
    .section {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid rgba(255,255,255,0.10);
    }}
    .section h3 {{
      margin: 0 0 8px 0;
      font-size: 12px;
      color: rgba(255,255,255,0.80);
      letter-spacing: 0.25px;
      text-transform: uppercase;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 8px 10px;
      font-size: 12px;
      color: var(--muted);
      font-family: var(--mono);
      overflow-wrap: anywhere;
    }}
    .kv .k {{ color: rgba(255,255,255,0.75); }}
    .kv .v {{ color: rgba(255,255,255,0.92); }}
    .list {{
      display: grid;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
      font-family: var(--mono);
    }}
    .row {{
      padding: 10px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.10);
      background: var(--panel2);
      cursor: pointer;
    }}
    .row:hover {{
      border-color: rgba(142,227,255,0.35);
    }}
    .row .top {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
    }}
    .row .top .id {{
      color: var(--text);
      font-weight: 600;
    }}
    .row .sub {{
      margin-top: 6px;
      color: rgba(255,255,255,0.70);
      overflow-wrap: anywhere;
      line-height: 1.35;
    }}
    .hint {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}
    @media (max-width: 980px) {{
      body {{ overflow: auto; height: auto; }}
      .wrap {{
        grid-template-columns: 1fr;
        height: auto;
      }}
      .card {{ height: 70vh; }}
    }}
  </style>
</head>
<body>
	  <div class="wrap">
	    <div class="card graph" id="graphCard">
	      <canvas id="cv"></canvas>
	      <div class="loading" id="loading">
	        <div style="display:flex;flex-direction:column;gap:10px;align-items:center;">
	          <div>Laying out graph...</div>
	          <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;">
	            <button id="skipLayoutBtn" title="Show immediately without running the layout">Skip layout</button>
	          </div>
	        </div>
	      </div>
	      <div class="hud">
	        <input id="search" type="text" placeholder="Search systems (label, top_files, top_symbols)..." />
	        <button id="fitBtn" title="Fit graph to view (F)">Fit</button>
	        <button id="resetBtn" title="Reset view (0)">Reset</button>
	        <div class="ctrl">
	          <span>Layout</span>
		          <select id="layoutSel" title="Choose layout mode">
		            <option value="force" selected>Force</option>
		            <option value="kcore">K-core layers</option>
		            <option value="hub">Hubs (radial)</option>
		            <option value="hub_grav">Hubs (gravity, implied)</option>
		          </select>
		        </div>
	        <div class="ctrl">
	          <span>Edge threshold</span>
	          <input id="thr" type="range" min="0" max="100" value="0" />
	        </div>
        <div class="ctrl">
          <span>Groups</span>
          <select id="groupsSel" title="Overlay systems-of-systems groupings">
            <option value="off" selected>Off</option>
            <option value="meta">Meta-Leiden</option>
          </select>
        </div>
        <div class="ctrl">
          <span>Granularity</span>
          <input id="grp" type="range" min="0" max="0" value="0" />
        </div>
        <div class="ctrl">
          <label><input id="nbrOnly" type="checkbox" /> neighborhood only</label>
        </div>
        <div class="ctrl">
          <label><input id="showLabels" type="checkbox" checked /> labels</label>
        </div>
        <div class="readout" id="camReadout"></div>
        <div class="readout" id="grpReadout"></div>
      </div>
    </div>
    <div class="card side">
      <header>
        <div class="title">Chunk Systems Graph</div>
        <div class="meta" id="metaLine"></div>
      </header>
      <div class="panel" id="panel">
        <div class="hint">
          Wheel to zoom. Drag background to pan. Shift+drag background to rotate. Drag nodes to reposition.
          Click a node to see system details and its strongest neighbors. Click a link to see evidence edges.
        </div>
        <div class="section">
          <h3>Legend</h3>
          <div class="hint">
            Nodes are systems (clusters). Links are aggregated cross-system chunk edges (score: weight_sum).
          </div>
        </div>
      </div>
    </div>
  </div>

	  <script type="application/json" id="chunkhound-data">{data_json}</script>
	  <script>
	  (() => {{
	    // NOTE: Keep boot lightweight so the page 'load' event fires quickly even for large graphs.
	    // We'll parse and initialize the visualization after load/idle.
	    const cv = document.getElementById("cv");
	    const card = document.getElementById("graphCard");
	    const ctx = cv.getContext("2d");
	    const camReadoutEl = document.getElementById("camReadout");
	    const grpReadoutEl = document.getElementById("grpReadout");
	    const nbrOnlyEl = document.getElementById("nbrOnly");
	    const showLabelsEl = document.getElementById("showLabels");
	    const loadingEl = document.getElementById("loading");
	    const skipLayoutBtn = document.getElementById("skipLayoutBtn");
	    const layoutSel = document.getElementById("layoutSel");
	    const groupsSel = document.getElementById("groupsSel");
	    const grpEl = document.getElementById("grp");
	    const metaLineEl = document.getElementById("metaLine");

	    if (metaLineEl) metaLineEl.textContent = "loading...";

	    let booted = false;
	    let skipInitialLayoutRequested = false;
	    let cancelInitialLayout = false;
	    let activeLayout = "force";
	    function boot() {{
	      if (booted) return;
	      booted = true;

	      const data = JSON.parse(document.getElementById("chunkhound-data").textContent);
	      const adjacency = data.adjacency || {{}};
	      const systemGroups = data.system_groups || {{}};
	      const systemGroupLabels = data.system_group_labels || {{}};
	      const chunkSystemLabels = data.chunk_system_labels || {{}};
	      function getChunkSystemLabelRow(clusterId) {{
	        const key = String(clusterId);
	        const row = chunkSystemLabels[key] || chunkSystemLabels[clusterId];
	        if (!row || typeof row !== "object") return null;
	        return row;
	      }}
	      const systems = (adjacency.systems || []).map(s => {{
	        const sc = {{...s}};
	        const cid = Number(sc.cluster_id);
	        const heur = String(sc.label || "").trim();
	        const row = getChunkSystemLabelRow(cid);
	        const llmLabel = row ? String(row.label || "").trim() : "";
	        const confRaw = row ? row.confidence : null;
	        let conf = null;
	        if (confRaw === 0 || confRaw) {{
	          const cn = Number(confRaw);
	          conf = isFinite(cn) ? cn : null;
	        }}
	        sc.label_heuristic = heur;
	        sc.label_llm = llmLabel;
	        sc.label_confidence = conf;
	        sc.label = llmLabel || heur;
	        return sc;
	      }});
	      const linksRaw = (adjacency.links || []).map(l => ({{...l}}));
	      const metrics = data.system_metrics || {{}};

	      const sysById = new Map();
	      for (const s of systems) sysById.set(Number(s.cluster_id), s);

	      const schema = String(adjacency.schema_version || "");
	      const rev = String(adjacency.schema_revision || "");
	      const isDirected = !!adjacency.directed || schema.includes("directed");
	      const trunc = adjacency.truncation || {{}};
	      if (metaLineEl) {{
	        metaLineEl.textContent =
	          `schema=${{schema}} rev=${{rev}} mode=${{isDirected ? "directed" : "mutual"}} links=${{linksRaw.length}} (before=${{Number(trunc.links_before||0)}} after=${{Number(trunc.links_after||0)}})`;
	      }}

	      const groupPartitions = Array.isArray(systemGroups.partitions) ? systemGroups.partitions : [];
	      const groupLabelPartitions = Array.isArray(systemGroupLabels.partitions) ? systemGroupLabels.partitions : [];
	      let groupsMode = "off";
	      let groupPartitionIndex = 0;
	      const groupBySystemId = new Map(); // system_id -> group_id (for current partition)
	      const groupSizeById = new Map(); // group_id -> member count
	      const groupLabelById = new Map(); // group_id -> label row (for current partition)

	      function setGroupsMode(mode) {{
	        const m = String(mode || "off");
	        groupsMode = (m === "meta") ? "meta" : "off";
	        refreshGrouping();
	      }}

	      function setGroupPartitionIndex(idx) {{
	        const i = Math.max(0, Math.min(Number(idx || 0), Math.max(0, groupPartitions.length - 1)));
	        groupPartitionIndex = i;
	        refreshGrouping();
	      }}

	      function refreshGrouping() {{
	        groupBySystemId.clear();
	        groupSizeById.clear();
	        groupLabelById.clear();
	        if (groupsMode === "off" || !groupPartitions.length) {{
	          for (const n of nodes) n._group = 0;
	          if (grpReadoutEl) grpReadoutEl.textContent = "";
	          scheduleDraw();
	          if (selectedNode) renderSystemPanel(selectedNode);
	          return;
	        }}
	        const part = groupPartitions[groupPartitionIndex] || null;
	        const membership = part && Array.isArray(part.membership) ? part.membership : [];
	        const labelPart = groupLabelPartitions[groupPartitionIndex] || null;
	        const labelsArr = labelPart && Array.isArray(labelPart.labels) ? labelPart.labels : [];
	        for (let i = 0; i < labelsArr.length; i++) {{
	          const row = labelsArr[i];
	          if (row && typeof row === "object") {{
	            groupLabelById.set(i + 1, row);
	          }}
	        }}
	        // systems order is by cluster_id in payload; nodes are built from systems in that order.
	        for (let i = 0; i < nodes.length; i++) {{
	          const gid = Number(membership[i] || 0) || 0;
	          const n = nodes[i];
	          n._group = gid;
	          groupBySystemId.set(n.id, gid);
	          if (gid) groupSizeById.set(gid, Number(groupSizeById.get(gid) || 0) + 1);
	        }}
	        const res = part ? Number(part.resolution || 0) : 0;
	        const gc = part ? Number(part.group_count || 0) : 0;
	        if (grpReadoutEl) grpReadoutEl.textContent = `groups r=${{fmtNum(res)}} g=${{gc}}`;
	        scheduleDraw();
	        if (selectedNode) renderSystemPanel(selectedNode);
	      }}

	    function clamp(x, a, b) {{ return Math.max(a, Math.min(b, x)); }}
	    function lerp(a, b, t) {{ return a + (b - a) * t; }}

    function hash32(n) {{
      n = (n | 0) + 0x7ed55d16 + (n << 12);
      n = (n ^ 0xc761c23c) ^ (n >>> 19);
      n = (n + 0x165667b1) + (n << 5);
      n = (n + 0xd3a2646c) ^ (n << 9);
      n = (n + 0xfd7046c5) + (n << 3);
      n = (n ^ 0xb55a4f09) ^ (n >>> 16);
      return n >>> 0;
    }}
	    function rand01(seed) {{
	      return (hash32(seed) % 1000000) / 1000000;
	    }}

	    function groupColor(gid, alpha) {{
	      const g = Number(gid || 0) | 0;
	      const a = (alpha === 0 || alpha) ? Number(alpha) : 1.0;
	      const hue = (hash32(g * 2654435761) % 360);
	      return `hsla(${{hue}}, 78%, 62%, ${{Math.max(0, Math.min(1, a))}})`;
	    }}

	      function shortText(text, maxLen) {{
	        const s = String(text || "").trim();
	        const m = Math.max(1, Number(maxLen || 24));
	        if (!s) return "";
	        if (s.length <= m) return s;
	        return s.slice(0, Math.max(0, m - 1)).trimEnd() + "…";
	      }}

	      // Build nodes.
	      const nodes = systems.map(s => {{
	      const id = Number(s.cluster_id);
	      const size = Number(s.size || 0);
	      const label = String(s.label || "");
	      const h = hash32(id);
	      const lblShort = shortText(label, 28);
	      return {{
	        id,
	        size,
	        label,
	        label_short: lblShort,
	        top_files: s.top_files || [],
	        top_symbols: s.top_symbols || [],
	        mx: metrics[String(id)] || metrics[id] || null,
	        _group: 0,
	        _wdeg: 0,
	        _kcore: 0,
	        x: 0, y: 0, vx: 0, vy: 0,
	        fx: null, fy: null,
	        _h: h,
	        _match: true,
	      }};
	      }});
	      const nodeById = new Map(nodes.map(n => [n.id, n]));

	      // Links.
	      const links = linksRaw
	      .map(l => {{
	        const a = isDirected ? Number(l.source) : Number(l.a);
	        const b = isDirected ? Number(l.target) : Number(l.b);
	        return {{
	          a, b,
	          wsum: Number(l.weight_sum || 0),
	          wmax: Number(l.weight_max || 0),
	          edge_count: Number(l.edge_count || 0),
	          evidence: l.evidence || [],
	          directed: isDirected,
	          _sel: false,
	          _hide: false,
	        }};
	      }})
	      .filter(l => nodeById.has(l.a) && nodeById.has(l.b) && l.a !== l.b);

	      const pairCount = new Map(); // undirected pair key -> number of directed edges present
	      if (isDirected) {{
	        for (const l of links) {{
	          const u = Math.min(l.a, l.b);
	          const v = Math.max(l.a, l.b);
	          const key = `${{u}}|${{v}}`;
	          pairCount.set(key, Number(pairCount.get(key) || 0) + 1);
	        }}
	      }}

	      // Precompute adjacency for metrics and for k-core layout. This is based on ALL links
	      // (not threshold-filtered), so changing the Edge threshold does not re-layer nodes.
	      const nbrs = new Map(); // id -> Array of neighbor objects (fields: id, w)
	      function pushNbr(a, b, w) {{
	        let arr = nbrs.get(a);
	        if (!arr) {{ arr = []; nbrs.set(a, arr); }}
	        arr.push({{ id: b, w }});
	      }}
	      for (const l of links) {{
	        pushNbr(l.a, l.b, l.wsum);
	        pushNbr(l.b, l.a, l.wsum);
	      }}
	      for (const n of nodes) {{
	        const arr = nbrs.get(n.id) || [];
	        let sum = 0;
	        for (const e of arr) sum += Number(e.w || 0);
	        n._wdeg = sum;
	      }}

	    let wMin = Infinity, wMax = 0;
	    for (const l of links) {{
	      wMin = Math.min(wMin, l.wsum);
      wMax = Math.max(wMax, l.wsum);
    }}
    if (!isFinite(wMin)) wMin = 0;

	    let selectedNode = null;
	    let selectedLink = null;
	    let selectedNbrWeights = null;
	    let selectedNbrMaxW = 0;

    // Camera maps world coords (node positions) to screen coords (canvas pixels).
    const cam = {{
      scale: 1.0,
      theta: 0.0,
      panX: 0.0,
      panY: 0.0,
    }};
		    let userMovedCamera = false;
		    let rafPending = false;
	    const layoutPositions = {{
	      force: null, // Map id -> position object
	      kcore: null, // Map id -> position object
	      hub: null, // Map id -> position object
	      hub_grav: null, // Map id -> position object
	    }};
	    let hubLayoutMeta = null;

	    function refreshSelectedNbrWeights() {{
	      if (activeLayout !== "hub_grav" || !selectedNode) {{
	        selectedNbrWeights = null;
	        selectedNbrMaxW = 0;
	        return;
	      }}
	      const m = new Map();
	      let maxW = 0;
	      for (const l of links) {{
	        if (l._hide) continue;
	        if (l.a === selectedNode.id) {{
	          m.set(l.b, l.wsum);
	          maxW = Math.max(maxW, l.wsum);
	        }} else if (l.b === selectedNode.id) {{
	          m.set(l.a, l.wsum);
	          maxW = Math.max(maxW, l.wsum);
	        }}
	      }}
	      selectedNbrWeights = m;
	      selectedNbrMaxW = maxW;
	    }}

	    function scheduleDraw() {{
	      if (rafPending) return;
	      rafPending = true;
      requestAnimationFrame(() => {{
        rafPending = false;
        draw();
      }});
    }}

	    function snapshotPositions() {{
	      const m = new Map();
	      for (const n of nodes) m.set(n.id, {{ x: n.x, y: n.y }});
	      return m;
	    }}
	    function restorePositions(pos) {{
	      if (!pos) return;
	      for (const n of nodes) {{
	        const p = pos.get(n.id);
	        if (!p) continue;
	        n.x = p.x; n.y = p.y;
	        n.vx = 0; n.vy = 0;
	      }}
	    }}

		    function resize() {{
	      const r = card.getBoundingClientRect();
	      const dpr = window.devicePixelRatio || 1;
      cv.width = Math.floor(r.width * dpr);
      cv.height = Math.floor(r.height * dpr);
      cv.style.width = r.width + "px";
      cv.style.height = r.height + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

	      // Seed initial positions deterministically on a loose ring.
	      // Keep the radius modest so we don't require extreme zoom-out on large graphs.
	      const radius = 80 + Math.sqrt(Math.max(1, nodes.length)) * 45;
	      const order = [...nodes].sort((a, b) => a._h - b._h);
	      for (let i = 0; i < order.length; i++) {{
	        const n = order[i];
	        if (n.x !== 0 || n.y !== 0) continue;
	        const t = (i / Math.max(1, order.length)) * Math.PI * 2;
	        const jitter = (rand01(n._h) - 0.5) * radius * 0.12;
	        n.x = Math.cos(t) * (radius + jitter);
	        n.y = Math.sin(t) * (radius + jitter);
	      }}
	    }}
	    resize();
	    window.addEventListener("resize", () => {{
	      resize();
	      scheduleDraw();
	    }});

    function setPanelHtml(html) {{
      document.getElementById("panel").innerHTML = html;
      // Wire click handlers for generated rows.
      for (const el of document.querySelectorAll("[data-node-id]")) {{
        el.addEventListener("click", () => {{
          const id = Number(el.getAttribute("data-node-id"));
          selectNode(id);
        }});
      }}
      for (const el of document.querySelectorAll("[data-link-key]")) {{
        el.addEventListener("click", () => {{
          const key = String(el.getAttribute("data-link-key") || "");
          const [a, b] = key.split(":").map(Number);
          selectLink(a, b);
        }});
      }}
    }}

    function fmtNum(x) {{
      if (x === null || x === undefined) return "n/a";
      const n = Number(x);
      if (!isFinite(n)) return "n/a";
      if (Math.abs(n) >= 1000) return n.toFixed(0);
      if (Math.abs(n) >= 10) return n.toFixed(2);
      return n.toFixed(3);
    }}

    function systemTitle(s) {{
      const id = Number(s.cluster_id);
      const label = String(s.label || "").trim();
      return label ? `#${{id}} ${{label}}` : `#${{id}}`;
    }}

	    function renderSystemPanel(node) {{
	      const s = sysById.get(node.id) || {{cluster_id: node.id, size: node.size, label: node.label, top_files: [], top_symbols: []}};
	      const mx = node.mx || {{}};
	      const title = systemTitle(s);
      const tf = (s.top_files || []).slice(0, 10);
      const ts = (s.top_symbols || []).slice(0, 10);

      function fmtTopFile(x) {{
        if (x === null || x === undefined) return {{ text: "", count: null }};
        if (typeof x === "string") return {{ text: x, count: null }};
        if (typeof x === "object") {{
          const p = x.path ? String(x.path) : "";
          const c = (x.count === 0 || x.count) ? Number(x.count) : null;
          return {{ text: p || "[object]", count: (c !== null && isFinite(c)) ? c : null }};
        }}
        return {{ text: String(x), count: null }};
      }}
      function fmtTopSymbol(x) {{
        if (x === null || x === undefined) return {{ text: "", count: null }};
        if (typeof x === "string") return {{ text: x, count: null }};
        if (typeof x === "object") {{
          const p = x.symbol ? String(x.symbol) : "";
          const c = (x.count === 0 || x.count) ? Number(x.count) : null;
          return {{ text: p || "[object]", count: (c !== null && isFinite(c)) ? c : null }};
        }}
        return {{ text: String(x), count: null }};
      }}

      // Neighbor list by visible links.
      const nbrs = [];
      const nbrsOut = [];
      const nbrsIn = [];
      for (const l of links) {{
        if (l._hide) continue;
        if (l.a === node.id) {{
          nbrs.push([l.b, l.wsum, l.edge_count]);
          nbrsOut.push([l.b, l.wsum, l.edge_count]);
        }} else if (l.b === node.id) {{
          nbrs.push([l.a, l.wsum, l.edge_count]);
          nbrsIn.push([l.a, l.wsum, l.edge_count]);
        }}
      }}
      const nbrSort = (x, y) => (y[1] - x[1]) || (y[2] - x[2]) || (x[0] - y[0]);
      nbrs.sort(nbrSort);
      nbrsOut.sort(nbrSort);
      nbrsIn.sort(nbrSort);

      let html = "";
	      html += `<div class="pill"><b>Selected</b> <span>${{title}}</span></div>`;
      html += `<div class="pill"><b>size</b> <span>${{Number(s.size||0)}}</span></div>`;
		      if (groupsMode !== "off") {{
		        const gid = Number(node._group || 0);
		        const gs = gid ? Number(groupSizeById.get(gid) || 0) : 0;
		        const gl = gid ? (groupLabelById.get(gid) || null) : null;
		        html += `<div class="pill"><b>group</b> <span>${{gid || "n/a"}}</span></div>`;
		        html += `<div class="pill"><b>group_sz</b> <span>${{gs || "n/a"}}</span></div>`;
		        if (gl && gl.label) {{
		          const gtxt = String(gl.label || "").trim();
		          if (gtxt) html += `<div class="pill"><b>group_label</b> <span>${{gtxt}}</span></div>`;
		          const gconfRaw = gl.confidence;
		          if (gconfRaw === 0 || gconfRaw) {{
		            const cn = Number(gconfRaw);
		            if (isFinite(cn)) html += `<div class="pill"><b>group_conf</b> <span>${{fmtNum(cn)}}</span></div>`;
		          }}
		        }}
		      }}
	      html += `<div class="pill"><b>degree</b> <span>${{Number(mx.degree||0)}}</span></div>`;
	      if (isDirected) {{
	        html += `<div class="pill"><b>out_deg</b> <span>${{Number(mx.out_degree||0)}}</span></div>`;
	        html += `<div class="pill"><b>in_deg</b> <span>${{Number(mx.in_degree||0)}}</span></div>`;
	      }}
	      html += `<div class="pill"><b>wdeg</b> <span>${{fmtNum(node._wdeg)}}</span></div>`;
	      html += `<div class="pill"><b>k_core</b> <span>${{Number(node._kcore||0)}}</span></div>`;
	      html += `<div class="pill"><b>ext_w</b> <span>${{fmtNum(mx.external_weight_sum)}}</span></div>`;
	      if (isDirected) {{
	        html += `<div class="pill"><b>ext_w_out</b> <span>${{fmtNum(mx.external_weight_sum_out)}}</span></div>`;
	        html += `<div class="pill"><b>ext_w_in</b> <span>${{fmtNum(mx.external_weight_sum_in)}}</span></div>`;
	      }}
	      html += `<div class="pill"><b>int_w</b> <span>${{fmtNum(mx.internal_weight_sum)}}</span></div>`;
	      html += `<div class="pill"><b>ext/int</b> <span>${{fmtNum(mx.external_internal_ratio)}}</span></div>`;

      html += `<div class="section"><h3>Summary</h3>`;
      html += `<div class="kv">`;
      html += `<div class="k">cluster_id</div><div class="v">${{Number(s.cluster_id)}}</div>`;
      html += `<div class="k">label</div><div class="v">${{String(s.label||"")}}</div>`;
      html += `<div class="k">size</div><div class="v">${{Number(s.size||0)}}</div>`;
      html += `</div></div>`;

      html += `<div class="section"><h3>Top Files</h3>`;
      if (!tf.length) html += `<div class="hint">n/a</div>`;
      else html += `<div class="list">` + tf.map(x => {{
        const t = fmtTopFile(x);
        const right = (t.count !== null) ? String(t.count) : "";
        return `<div class="row"><div class="top"><div class="id">${{t.text}}</div><div>${{right}}</div></div></div>`;
      }}).join("") + `</div>`;
      html += `</div>`;

      html += `<div class="section"><h3>Top Symbols</h3>`;
      if (!ts.length) html += `<div class="hint">n/a</div>`;
      else html += `<div class="list">` + ts.map(x => {{
        const t = fmtTopSymbol(x);
        const right = (t.count !== null) ? String(t.count) : "";
        return `<div class="row"><div class="top"><div class="id">${{t.text}}</div><div>${{right}}</div></div></div>`;
      }}).join("") + `</div>`;
      html += `</div>`;

      if (!isDirected) {{
        html += `<div class="section"><h3>Neighbors (by weight_sum)</h3>`;
        if (!nbrs.length) {{
          html += `<div class="hint">No visible neighbors (adjust edge threshold?).</div>`;
        }} else {{
          html += `<div class="list">` + nbrs.slice(0, 30).map(([id, w, ec]) => {{
            const ns = sysById.get(id) || {{cluster_id: id, label: "", size: 0}};
            const tt = systemTitle(ns);
            return `<div class="row" data-node-id="${{id}}"><div class="top"><div class="id">${{tt}}</div><div>${{fmtNum(w)}}</div></div><div class="sub">edges=${{ec}}</div></div>`;
          }}).join("") + `</div>`;
        }}
        html += `</div>`;
      }} else {{
        html += `<div class="section"><h3>Outgoing (by weight_sum)</h3>`;
        if (!nbrsOut.length) {{
          html += `<div class="hint">No visible outgoing links (adjust edge threshold?).</div>`;
        }} else {{
          html += `<div class="list">` + nbrsOut.slice(0, 30).map(([id, w, ec]) => {{
            const ns = sysById.get(id) || {{cluster_id: id, label: "", size: 0}};
            const tt = systemTitle(ns);
            return `<div class="row" data-node-id="${{id}}"><div class="top"><div class="id">${{tt}}</div><div>${{fmtNum(w)}}</div></div><div class="sub">arcs=${{ec}}</div></div>`;
          }}).join("") + `</div>`;
        }}
        html += `</div>`;

        html += `<div class="section"><h3>Incoming (by weight_sum)</h3>`;
        if (!nbrsIn.length) {{
          html += `<div class="hint">No visible incoming links (adjust edge threshold?).</div>`;
        }} else {{
          html += `<div class="list">` + nbrsIn.slice(0, 30).map(([id, w, ec]) => {{
            const ns = sysById.get(id) || {{cluster_id: id, label: "", size: 0}};
            const tt = systemTitle(ns);
            return `<div class="row" data-node-id="${{id}}"><div class="top"><div class="id">${{tt}}</div><div>${{fmtNum(w)}}</div></div><div class="sub">arcs=${{ec}}</div></div>`;
          }}).join("") + `</div>`;
        }}
        html += `</div>`;
      }}

      setPanelHtml(html);
    }}

    function renderLinkPanel(link) {{
      const a = sysById.get(link.a) || {{cluster_id: link.a, label: "", size: 0}};
      const b = sysById.get(link.b) || {{cluster_id: link.b, label: "", size: 0}};
      const arrow = link.directed ? "→" : "⇄";
      const title = `${{systemTitle(a)}}  <span style="color:var(--muted)">${{arrow}}</span>  ${{systemTitle(b)}}`;

      let html = "";
      html += `<div class="pill"><b>Selected link</b> <span>${{title}}</span></div>`;
      html += `<div class="pill"><b>weight_sum</b> <span>${{fmtNum(link.wsum)}}</span></div>`;
      html += `<div class="pill"><b>weight_max</b> <span>${{fmtNum(link.wmax)}}</span></div>`;
      html += `<div class="pill"><b>edge_count</b> <span>${{Number(link.edge_count||0)}}</span></div>`;
      html += `<div class="section"><h3>Evidence</h3>`;
      if (!link.evidence || !link.evidence.length) {{
        html += `<div class="hint">n/a</div>`;
      }} else {{
        html += `<div class="list">` + link.evidence.map(ev => {{
          const kind = String(ev.kind || "");
          const w = fmtNum(ev.w);
          const leftPath = String(ev.a_path || ev.source_path || "");
          const rightPath = String(ev.b_path || ev.target_path || "");
          const leftStart = Number(ev.a_start_line || ev.source_start_line || 0);
          const leftEnd = Number(ev.a_end_line || ev.source_end_line || 0);
          const rightStart = Number(ev.b_start_line || ev.target_start_line || 0);
          const rightEnd = Number(ev.b_end_line || ev.target_end_line || 0);
          const left = `${{leftPath}}:${{leftStart}}-${{leftEnd}}`;
          const right = `${{rightPath}}:${{rightStart}}-${{rightEnd}}`;
          const asSym = ev.a_symbol || ev.source_symbol;
          const bsSym = ev.b_symbol || ev.target_symbol;
          const as = asSym ? (" (" + String(asSym) + ")") : "";
          const bs = bsSym ? (" (" + String(bsSym) + ")") : "";
          return `<div class="row"><div class="top"><div class="id">${{kind}}</div><div>${{w}}</div></div><div class="sub">${{left}}${{as}}<br/>${{right}}${{bs}}</div></div>`;
        }}).join("") + `</div>`;
      }}
      html += `</div>`;
      setPanelHtml(html);
    }}

    function worldToScreen(wx, wy) {{
      const r = card.getBoundingClientRect();
      const cx = r.width / 2;
      const cy = r.height / 2;
      const ct = Math.cos(cam.theta);
      const st = Math.sin(cam.theta);
      const rx = wx * ct - wy * st;
      const ry = wx * st + wy * ct;
      return {{
        x: rx * cam.scale + cx + cam.panX,
        y: ry * cam.scale + cy + cam.panY,
      }};
    }}

    function screenToWorld(sx, sy) {{
      const r = card.getBoundingClientRect();
      const cx = r.width / 2;
      const cy = r.height / 2;
      const x = (sx - cx - cam.panX) / cam.scale;
      const y = (sy - cy - cam.panY) / cam.scale;
      const ct = Math.cos(cam.theta);
      const st = Math.sin(cam.theta);
      // Inverse rotation.
      return {{
        x: x * ct + y * st,
        y: -x * st + y * ct,
      }};
    }}

    function nodeRadiusWorld(n) {{
      // Size encodes system (cluster) size. Use a sqrt scale (perceptual) with
      // a wider cap so differences are visually obvious for operators.
      const s = Math.max(1, Number(n.size || 1));
      return clamp(4 + Math.sqrt(s) * 2.2, 6, 46);
    }}

    function updateReadout() {{
      if (!camReadoutEl) return;
      const z = Math.round(cam.scale * 100);
      const deg = Math.round((cam.theta * 180) / Math.PI);
      camReadoutEl.textContent = `zoom=${{z}}% rot=${{deg}}°`;
    }}

	    function computeFitTarget() {{
      const r = card.getBoundingClientRect();
      const w = r.width;
      const h = r.height;
      const pad = 40;
      const ct = Math.cos(cam.theta);
      const st = Math.sin(cam.theta);

      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      let any = false;
      for (const n of nodes) {{
        if (!visibleNode(n)) continue;
        const rr = nodeRadiusWorld(n);
        const rx = n.x * ct - n.y * st;
        const ry = n.x * st + n.y * ct;
        minX = Math.min(minX, rx - rr);
        maxX = Math.max(maxX, rx + rr);
        minY = Math.min(minY, ry - rr);
        maxY = Math.max(maxY, ry + rr);
        any = true;
      }}
      if (!any) return null;

      const bboxW = Math.max(1e-6, maxX - minX);
      const bboxH = Math.max(1e-6, maxY - minY);
      const sx = (w - 2 * pad) / bboxW;
      const sy = (h - 2 * pad) / bboxH;
	      // Allow deeper zoom-out for very large graphs (operators should not need browser zoom).
	      const scale = clamp(Math.min(sx, sy), 0.005, 8.0);

      const cxr = (minX + maxX) / 2;
      const cyr = (minY + maxY) / 2;
      return {{
        scale,
        panX: -cxr * scale,
        panY: -cyr * scale,
      }};
    }}

	    function fitToView() {{
	      const t = computeFitTarget();
	      if (!t) return;
	      cam.scale = t.scale;
      cam.panX = t.panX;
      cam.panY = t.panY;
      updateReadout();
    }}

    function selectNode(id) {{
      selectedLink = null;
      selectedNode = nodeById.get(Number(id)) || null;
      if (selectedNode) renderSystemPanel(selectedNode);
      refreshSelectedNbrWeights();
      scheduleDraw();
    }}
	    function selectLink(a, b) {{
      selectedNode = null;
      selectedLink = null;
      refreshSelectedNbrWeights();
      const aa = Number(a), bb = Number(b);
      for (const l of links) {{
        if (isDirected) {{
          if (l.a === aa && l.b === bb) {{ selectedLink = l; break; }}
        }} else {{
          if ((l.a === aa && l.b === bb) || (l.a === bb && l.b === aa)) {{
            selectedLink = l;
            break;
          }}
        }}
      }}
      if (selectedLink) renderLinkPanel(selectedLink);
      scheduleDraw();
    }}

	    const searchEl = document.getElementById("search");
    function applySearch() {{
      const q = String(searchEl.value || "").trim().toLowerCase();
      for (const n of nodes) {{
        if (!q) {{ n._match = true; continue; }}
        const s = sysById.get(n.id) || {{}};
        const tf = (s.top_files || []).map(x => {{
          if (typeof x === "string") return x;
          if (x && typeof x === "object" && x.path) return String(x.path);
          return "";
        }});
        const ts = (s.top_symbols || []).map(x => {{
          if (typeof x === "string") return x;
          if (x && typeof x === "object" && x.symbol) return String(x.symbol);
          return "";
        }});
        const hay = [
          String(s.label || ""),
          String(s.label_llm || ""),
          String(s.label_heuristic || ""),
          ...tf,
          ...ts,
        ].join(" ").toLowerCase();
        n._match = hay.includes(q) || String(n.id).includes(q);
      }}
      scheduleDraw();
    }}
    searchEl.addEventListener("input", applySearch);

	    const thrEl = document.getElementById("thr");
    function applyThreshold() {{
      const t = Number(thrEl.value || 0) / 100;
      const cut = lerp(wMin, wMax, t);
      for (const l of links) {{
        l._hide = l.wsum < cut;
      }}
      if (selectedNode) renderSystemPanel(selectedNode);
      if (selectedLink) renderLinkPanel(selectedLink);
      refreshSelectedNbrWeights();
      scheduleDraw();
    }}
    thrEl.addEventListener("input", applyThreshold);

	    nbrOnlyEl.addEventListener("change", () => {{
	      scheduleDraw();
	    }});
	    if (showLabelsEl) {{
	      showLabelsEl.addEventListener("change", () => {{
	        scheduleDraw();
	      }});
	    }}

	    function initGroupingUi() {{
	      if (!groupsSel || !grpEl) return;
	      const have = !!groupPartitions.length;
	      groupsSel.disabled = !have;
	      grpEl.disabled = true;
	      grpEl.min = "0";
	      grpEl.max = String(Math.max(0, groupPartitions.length - 1));
	      grpEl.value = "0";
	      if (grpReadoutEl) grpReadoutEl.textContent = "";

	      function applyGroupsMode() {{
	        const v = groupsSel.value || "off";
	        setGroupsMode(v);
	        const enableSlider = (groupsMode !== "off") && have;
	        grpEl.disabled = !enableSlider;
	      }}

	      groupsSel.addEventListener("change", () => {{
	        applyGroupsMode();
	      }});
	      grpEl.addEventListener("input", () => {{
	        setGroupPartitionIndex(Number(grpEl.value || 0));
	      }});

	      applyGroupsMode();
	      setGroupPartitionIndex(0);
	    }}

	    initGroupingUi();

	    function computeKCoreNumbers() {{
	      // Linear-time k-core decomposition (Matula-Beck). Unweighted.
	      // We compute on the undirected simple graph induced by system adjacency.
	      const ids = nodes.map(n => n.id).slice().sort((a, b) => a - b);
	      const idxById = new Map(ids.map((id, i) => [id, i]));
	      const adj = new Array(ids.length);
	      for (let i = 0; i < ids.length; i++) adj[i] = [];
	      for (const l of links) {{
	        const ia = idxById.get(l.a);
	        const ib = idxById.get(l.b);
	        if (ia === undefined || ib === undefined || ia === ib) continue;
	        adj[ia].push(ib);
	        adj[ib].push(ia);
	      }}
	      // Dedupe neighbors for safety.
	      for (let i = 0; i < adj.length; i++) {{
	        const s = new Set(adj[i]);
	        adj[i] = Array.from(s);
	      }}

	      const n = adj.length;
	      const deg = new Array(n);
	      let maxDeg = 0;
	      for (let i = 0; i < n; i++) {{
	        const d = adj[i].length;
	        deg[i] = d;
	        if (d > maxDeg) maxDeg = d;
	      }}
	      const bin = new Array(maxDeg + 1).fill(0);
	      for (let i = 0; i < n; i++) bin[deg[i]] += 1;
	      let start = 0;
	      for (let d = 0; d <= maxDeg; d++) {{
	        const num = bin[d];
	        bin[d] = start;
	        start += num;
	      }}
	      const vert = new Array(n);
	      const pos = new Array(n);
	      for (let v = 0; v < n; v++) {{
	        pos[v] = bin[deg[v]];
	        vert[pos[v]] = v;
	        bin[deg[v]] += 1;
	      }}
	      for (let d = maxDeg; d >= 1; d--) bin[d] = bin[d - 1];
	      bin[0] = 0;

	      for (let i = 0; i < n; i++) {{
	        const v = vert[i];
	        for (const u of adj[v]) {{
	          if (deg[u] > deg[v]) {{
	            const du = deg[u];
	            const pu = pos[u];
	            const pw = bin[du];
	            const w = vert[pw];
	            if (u !== w) {{
	              vert[pu] = w; pos[w] = pu;
	              vert[pw] = u; pos[u] = pw;
	            }}
	            bin[du] += 1;
	            deg[u] -= 1;
	          }}
	        }}
	      }}
	      const coreById = new Map();
	      for (let i = 0; i < ids.length; i++) coreById.set(ids[i], deg[i]);
	      for (const node of nodes) node._kcore = coreById.get(node.id) || 0;
	    }}

	    function centerSlotOrder(k) {{
	      // Returns slot indices in "center-out" order: mid, mid-1, mid+1, ...
	      if (k <= 0) return [];
	      const midL = Math.floor((k - 1) / 2);
	      const out = [midL];
	      for (let step = 1; out.length < k; step++) {{
	        const a = midL - step;
	        const b = midL + step;
	        if (a >= 0) out.push(a);
	        if (out.length >= k) break;
	        if (b < k) out.push(b);
	      }}
	      return out;
	    }}

	    function computeKCoreLayoutPositions() {{
	      if (!nodes.length) return new Map();
	      // Ensure k-core numbers exist.
	      computeKCoreNumbers();

	      const layersByK = new Map(); // k -> node[]
	      for (const n of nodes) {{
	        const k = Number(n._kcore || 0);
	        let arr = layersByK.get(k);
	        if (!arr) {{ arr = []; layersByK.set(k, arr); }}
	        arr.push(n);
	      }}
	      const ks = Array.from(layersByK.keys()).sort((a, b) => b - a);

	      // Deterministic initial ordering by importance.
	      for (const k of ks) {{
	        const arr = layersByK.get(k) || [];
	        arr.sort((a, b) => (b._wdeg - a._wdeg) || (b.size - a.size) || (a.id - b.id));
	      }}

	      const yGap = 240;
	      const xGap = 140;

	      // Assign provisional x/y positions.
	      const layerIndexById = new Map();
	      const pos = new Map(); // id -> position object
	      const L = ks.length;
	      for (let li = 0; li < ks.length; li++) {{
	        const k = ks[li];
	        const arr = layersByK.get(k) || [];
	        const y = (li - (L - 1) / 2) * yGap;
	        const slots = centerSlotOrder(arr.length);
	        // precompute slot->x
	        const slotX = new Array(arr.length);
	        for (let s = 0; s < arr.length; s++) {{
	          slotX[s] = (s - (arr.length - 1) / 2) * xGap;
	        }}
	        for (let rank = 0; rank < arr.length; rank++) {{
	          const n = arr[rank];
	          const slot = slots[rank] ?? rank;
	          const x = slotX[slot];
	          pos.set(n.id, {{ x, y }});
	          layerIndexById.set(n.id, li);
	        }}
	      }}

	      // Crossing reduction: barycentric reordering within layers.
	      // We use neighbors from ALL links, weighted by wsum.
	      function barycenterFor(nodeId, neighborLayerPredicate) {{
	        const arr = nbrs.get(nodeId) || [];
	        let num = 0;
	        let den = 0;
	        for (const e of arr) {{
	          const nbId = e.id;
	          const li = layerIndexById.get(nbId);
	          if (li === undefined) continue;
	          if (!neighborLayerPredicate(li)) continue;
	          const p = pos.get(nbId);
	          if (!p) continue;
	          const w = Number(e.w || 0);
	          num += p.x * w;
	          den += w;
	        }}
	        if (den <= 0) return null;
	        return num / den;
	      }}

	      function reorderLayer(li, useAbove) {{
	        const k = ks[li];
	        const arr = (layersByK.get(k) || []).slice();
	        // Current order key for stable tie-break.
	        const current = new Map();
	        for (let i = 0; i < arr.length; i++) current.set(arr[i].id, i);
	        const pred = (nli) => useAbove ? (nli < li) : (nli > li);
	        const keyed = arr.map(n => {{
	          const bc = barycenterFor(n.id, pred);
	          return {{
	            n,
	            bc: (bc === null) ? Infinity : bc,
	            has: (bc !== null),
	            ord: current.get(n.id) || 0,
	          }};
	        }});
	        keyed.sort((a, b) => {{
	          // Put nodes with barycenter first; keep isolates in their relative order.
	          if (a.has !== b.has) return a.has ? -1 : 1;
	          if (a.bc !== b.bc) return a.bc < b.bc ? -1 : 1;
	          if (a.ord !== b.ord) return a.ord - b.ord;
	          return a.n.id - b.n.id;
	        }});
	        const ordered = keyed.map(x => x.n);
	        // Reassign evenly spaced x positions in this new order.
	        const slotX = new Array(ordered.length);
	        for (let s = 0; s < ordered.length; s++) {{
	          slotX[s] = (s - (ordered.length - 1) / 2) * xGap;
	        }}
	        for (let i = 0; i < ordered.length; i++) {{
	          const n = ordered[i];
	          const p = pos.get(n.id);
	          if (!p) continue;
	          p.x = slotX[i];
	        }}
	        // Persist new order into layersByK for subsequent sweeps.
	        layersByK.set(k, ordered);
	      }}

	      const sweeps = 2;
	      for (let s = 0; s < sweeps; s++) {{
	        for (let li = 0; li < ks.length; li++) reorderLayer(li, true);
	        for (let li = ks.length - 1; li >= 0; li--) reorderLayer(li, false);
	      }}

	      // Return final positions map.
	      const out = new Map();
	      for (const n of nodes) {{
	        const p = pos.get(n.id);
	        if (p) out.set(n.id, {{ x: p.x, y: p.y }});
	      }}
	      return out;
	    }}

	    function setLayout(mode) {{
	      const m = String(mode || "force");
	      activeLayout = m;
	      if (m === "kcore") {{
	        if (!layoutPositions.kcore) {{
	          layoutPositions.kcore = computeKCoreLayoutPositions();
	        }}
	        restorePositions(layoutPositions.kcore);
	        refreshSelectedNbrWeights();
	      }} else if (m === "hub") {{
	        if (!layoutPositions.hub) {{
	          const res = computeHubRadialLayoutPositions();
	          layoutPositions.hub = res.positions;
	          hubLayoutMeta = res.meta;
	        }}
	        restorePositions(layoutPositions.hub);
	        refreshSelectedNbrWeights();
	      }} else if (m === "hub_grav") {{
	        if (!layoutPositions.hub_grav) {{
	          layoutPositions.hub_grav = computeHubGravityLayoutPositions();
	        }}
	        restorePositions(layoutPositions.hub_grav);
	        // Link selection is confusing when links are not drawn.
	        const hadLinkSel = !!selectedLink;
	        selectedLink = null;
	        if (hadLinkSel) {{
	          setPanelHtml(`<div class="hint">Click a node to see details. Neighbor relationships are implied by proximity.</div>`);
	        }}
	        refreshSelectedNbrWeights();
	      }} else {{
	        // Default to force.
	        restorePositions(layoutPositions.force);
	        refreshSelectedNbrWeights();
	      }}
	      if (layoutSel && layoutSel.value !== m) layoutSel.value = m;
	      fitToView();
	      scheduleDraw();
	    }}

		    if (layoutSel) {{
		      layoutSel.addEventListener("change", () => {{
		        setLayout(layoutSel.value);
		      }});
		    }}

		    function computeHubRadialLayoutPositions() {{
		      // Hub-centric onion/radial layout:
		      // - rank nodes by external weighted degree (node._wdeg)
		      // - place strongest hubs near the center
		      // - assign remaining nodes to rings with capacity based on circumference
		      // - refine angular ordering by neighbor barycenter
		      const positions = new Map();
		      const meta = {{ radii: [] }};
		      if (!nodes.length) return {{ positions, meta }};

		      const ordered = nodes.slice().sort((a, b) => (b._wdeg - a._wdeg) || (b.size - a.size) || (a.id - b.id));

		      // Spacing derived from max node radius (world units).
		      let maxNodeR = 0;
		      for (const n of nodes) maxNodeR = Math.max(maxNodeR, nodeRadiusWorld(n));
		      const minSpacing = clamp((2 * maxNodeR) + 40, 120, 220);
		      const ringGap = Math.max(220, 1.9 * minSpacing);
		      meta.min_spacing = minSpacing;
		      meta.ring_gap = ringGap;

		      const coreCount = 1;
		      const center = ordered[0];
		      positions.set(center.id, {{ x: 0, y: 0 }});
		      meta.center_id = center.id;

		      // Ring assignment: capacity-based fill.
		      const rings = [];
		      let idx = coreCount;
		      for (let ringIdx = 1; idx < ordered.length; ringIdx++) {{
		        const r = ringIdx * ringGap;
		        const cap = Math.max(1, Math.floor((2 * Math.PI * r) / minSpacing));
		        const slice = ordered.slice(idx, idx + cap);
		        idx += slice.length;
		        const base = rand01(hash32(ringIdx)) * Math.PI * 2;
		        rings.push({{ ringIdx, r, base, nodes: slice }});
		        meta.radii.push(r);
		      }}

		      const thetaById = new Map();
		      thetaById.set(center.id, 0);

		      function normTheta(t) {{
		        let x = t % (Math.PI * 2);
		        if (x < 0) x += Math.PI * 2;
		        return x;
		      }}

		      function assignEvenAngles(ring) {{
		        const arr = ring.nodes;
		        const m = arr.length;
		        if (!m) return;
		        for (let j = 0; j < m; j++) {{
		          const n = arr[j];
		          const theta = ring.base + (j / m) * Math.PI * 2;
		          thetaById.set(n.id, normTheta(theta));
		        }}
		      }}

		      // Initial even spacing per ring.
		      for (const ring of rings) assignEvenAngles(ring);

		      function barycenterTheta(nodeId) {{
		        const arr = nbrs.get(nodeId) || [];
		        let sx = 0;
		        let sy = 0;
		        let den = 0;
		        for (const e of arr) {{
		          const nbId = e.id;
		          const th = thetaById.get(nbId);
		          if (th === undefined) continue;
		          const w = Number(e.w || 0);
		          if (!(w > 0)) continue;
		          sx += Math.cos(th) * w;
		          sy += Math.sin(th) * w;
		          den += w;
		        }}
		        if (den <= 0) return null;
		        return normTheta(Math.atan2(sy, sx));
		      }}

		      // Angular refinement by neighbor barycenter (deterministic).
		      const angularSweeps = 3;
		      for (let s = 0; s < angularSweeps; s++) {{
		        for (const ring of rings) {{
		          const current = new Map();
		          for (let i = 0; i < ring.nodes.length; i++) current.set(ring.nodes[i].id, i);
		          const keyed = ring.nodes.map(n => {{
		            const bc = barycenterTheta(n.id);
		            return {{
		              n,
		              bc: (bc === null) ? Infinity : bc,
		              has: (bc !== null),
		              ord: current.get(n.id) || 0,
		            }};
		          }});
		          keyed.sort((a, b) => {{
		            if (a.has !== b.has) return a.has ? -1 : 1;
		            if (a.bc !== b.bc) return a.bc < b.bc ? -1 : 1;
		            if (a.ord !== b.ord) return a.ord - b.ord;
		            return a.n.id - b.n.id;
		          }});
		          ring.nodes = keyed.map(x => x.n);
		          assignEvenAngles(ring);
		        }}
		      }}

		      // Convert polar to Cartesian.
		      for (const ring of rings) {{
		        for (const n of ring.nodes) {{
		          const th = thetaById.get(n.id) || 0;
		          positions.set(n.id, {{ x: ring.r * Math.cos(th), y: ring.r * Math.sin(th) }});
		        }}
		      }}

		      return {{ positions, meta }};
		    }}

		    function computeHubGravityLayoutPositions() {{
		      // Edge-less hub-centric layout computed by time-bounded physics relaxation.
		      // Seed is a hub-ranked scatter with mild randomness so the final arrangement is
		      // less rigid and less ring-like than the hub radial layout.
		      const seed0 = (Date.now() ^ (Math.floor(Math.random() * 0x7fffffff) | 0)) | 0;
		      function rand01s(key) {{ return rand01((seed0 ^ (key | 0)) | 0); }}

		      const out = new Map();
		      if (!nodes.length) return out;

		      // Stable node order for simulation loops.
		      const simNodes = nodes.slice().sort((a, b) => a.id - b.id);
		      const idxById = new Map();
		      for (let i = 0; i < simNodes.length; i++) idxById.set(simNodes[i].id, i);

		      const x = new Array(simNodes.length);
		      const y = new Array(simNodes.length);
		      const vx = new Array(simNodes.length).fill(0);
		      const vy = new Array(simNodes.length).fill(0);
		      const mass = new Array(simNodes.length);
		      const radius = new Array(simNodes.length);

		      let wdegMin = Infinity, wdegMax = -Infinity;
		      for (const n of simNodes) {{
		        const w = Number(n._wdeg || 0);
		        wdegMin = Math.min(wdegMin, w);
		        wdegMax = Math.max(wdegMax, w);
		      }}
		      if (!isFinite(wdegMin)) {{ wdegMin = 0; wdegMax = 0; }}
		      const wdegSpan = Math.max(1e-9, (wdegMax - wdegMin));

		      const nn = Math.max(1, simNodes.length);
		      const initRadius = 120 + Math.sqrt(nn) * 75;
		      const gamma = 1.7;
		      for (let i = 0; i < simNodes.length; i++) {{
		        const n = simNodes[i];
		        const hubT = clamp((Number(n._wdeg || 0) - wdegMin) / wdegSpan, 0, 1);
		        let r = initRadius * Math.pow(1 - hubT, gamma);
		        const jr = lerp(0.75, 1.25, rand01s(n.id + 11));
		        r = r * jr;
		        const ang = rand01s(n.id + 97) * Math.PI * 2;
		        const radialJ = (rand01s(n.id + 151) - 0.5) * initRadius * 0.18;
		        r = Math.max(0, r + radialJ);
		        x[i] = Math.cos(ang) * r + (rand01s(n.id + 251) - 0.5) * initRadius * 0.18;
		        y[i] = Math.sin(ang) * r + (rand01s(n.id + 381) - 0.5) * initRadius * 0.18;
		        const s = Math.max(1, Number(n.size || 1));
		        mass[i] = 1 + Math.sqrt(s);
		        radius[i] = nodeRadiusWorld(n);
		      }}

		      const repulseK = 5600 / Math.sqrt(nn);
		      const collideK = 0.60;
		      const springK = 0.0035;
		      const centerPull = 0.0020 / Math.sqrt(nn);
		      const damping = 0.86;
		      const dt = 0.016;
		      const pad = 10;
		      const maxSpeed = 1400;

		      const wSpan = Math.max(1e-9, (wMax - wMin));
		      const edgeBase = 26;

		      const fx = new Array(simNodes.length);
		      const fy = new Array(simNodes.length);

		      const maxSteps = 650;
		      const minSteps = 90;
		      const maxMs = 320;
		      const start = (typeof performance !== "undefined" ? performance.now() : Date.now());

		      function clampSpeed(i) {{
		        const sx = vx[i], sy = vy[i];
		        const sp = Math.hypot(sx, sy);
		        if (sp > maxSpeed) {{
		          const k = maxSpeed / Math.max(1e-9, sp);
		          vx[i] = sx * k;
		          vy[i] = sy * k;
		        }}
		      }}

		      for (let step = 0; step < maxSteps; step++) {{
		        const now = (typeof performance !== "undefined" ? performance.now() : Date.now());
		        if (step >= minSteps && (now - start) > maxMs) break;

		        for (let i = 0; i < nn; i++) {{ fx[i] = 0; fy[i] = 0; }}

		        // Repulsion + collision (all pairs).
		        for (let i = 0; i < nn; i++) {{
		          for (let j = i + 1; j < nn; j++) {{
		            const dx = x[i] - x[j];
		            const dy = y[i] - y[j];
		            const d2 = dx*dx + dy*dy + 0.05;
		            const d = Math.sqrt(d2);

		            // Base repulsion.
		            const fr = repulseK / d2;
		            const rx = dx * fr;
		            const ry = dy * fr;
		            fx[i] += rx; fy[i] += ry;
		            fx[j] -= rx; fy[j] -= ry;

		            // Collision separation.
		            const minD = radius[i] + radius[j] + pad;
		            if (d < minD) {{
		              const overlap = (minD - d);
		              const fc = (overlap / Math.max(1e-9, d)) * collideK;
		              const cx = dx * fc;
		              const cy = dy * fc;
		              fx[i] += cx; fy[i] += cy;
		              fx[j] -= cx; fy[j] -= cy;
		            }}
		          }}
		        }}

		        // Attraction along ALL edges (not threshold-filtered).
		        for (const l of links) {{
		          const ia = idxById.get(l.a);
		          const ib = idxById.get(l.b);
		          if (ia === undefined || ib === undefined || ia === ib) continue;
		          const dx = x[ib] - x[ia];
		          const dy = y[ib] - y[ia];
		          const dist = Math.sqrt(dx*dx + dy*dy) + 1e-6;
		          const t = clamp((l.wsum - wMin) / wSpan, 0, 1);
		          // Weight influences attraction strength, not a target distance (less "idealized").
		          const target = (radius[ia] + radius[ib]) + edgeBase;
		          const k = springK * lerp(0.20, 1.00, Math.pow(t, 0.65));
		          const f = (dist - target) * k;
		          const ux = dx / dist;
		          const uy = dy / dist;
		          fx[ia] += ux * f;
		          fy[ia] += uy * f;
		          fx[ib] -= ux * f;
		          fy[ib] -= uy * f;
		        }}

		        // Centering force + integrate with size-based inertia.
		        let ke = 0;
		        for (let i = 0; i < nn; i++) {{
		          fx[i] += (0 - x[i]) * centerPull;
		          fy[i] += (0 - y[i]) * centerPull;

		          vx[i] = (vx[i] + (fx[i] / mass[i])) * damping;
		          vy[i] = (vy[i] + (fy[i] / mass[i])) * damping;
		          clampSpeed(i);
		          x[i] += vx[i] * dt;
		          y[i] += vy[i] * dt;
		          ke += (vx[i] * vx[i]) + (vy[i] * vy[i]);
		        }}

		        // Early stop: low kinetic energy after minimum settling.
		        if (step >= minSteps && ke < 0.08) break;
		      }}

		      for (let i = 0; i < nn; i++) {{
		        const id = simNodes[i].id;
		        out.set(id, {{ x: x[i], y: y[i] }});
		      }}
		      return out;
		    }}

	    // Basic force simulation (deterministic initial seed, then numeric integration).
		    function stepSim() {{
	      // Scale repulsion down as graph size grows; otherwise large graphs "explode" and require extreme zoom.
	      const nn = Math.max(1, nodes.length);
	      const repulse = 5200 / Math.sqrt(nn);
	      const spring = 0.0014;
	      const damping = 0.86;
	      const centerPull = 0.002;

      // Charge (repulsion).
      for (let i = 0; i < nodes.length; i++) {{
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {{
          const b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d2 = dx*dx + dy*dy + 0.01;
          const f = repulse / d2;
          const fx = dx * f, fy = dy * f;
          a.vx += fx; a.vy += fy;
          b.vx -= fx; b.vy -= fy;
        }}
      }}

      // Springs along visible links.
      for (const l of links) {{
        if (l._hide) continue;
        const a = nodeById.get(l.a);
        const b = nodeById.get(l.b);
        if (!a || !b) continue;
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.sqrt(dx*dx + dy*dy) + 0.001;
        const target = 90 + 140 * (1 - clamp((l.wsum - wMin) / Math.max(1e-9, (wMax - wMin)), 0, 1));
        const f = (dist - target) * spring;
        const fx = (dx / dist) * f;
        const fy = (dy / dist) * f;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }}

      for (const n of nodes) {{
        if (n.fx !== null && n.fy !== null) {{
          n.x = n.fx; n.y = n.fy;
          n.vx = 0; n.vy = 0;
          continue;
        }}
        // Gentle centering.
        n.vx += (0 - n.x) * centerPull;
        n.vy += (0 - n.y) * centerPull;
        n.vx *= damping;
        n.vy *= damping;
        n.x += n.vx * 0.016;
        n.y += n.vy * 0.016;
      }}

      let s2 = 0;
      for (const n of nodes) {{
        s2 += (n.vx * n.vx) + (n.vy * n.vy);
      }}
      return s2;
    }}

    function visibleNode(n) {{
      if (!n._match) return false;
      if (!nbrOnlyEl.checked) return true;
      if (!selectedNode) return true;
      if (n.id === selectedNode.id) return true;
      for (const l of links) {{
        if (l._hide) continue;
        if ((l.a === n.id && l.b === selectedNode.id) || (l.b === n.id && l.a === selectedNode.id)) return true;
      }}
      return false;
    }}

    function visibleLink(l) {{
      if (l._hide) return false;
      const na = nodeById.get(l.a);
      const nb = nodeById.get(l.b);
      if (!na || !nb) return false;
      if (!visibleNode(na) || !visibleNode(nb)) return false;
      return true;
    }}

    function linkIsSelected(l) {{
      if (!selectedLink) return false;
      if (isDirected) return (selectedLink.a === l.a && selectedLink.b === l.b);
      return (
        (selectedLink.a === l.a && selectedLink.b === l.b) ||
        (selectedLink.a === l.b && selectedLink.b === l.a)
      );
    }}

    function linkEndpointsScreen(l) {{
      const a = nodeById.get(l.a);
      const b = nodeById.get(l.b);
      if (!a || !b) return null;
      const sa0 = worldToScreen(a.x, a.y);
      const sb0 = worldToScreen(b.x, b.y);
      if (!isDirected) return {{ sa: sa0, sb: sb0 }};
      const u = Math.min(l.a, l.b);
      const v = Math.max(l.a, l.b);
      const key = `${{u}}|${{v}}`;
      if (Number(pairCount.get(key) || 0) <= 1) return {{ sa: sa0, sb: sb0 }};

      const dx = sb0.x - sa0.x;
      const dy = sb0.y - sa0.y;
      const len = Math.hypot(dx, dy) || 1;
      const px = -dy / len;
      const py = dx / len;
      const sign = (l.a < l.b) ? 1 : -1;
      const off = 7;
      return {{
        sa: {{ x: sa0.x + px * off * sign, y: sa0.y + py * off * sign }},
        sb: {{ x: sb0.x + px * off * sign, y: sb0.y + py * off * sign }},
      }};
    }}

	    function draw() {{
	      const r = card.getBoundingClientRect();
	      ctx.clearRect(0, 0, r.width, r.height);

	      // Group circles (behind links).
	      if (groupsMode !== "off" && groupBySystemId.size) {{
	        const groups = new Map(); // gid -> Array of node refs
	        for (const n of nodes) {{
	          if (!visibleNode(n)) continue;
	          const gid = Number(n._group || 0);
	          if (!gid) continue;
	          let arr = groups.get(gid);
	          if (!arr) {{ arr = []; groups.set(gid, arr); }}
	          arr.push(n);
	        }}

	        for (const [gid, arr] of groups.entries()) {{
	          if (!arr || !arr.length) continue;
	          let cx = 0, cy = 0;
	          for (const n of arr) {{ cx += n.x; cy += n.y; }}
	          cx /= arr.length; cy /= arr.length;
	          let rmax = 0;
	          for (const n of arr) {{
	            const dx = n.x - cx;
	            const dy = n.y - cy;
	            rmax = Math.max(rmax, Math.hypot(dx, dy));
	          }}
	          // Pad by node radii and a little breathing room.
	          let pad = 0;
	          for (const n of arr) pad = Math.max(pad, nodeRadiusWorld(n));
	          const radW = rmax + pad + 10;

	          const c = worldToScreen(cx, cy);
	          const radS = Math.max(6, radW * cam.scale);
	          ctx.fillStyle = groupColor(gid, 0.10);
	          ctx.strokeStyle = groupColor(gid, 0.28);
	          ctx.lineWidth = 2;
	          ctx.beginPath();
	          ctx.arc(c.x, c.y, radS, 0, Math.PI * 2);
	          ctx.fill();
	          ctx.stroke();
	        }}
	      }}

	      // Links.
	      if (activeLayout !== "hub_grav") {{
	        for (const l of links) {{
	          if (!visibleLink(l)) continue;
	          const ep = linkEndpointsScreen(l);
	          if (!ep) continue;
          const sa = ep.sa;
          const sb = ep.sb;
          const t = clamp((l.wsum - wMin) / Math.max(1e-9, (wMax - wMin)), 0, 1);
          const alpha = lerp(0.12, 0.55, t);
          const width = lerp(1.0, 4.0, t);

          let stroke = `rgba(255,255,255,${{alpha}})`;
          if (linkIsSelected(l)) {{
            stroke = "rgba(142,227,255,0.92)";
          }} else if (selectedNode && (l.a === selectedNode.id || l.b === selectedNode.id)) {{
            stroke = "rgba(255,184,107,0.75)";
          }}

          ctx.strokeStyle = stroke;
          ctx.lineWidth = width;
          ctx.beginPath();
          ctx.moveTo(sa.x, sa.y);
          ctx.lineTo(sb.x, sb.y);
          ctx.stroke();

          if (isDirected) {{
            const dx = sb.x - sa.x;
            const dy = sb.y - sa.y;
            const len = Math.hypot(dx, dy);
            if (len && len > 1e-6) {{
              const ux = dx / len;
              const uy = dy / len;
              const nx = -uy;
              const ny = ux;
              const arrowLen = clamp(8 + width * 1.2, 8, 14);
              const arrowW = arrowLen * 0.60;
              const tipX = sb.x;
              const tipY = sb.y;
              const baseX = tipX - ux * arrowLen;
              const baseY = tipY - uy * arrowLen;
              ctx.fillStyle = stroke;
              ctx.beginPath();
              ctx.moveTo(tipX, tipY);
              ctx.lineTo(baseX + nx * arrowW * 0.5, baseY + ny * arrowW * 0.5);
              ctx.lineTo(baseX - nx * arrowW * 0.5, baseY - ny * arrowW * 0.5);
              ctx.closePath();
              ctx.fill();
            }}
          }}
        }}
	      }}

      // Nodes.
      for (const n of nodes) {{
        if (!visibleNode(n)) continue;
        const s = worldToScreen(n.x, n.y);
        const rr = nodeRadiusWorld(n) * cam.scale;
        const rrDraw = clamp(rr, 3, 40);
        const isSel = selectedNode && selectedNode.id === n.id;

        let fill = isSel ? "rgba(255,184,107,0.95)" : "rgba(255,255,255,0.88)";
        let ring = (!n._match) ? "rgba(255,255,255,0.20)" : "rgba(142,227,255,0.22)";
        if (groupsMode !== "off" && !isSel) {{
          const gid = Number(n._group || 0);
          if (gid) ring = groupColor(gid, 0.55);
        }}
        if (activeLayout === "hub_grav") {{
          const isNbr = (!!selectedNode && !isSel && selectedNbrWeights && selectedNbrWeights.has(n.id));
          if (selectedNode && !isSel && !isNbr) {{
            fill = "rgba(255,255,255,0.18)";
            ring = "rgba(255,255,255,0.08)";
          }} else if (isNbr && selectedNbrWeights) {{
            const w = Number(selectedNbrWeights.get(n.id) || 0);
            const t = clamp(w / Math.max(1e-9, selectedNbrMaxW), 0, 1);
            const a = lerp(0.28, 0.80, Math.sqrt(t));
            fill = `rgba(255,255,255,${{a}})`;
            ring = `rgba(142,227,255,${{lerp(0.35, 0.95, Math.sqrt(t))}})`;
          }}
        }}
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.arc(s.x, s.y, rrDraw, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = ring;
        ctx.lineWidth = 2;
        ctx.stroke();
      }}

      // Labels.
      const showLabels = (!showLabelsEl) || !!showLabelsEl.checked;
      if (showLabels) {{
        // If the graph is very zoomed out, labels are unreadable and just add noise.
        // Exception: keep selected/neighborhood labels when a selection exists.
        const allowAll = (cam.scale >= 0.12) || (nodes.length <= 48);
        const allowFocusOnly = !!selectedNode;
        if (allowAll || allowFocusOnly) {{
          const fs = clamp(10 + Math.log2(1 + cam.scale) * 4, 10, 16);
          ctx.font = `${{fs}}px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif`;
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";

          for (const n of nodes) {{
            if (!visibleNode(n)) continue;
            const isSel = selectedNode && selectedNode.id === n.id;
            const isNbr = (!!selectedNode && !isSel && selectedNbrWeights && selectedNbrWeights.has(n.id));
            if (!allowAll && !(isSel || isNbr)) continue;

            const s = worldToScreen(n.x, n.y);
            const rr = nodeRadiusWorld(n) * cam.scale;
            const rrDraw = clamp(rr, 3, 40);
            const text = (n.label_short && String(n.label_short).trim())
              ? `#${{n.id}} ${{n.label_short}}`
              : `#${{n.id}}`;

            let alpha = 0.55;
            if (activeLayout === "hub_grav" && selectedNode) {{
              if (isSel) alpha = 0.92;
              else if (isNbr) alpha = 0.80;
              else alpha = 0.12;
            }} else if (selectedNode) {{
              if (isSel) alpha = 0.92;
              else if (isNbr) alpha = 0.75;
              else alpha = 0.28;
            }}

            const tx = s.x + rrDraw + 6;
            const ty = s.y;
            ctx.lineWidth = 4;
            ctx.strokeStyle = `rgba(0,0,0,${{0.70 * alpha}})`;
            ctx.fillStyle = `rgba(255,255,255,${{alpha}})`;
            ctx.strokeText(text, tx, ty);
            ctx.fillText(text, tx, ty);
          }}
        }}
      }}
    }}

    // Hit testing.
    function findNodeAt(x, y) {{
      for (let i = nodes.length - 1; i >= 0; i--) {{
        const n = nodes[i];
        if (!visibleNode(n)) continue;
        const s = worldToScreen(n.x, n.y);
        const rr = Math.max(7, nodeRadiusWorld(n) * cam.scale);
        const dx = x - s.x, dy = y - s.y;
        if (dx*dx + dy*dy <= rr*rr) return n;
      }}
      return null;
    }}
    function distPointToSegment(px, py, ax, ay, bx, by) {{
      const vx = bx - ax, vy = by - ay;
      const wx = px - ax, wy = py - ay;
      const c1 = vx*wx + vy*wy;
      if (c1 <= 0) return Math.hypot(px - ax, py - ay);
      const c2 = vx*vx + vy*vy;
      if (c2 <= c1) return Math.hypot(px - bx, py - by);
      const t = c1 / c2;
      const hx = ax + t * vx, hy = ay + t * vy;
      return Math.hypot(px - hx, py - hy);
    }}
    function findLinkAt(x, y) {{
      let best = null;
      let bestD = 1e9;
      for (const l of links) {{
        if (!visibleLink(l)) continue;
        const ep = linkEndpointsScreen(l);
        if (!ep) continue;
        const sa = ep.sa;
        const sb = ep.sb;
        const d = distPointToSegment(x, y, sa.x, sa.y, sb.x, sb.y);
        if (d < 6 && d < bestD) {{
          bestD = d;
          best = l;
        }}
      }}
      return best;
    }}

    // Drag / pan / rotate.
    let dragging = null;
    let dragOffset = {{x: 0, y: 0}};
    let dragMode = null; // "node" | "pan" | "rot"
    let downX = 0, downY = 0;
    let startPanX = 0, startPanY = 0;
    let startTheta = 0;
    let rotateAnchorW = null;
    let suppressClick = false;
    cv.addEventListener("mousedown", (ev) => {{
      const rect = cv.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      downX = x; downY = y;
      suppressClick = false;
      const n = findNodeAt(x, y);
      if (n) {{
        dragging = n;
        dragMode = "node";
        const w = screenToWorld(x, y);
        dragOffset = {{x: n.x - w.x, y: n.y - w.y}};
        // Static mode: move the node directly in world coords.
        n.x = w.x + dragOffset.x;
        n.y = w.y + dragOffset.y;
        n.vx = 0; n.vy = 0;
        scheduleDraw();
        return;
      }}

      // Background gesture: pan, or rotate if Shift is held.
      userMovedCamera = true;
      startPanX = cam.panX;
      startPanY = cam.panY;
      startTheta = cam.theta;
      if (ev.shiftKey) {{
        dragMode = "rot";
        rotateAnchorW = screenToWorld(x, y);
      }} else {{
        dragMode = "pan";
      }}
    }});
    window.addEventListener("mousemove", (ev) => {{
      if (!dragMode) return;
      const rect = cv.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      const dx = x - downX;
      const dy = y - downY;
      if ((dx*dx + dy*dy) > 16) suppressClick = true;

      if (dragMode === "node" && dragging) {{
        const w = screenToWorld(x, y);
        dragging.x = w.x + dragOffset.x;
        dragging.y = w.y + dragOffset.y;
        dragging.vx = 0; dragging.vy = 0;
        scheduleDraw();
        return;
      }}
      if (dragMode === "pan") {{
        cam.panX = startPanX + dx;
        cam.panY = startPanY + dy;
        updateReadout();
        scheduleDraw();
        return;
      }}
      if (dragMode === "rot") {{
        const anchor = rotateAnchorW;
        if (!anchor) return;
        cam.theta = startTheta + dx * 0.005;
        // Keep the anchor under the original down point.
        const r2 = card.getBoundingClientRect();
        const cx = r2.width / 2;
        const cy = r2.height / 2;
        const ct = Math.cos(cam.theta);
        const st = Math.sin(cam.theta);
        const rx = anchor.x * ct - anchor.y * st;
        const ry = anchor.x * st + anchor.y * ct;
        cam.panX = downX - cx - rx * cam.scale;
        cam.panY = downY - cy - ry * cam.scale;
        updateReadout();
        scheduleDraw();
        return;
      }}
    }});
    window.addEventListener("mouseup", () => {{
      dragging = null;
      dragMode = null;
      rotateAnchorW = null;
    }});

	    cv.addEventListener("wheel", (ev) => {{
      // Zoom around cursor. Prevent page scroll.
      ev.preventDefault();
      const rect = cv.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const my = ev.clientY - rect.top;
      const anchorW = screenToWorld(mx, my);

      const factor = Math.exp(-ev.deltaY * 0.001);
	      const newScale = clamp(cam.scale * factor, 0.005, 8.0);
	      cam.scale = newScale;
	      userMovedCamera = true;

      // Adjust pan so anchor stays under the cursor.
      const r2 = card.getBoundingClientRect();
      const cx = r2.width / 2;
      const cy = r2.height / 2;
      const ct = Math.cos(cam.theta);
      const st = Math.sin(cam.theta);
      const rx = anchorW.x * ct - anchorW.y * st;
      const ry = anchorW.x * st + anchorW.y * ct;
      cam.panX = mx - cx - rx * cam.scale;
      cam.panY = my - cy - ry * cam.scale;
      updateReadout();
      scheduleDraw();
    }}, {{ passive: false }});

    document.getElementById("fitBtn").addEventListener("click", () => {{
      fitToView();
      scheduleDraw();
    }});
    document.getElementById("resetBtn").addEventListener("click", () => {{
      cam.theta = 0.0;
      cam.scale = 1.0;
      cam.panX = 0.0;
      cam.panY = 0.0;
      fitToView();
      scheduleDraw();
    }});
    window.addEventListener("keydown", (ev) => {{
      if (ev.key === "f" || ev.key === "F") {{
        fitToView();
        scheduleDraw();
      }}
      if (ev.key === "0") {{
        cam.theta = 0.0;
        cam.scale = 1.0;
        cam.panX = 0.0;
        cam.panY = 0.0;
        fitToView();
        scheduleDraw();
      }}
    }});

    cv.addEventListener("click", (ev) => {{
      if (suppressClick) return;
      const rect = cv.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      const n = findNodeAt(x, y);
      if (n) {{
        selectNode(n.id);
        return;
      }}
      if (activeLayout !== "hub_grav") {{
        const l = findLinkAt(x, y);
        if (l) {{
          selectLink(l.a, l.b);
          return;
        }}
      }}
      selectedNode = null;
      selectedLink = null;
      refreshSelectedNbrWeights();
      const hint = (activeLayout === "hub_grav")
        ? `<div class="hint">Click a node to see details. Neighbor relationships are implied by proximity.</div>`
        : `<div class="hint">Click a node to see details. Click a link to see evidence.</div>`;
      setPanelHtml(hint);
      scheduleDraw();
    }});

	    function runInitialLayout() {{
	      // One-time layout warmup: compute positions while the canvas is hidden to avoid
	      // visible reflow. After this, the graph is static unless the operator moves nodes.
	      cv.style.opacity = "0";
	      if (loadingEl) loadingEl.style.display = "flex";

	      // Time-bounded warmup. For large graphs, keep it short to avoid blocking.
	      const maxMs = 1800;
	      const minSteps = 30;
	      const maxSteps = 320;
	      let steps = 0;
	      const start = (typeof performance !== "undefined" ? performance.now() : Date.now());

	      function finalizeStatic() {{
	        // Freeze velocities.
	        for (const n of nodes) {{
	          n.vx = 0; n.vy = 0;
	        }}
	        // Snapshot the force layout for later toggling.
	        layoutPositions.force = snapshotPositions();
	        fitToView();
	        if (loadingEl) loadingEl.style.display = "none";
	        cv.style.opacity = "1";
	        scheduleDraw();
	      }}

	      function frame() {{
	        if (cancelInitialLayout) {{
	          finalizeStatic();
	          return;
	        }}
	        const frameStart = (typeof performance !== "undefined" ? performance.now() : Date.now());
	        // Budget about half a frame worth of work.
	        let energy = 0;
	        while (true) {{
	          const now = (typeof performance !== "undefined" ? performance.now() : Date.now());
	          if ((now - frameStart) > 12) break;
	          energy = stepSim();
	          steps += 1;
	          if (steps >= maxSteps) break;
	        }}

	        const now2 = (typeof performance !== "undefined" ? performance.now() : Date.now());
	        const elapsed = now2 - start;
	        const ePerNode = energy / Math.max(1, nodes.length);
	        const done = (steps >= maxSteps) || ((elapsed >= maxMs) && (steps >= minSteps)) || ((steps >= minSteps) && (ePerNode < 0.0005));
	        if (!done) {{
	          requestAnimationFrame(frame);
	          return;
	        }}
	        finalizeStatic();
	      }}

	      requestAnimationFrame(frame);
	    }}

	    // Debug hooks for automated checking / operator troubleshooting.
	    window.__chunkhound_viz = {{
	      nodes,
	      links,
	      cam,
	      fitToView,
	      scheduleDraw,
	      selectNode,
	      selectLink,
	      runInitialLayout,
	      setLayout,
	    }};

	    applyThreshold();
	    applySearch();
	    // Compute k-core numbers up-front so the panel can show them even in Force mode.
	    computeKCoreNumbers();
	    updateReadout();
		    if (nodes.length) {{
	      // Default-select the largest system to give operators immediate context.
	      let best = nodes[0];
	      for (const n of nodes) {{
	        if (Number(n.size || 0) > Number(best.size || 0)) best = n;
	      }}
	      selectNode(best.id);
		    }}
		    if (skipInitialLayoutRequested) {{
		      // Show immediately without running the layout.
		      for (const n of nodes) {{ n.vx = 0; n.vy = 0; }}
		      layoutPositions.force = snapshotPositions();
		      fitToView();
		      if (loadingEl) loadingEl.style.display = "none";
		      cv.style.opacity = "1";
		      scheduleDraw();
		    }} else {{
		      runInitialLayout();
		    }}
		    // Ensure the selector matches the active layout.
		    if (layoutSel) layoutSel.value = activeLayout;
		    }}

		    // Allow operators to bypass layout if they want instant, static viewing.
		    if (skipLayoutBtn) {{
		      skipLayoutBtn.addEventListener("click", () => {{
		        skipInitialLayoutRequested = true;
		        cancelInitialLayout = true;
		        // Boot if the user clicks before the idle boot has fired; boot() will honor
		        // skipInitialLayoutRequested and avoid running the warmup layout.
		        if (!booted) {{
		          boot();
		          return;
		        }}
		        const v = window.__chunkhound_viz;
		        if (!v) return;
		        for (const n of v.nodes || []) {{ n.vx = 0; n.vy = 0; }}
		        v.fitToView();
		        if (loadingEl) loadingEl.style.display = "none";
		        cv.style.opacity = "1";
		        v.scheduleDraw();
		      }});
		    }}

	    window.addEventListener("load", () => {{
	      // Accessibility: if the operator prefers reduced motion, skip the warmup layout.
	      try {{
	        if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {{
	          skipInitialLayoutRequested = true;
	          cancelInitialLayout = true;
	        }}
	      }} catch (_) {{}}
	      const run = () => {{
	        try {{
	          boot();
	        }} catch (e) {{
	          console.error("ChunkHound viz boot failed:", e);
	          if (loadingEl) loadingEl.textContent = "Failed to load graph. See console.";
	        }}
	      }};
	      if (typeof window.requestIdleCallback === "function") {{
	        window.requestIdleCallback(run, {{ timeout: 1200 }});
	      }} else {{
	        setTimeout(run, 0);
	      }}
	    }});
	  }})();
	  </script>
	</body>
	</html>
	"""


__all__ = ["render_chunk_systems_viz_html"]
