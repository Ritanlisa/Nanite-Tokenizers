"""
Gephi Graph Streaming client for real-time KG visualization.

Connects to Gephi's Graph Streaming plugin (WebSocket) and pushes
graph operations as they happen. Start Gephi → Tools → Streaming → Server → Start (port 8080).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

try:
    import websocket as _ws
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

logger = logging.getLogger(__name__)

# Default Gephi Streaming port (configurable via gephi.port in settings)
DEFAULT_PORT = 8080
DEFAULT_HOST = "localhost"


class GephiStreamer:
    """Push KG entities/relations to Gephi via Graph Streaming WebSocket."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.ws: Optional[_ws.WebSocket] = None
        self.connected = False
        self._node_ids: set[str] = set()  # track what's been sent
        self._edge_ids: set[str] = set()

    @property
    def url(self) -> str:
        # Gephi Graph Streaming Plugin default: ws://host:port/ (root), or /workspace0
        return f"ws://{self.host}:{self.port}/"

    def connect(self) -> bool:
        """Connect to Gephi Streaming plugin. Returns True if successful."""
        if not HAS_WEBSOCKET:
            logger.warning("Gephi: websocket-client not installed")
            return False
        try:
            self.ws = _ws.create_connection(self.url, timeout=3)
            self.connected = True
            logger.info("Gephi: connected to %s", self.url)
            return True
        except Exception as e:
            logger.debug("Gephi: connection failed (%s) — skip streaming", e)
            self.connected = False
            return False

    def disconnect(self):
        if self.ws and self.connected:
            try:
                self.ws.close()
            except Exception:
                pass
        self.connected = False
        self.ws = None

    def send(self, msg: dict) -> bool:
        """Send a JSON message. Returns True if sent."""
        if not self.connected:
            return False
        try:
            self.ws.send(json.dumps(msg, ensure_ascii=False))
            return True
        except Exception as e:
            logger.warning("Gephi: send failed (%s), reconnecting...", e)
            self.connected = False
            if self.connect():
                try:
                    self.ws.send(json.dumps(msg, ensure_ascii=False))
                    return True
                except Exception:
                    pass
            return False

    # ── High-level operations ──

    def clear_graph(self) -> bool:
        """Clear all nodes and edges in Gephi."""
        return self.send({"type": "clear_graph"})

    def add_node(
        self,
        node_id: str,
        label: str = "",
        node_type: str = "",
        x: float = 0.0,
        y: float = 0.0,
        size: float = 1.0,
        color: str = "",
        **attributes,
    ) -> bool:
        """Add or update a node in Gephi."""
        msg: Dict[str, Any] = {"type": "node", "id": node_id}
        if label:
            msg["label"] = label
        if node_type:
            msg["attributes"] = msg.get("attributes", {})
            msg["attributes"]["entity_type"] = node_type
        if x or y:
            msg["x"] = x
            msg["y"] = y
        if size != 1.0:
            msg["size"] = size
        if color:
            msg["color"] = color
        if attributes:
            msg["attributes"] = {**msg.get("attributes", {}), **attributes}
        sent = self.send(msg)
        if sent:
            self._node_ids.add(node_id)
        return sent

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        label: str = "",
        relation_type: str = "",
        directed: bool = True,
        **attributes,
    ) -> bool:
        """Add an edge in Gephi."""
        msg: Dict[str, Any] = {
            "type": "edge",
            "id": edge_id,
            "source": source,
            "target": target,
            "directed": directed,
        }
        if label:
            msg["label"] = label
        if relation_type:
            msg["attributes"] = msg.get("attributes", {})
            msg["attributes"]["relation_type"] = relation_type
        if attributes:
            msg["attributes"] = {**msg.get("attributes", {}), **attributes}
        sent = self.send(msg)
        if sent:
            self._edge_ids.add(edge_id)
        return sent

    def delete_node(self, node_id: str) -> bool:
        """Delete a node."""
        return self.send({"type": "delete_node", "id": node_id})

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        return self.send({"type": "delete_edge", "id": edge_id})

    def sync_entity(self, qualified_name: str, entity_type: str = "PartDef", **attrs) -> None:
        """Convenience: push a KG entity as a Gephi node."""
        self.add_node(
            node_id=qualified_name,
            label=qualified_name.split("::")[-1] if "::" in qualified_name else qualified_name,
            node_type=entity_type,
            **attrs,
        )

    def sync_relation(self, relation_id: str, source: str, target: str, rel_type: str, **attrs) -> None:
        """Convenience: push a KG relation as a Gephi edge."""
        self.add_edge(
            edge_id=relation_id,
            source=source,
            target=target,
            relation_type=rel_type,
            **attrs,
        )

    # ── Batch sync ──

    def sync_knowledge_graph(self, entities: List[Dict], relations: List[Dict]) -> None:
        """Batch-sync an entire KG to Gephi."""
        if not self.connected and not self.connect():
            logger.warning("Gephi: cannot sync — not connected")
            return
        self.clear_graph()
        time.sleep(0.2)
        for ent in entities:
            qn = ent.get("qualified_name") or ent.get("name") or ""
            if not qn:
                continue
            self.sync_entity(
                qualified_name=qn,
                entity_type=ent.get("entity_type") or ent.get("type") or "PartDef",
            )
        for rel in relations:
            rid = rel.get("id") or rel.get("qualified_name") or f"{rel['source']}->{rel['target']}"
            self.sync_relation(
                relation_id=rid,
                source=rel["source"],
                target=rel["target"],
                rel_type=rel.get("relation_type") or rel.get("type") or "ReferenceUsage",
            )
        logger.info("Gephi: synced %d entities, %d relations", len(entities), len(relations))
