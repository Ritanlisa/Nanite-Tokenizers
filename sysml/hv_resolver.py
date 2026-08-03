"""
Hypervariable Resolver
—— Pure protocol-driver layer, completely independent of SysML model.

Each driver implements:
  read(source, selector) -> str       for Instant/Block/ModifierVariable
  write(source, selector, value) -> bool  for Parameter
  execute(source, selector, params) -> str  for Operation

Protocol scheme mapping:
  file://path          → read local files
  bash://              → bash -c <selector>
  cmd://               → exec <selector> directly (no shell)
  ps://                → powershell -c <selector>
  http://host:port     → plain HTTP GET
  https://host:port    → TLS HTTP GET
  ipmi://[host]        → ipmitool
  sensor://            → abstract sensor (stub)
  config://            → config value (stub)
  snmp://host          → SNMP walk (stub)

Usage (model-independent):
  resolver = HVResolver()
  value = resolver.resolve("file:///proc/cpuinfo", "model name")
  result = resolver.execute("bash://", "uptime && whoami")
  ok = resolver.write("file:///tmp/flag", "1")
  enriched = resolver.enrich_text("temp is <instvar:fan_speed:Fan Speed>")
"""

from __future__ import annotations
import re, subprocess, shlex, json
from pathlib import Path


# ── Protocol Driver Interface ──

class HVDriver:
    protocol: str = ""

    def read(self, source: str, selector: str) -> str:
        raise NotImplementedError

    def write(self, source: str, selector: str, value: str) -> bool:
        raise NotImplementedError

    def execute(self, source: str, selector: str, params: dict = None) -> str:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════
#  Real Drivers
# ══════════════════════════════════════════════════════════════

class FileDriver(HVDriver):
    """file:// — read/write local files. selector is relative path from source."""
    protocol = "file"

    def _resolve_path(self, source: str, selector: str) -> Path:
        base = source.split("://", 1)[1] if "://" in source else ""
        if selector:
            return Path(base).expanduser() / selector if base else Path(selector).expanduser()
        return Path(base).expanduser() if base else Path(".")

    def read(self, source: str, selector: str) -> str:
        try:
            p = self._resolve_path(source, selector)
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="replace").strip()[:10000]
            return f"[file not found: {p}]"
        except Exception as e:
            return f"[file error: {e}]"

    def write(self, source: str, selector: str, value: str) -> bool:
        try:
            p = self._resolve_path(source, selector)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(value, encoding="utf-8")
            return True
        except Exception:
            return False

    def execute(self, source: str, selector: str, params: dict = None) -> str:
        return self.read(source, selector)


class BashDriver(HVDriver):
    """bash:// — runs selector as a bash -c command."""
    protocol = "bash"

    def _run(self, cmd: str, timeout: int = 30) -> str:
        if not cmd:
            return "[bash: empty command]"
        try:
            r = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
            )
            out = r.stdout.strip() or r.stderr.strip()
            return out[:10000] if out else f"[bash: empty]"
        except FileNotFoundError:
            return "[bash not found]"
        except subprocess.TimeoutExpired:
            return "[bash: timeout]"
        except Exception as e:
            return f"[bash error: {e}]"

    def read(self, source: str, selector: str) -> str:
        return self._run(selector, timeout=5)

    def execute(self, source: str, selector: str, params: dict = None) -> str:
        return self._run(selector, timeout=120)


class CommandDriver(HVDriver):
    """cmd:// — executes selector as a direct executable (no shell)."""
    protocol = "cmd"

    def _run(self, cmdline: str, timeout: int = 30) -> str:
        if not cmdline:
            return "[cmd: empty]"
        try:
            parts = shlex.split(cmdline)
            if not parts:
                return "[cmd: empty]"
            r = subprocess.run(parts, capture_output=True, text=True, timeout=timeout)
            out = r.stdout.strip() or r.stderr.strip()
            return out[:10000] if out else f"[cmd: empty output]"
        except FileNotFoundError:
            return f"[cmd not found: {cmdline.split()[0]}]"
        except subprocess.TimeoutExpired:
            return "[cmd: timeout]"
        except Exception as e:
            return f"[cmd error: {e}]"

    def read(self, source: str, selector: str) -> str:
        return self._run(selector, timeout=10)

    def execute(self, source: str, selector: str, params: dict = None) -> str:
        return self._run(selector, timeout=120)


class PowerShellDriver(HVDriver):
    """ps:// — runs selector as PowerShell command."""
    protocol = "ps"

    def _run(self, cmd: str, timeout: int = 30) -> str:
        if not cmd:
            return "[ps: empty]"
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True, text=True, timeout=timeout,
            )
            out = r.stdout.strip() or r.stderr.strip()
            return out[:10000] if out else "[ps: empty]"
        except FileNotFoundError:
            return "[powershell not found]"
        except subprocess.TimeoutExpired:
            return "[ps: timeout]"
        except Exception as e:
            return f"[ps error: {e}]"

    def read(self, source: str, selector: str) -> str:
        return self._run(selector, timeout=10)

    def execute(self, source: str, selector: str, params: dict = None) -> str:
        return self._run(selector, timeout=120)


class HTTPDriver(HVDriver):
    """http:// — plain HTTP GET. source = http://host:port/path, selector = query string."""
    protocol = "http"

    def read(self, source: str, selector: str) -> str:
        try:
            import httpx
            url = source.rstrip("/")
            if selector:
                url += f"?{selector.lstrip('?')}"
            r = httpx.get(url, timeout=10, verify=False)
            return r.text[:10000] if r.text else f"[http: {r.status_code}]"
        except ImportError:
            return "[httpx not installed]"
        except Exception as e:
            return f"[http error: {e}]"

    def execute(self, source: str, selector: str, params: dict = None) -> str:
        try:
            import httpx
            url = source.rstrip("/")
            method = (params or {}).get("method", "POST")
            body = json.dumps(params) if params and "method" not in params else None
            r = httpx.request(method, url, content=body, timeout=30, verify=False)
            return r.text[:10000] if r.text else f"[http: {r.status_code}]"
        except ImportError:
            return "[httpx not installed]"
        except Exception as e:
            return f"[http error: {e}]"

    def write(self, source: str, selector: str, value: str) -> bool:
        try:
            import httpx
            url = source.rstrip("/")
            r = httpx.put(url, content=value, timeout=10, verify=False)
            return r.is_success
        except Exception:
            return False


class HTTPSDriver(HVDriver):
    """https:// — TLS HTTP GET. Same as HTTPDriver but with TLS."""
    protocol = "https"

    def read(self, source: str, selector: str) -> str:
        return HTTPDriver().read(source, selector)

    def execute(self, source: str, selector: str, params: dict = None) -> str:
        return HTTPDriver().execute(source, selector, params)

    def write(self, source: str, selector: str, value: str) -> bool:
        return HTTPDriver().write(source, selector, value)


class IPMIDriver(HVDriver):
    """ipmi:// — IPMI sensor/control via ipmitool."""
    protocol = "ipmi"

    def _run(self, args: list[str]) -> str:
        try:
            r = subprocess.run(["ipmitool"] + args, capture_output=True, text=True, timeout=30)
            out = r.stdout.strip() or r.stderr.strip()
            return out[:10000] if out else f"[ipmi: empty]"
        except FileNotFoundError:
            return "[ipmitool not installed]"
        except subprocess.TimeoutExpired:
            return "[ipmi: timeout]"
        except Exception as e:
            return f"[ipmi error: {e}]"

    def read(self, source: str, selector: str) -> str:
        host = source.split("://")[-1] if "://" in source else ""
        if "sensor" in source.lower() or not selector.startswith("sdr"):
            return self._run(["sdr", "get", selector] if host else ["sensor", "get", selector])
        return self._run([selector])

    def execute(self, source: str, selector: str, params: dict = None) -> str:
        host = source.split("://")[-1] if "://" in source else ""
        cmd = ["-H", host] if host else []
        cmd += shlex.split(selector)
        return self._run(cmd)


# ══════════════════════════════════════════════════════════════
#  Stub Drivers (placeholder)
# ══════════════════════════════════════════════════════════════

class StubDriver(HVDriver):
    protocol = "stub"
    def read(self, source, selector): return f"[stub:{selector}]"
    def write(self, source, selector, value): return True
    def execute(self, source, selector, params=None): return f"[stub:exec:{selector}]"


class SensorDriver(HVDriver):
    protocol = "sensor"
    def read(self, source, selector): return f"[sensor:{selector}]"


class ConfigDriver(HVDriver):
    protocol = "config"
    def read(self, source, selector): return f"[config:{selector}]"
    def write(self, source, selector, value): return True


class SNMPDriver(HVDriver):
    protocol = "snmp"
    def read(self, source, selector):
        try:
            host = source.split("://")[-1] if "://" in source else "localhost"
            r = subprocess.run(["snmpget", "-v2c", "-c", "public", host, selector],
                               capture_output=True, text=True, timeout=10)
            return r.stdout.strip()[:500] or f"[snmp: {r.stderr.strip()}]"
        except FileNotFoundError:
            return "[snmpget not installed]"
        except Exception as e:
            return f"[snmp error: {e}]"


class PrometheusDriver(HVDriver):
    protocol = "prom"
    def read(self, source, selector):
        try:
            import httpx
            base = source.split("://")[-1] if "://" in source else "http://localhost:9090"
            r = httpx.get(f"{base}/api/v1/query", params={"query": selector}, timeout=10)
            data = r.json()
            results = data.get("data", {}).get("result", [])
            if results:
                return str(results[0].get("value", results[0]))
            return f"[prom: no data for {selector}]"
        except ImportError:
            return "[httpx not installed]"
        except Exception as e:
            return f"[prom error: {e}]"


# ══════════════════════════════════════════════════════════════
#  Resolver — pure protocol dispatch, NO model dependency
# ══════════════════════════════════════════════════════════════

_HV_TAG_RE = re.compile(r'<(\w+):([a-zA-Z_][a-zA-Z0-9_]*):([^>]+)>')

_BUILTIN_DRIVERS: dict[str, HVDriver] = {
    "file": FileDriver(),
    "bash": BashDriver(),
    "cmd": CommandDriver(),
    "ps": PowerShellDriver(),
    "http": HTTPDriver(),
    "https": HTTPSDriver(),
    "ipmi": IPMIDriver(),
    "sensor": SensorDriver(),
    "config": ConfigDriver(),
    "snmp": SNMPDriver(),
    "prom": PrometheusDriver(),
    "stub": StubDriver(),
}


class HVResolver:
    """Pure protocol-driven resolver. No SysML model dependency.

    Usage:
        resolver = HVResolver()
        # Resolve a value from source+selector
        val = resolver.resolve("file:///proc/cpuinfo", "model name")
        val = resolver.resolve("bash://", "echo hello")
        val = resolver.resolve("cmd://", "uptime -p")
        # Execute an operation
        result = resolver.execute("cmd://", "reboot", {"delay": "5m"})
        # Write a parameter
        ok = resolver.write("file:///tmp/value", "42")
        # Enrich tagged text
        text = resolver.enrich_text("temp is <instvar:fan:Fan Speed>")
    """

    def __init__(self):
        self._drivers: dict[str, HVDriver] = dict(_BUILTIN_DRIVERS)

    def register_driver(self, protocol: str, driver: HVDriver):
        self._drivers[protocol] = driver

    def _parse_source(self, source: str) -> tuple[str, HVDriver]:
        """Extract protocol from source and return (protocol, driver)."""
        if "://" in source:
            proto = source.split("://")[0].lower()
        else:
            proto = "bash"  # bare string → bash command
        return proto, self._drivers.get(proto, self._drivers["stub"])

    def resolve(self, source: str, selector: str) -> str:
        """Resolve value from source+selector."""
        proto, driver = self._parse_source(source)
        return driver.read(source, selector)

    def execute(self, source: str, selector: str, params: dict = None) -> str:
        """Execute an operation."""
        proto, driver = self._parse_source(source)
        return driver.execute(source, selector, params or {})

    def write(self, source: str, selector: str, value: str) -> bool:
        """Write a parameter value."""
        proto, driver = self._parse_source(source)
        return driver.write(source, selector, value)

    def enrich_text(self, text: str, lookup: dict[str, tuple[str, str]] = None) -> str:
        """Replace HV tags with resolved values.

        Tags: <instvar:id:Name>, <blockvar:id:Name>, <param:id:Name>, <operation:id:Name>

        If `lookup` is provided, it maps hv_id → (source, selector) for resolution.
        Without lookup, tags are replaced with their display name only (no live value).

        Args:
            text: Tagged text like "If <instvar:fan_speed:Fan Speed> > 3000"
            lookup: Optional dict of hv_id → (source, selector) for live resolution

        Returns:
            Text with tags replaced. If lookup provides config, values are live-resolved.
        """
        def replacer(m):
            tag_type = m.group(1)
            hv_id = m.group(2)
            display = m.group(3)
            if lookup and hv_id in lookup:
                src, sel = lookup[hv_id]
                value = self.resolve(src, sel)
                return f"{display}:{value}"
            return display
        return _HV_TAG_RE.sub(replacer, text)
