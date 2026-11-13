"""Which address gives a camera's video (the page's "Testa kameran", ARCHITECTURE §13): for a camera on the store's
own network, which ports answer, what its web page says it is, and how it answers an RTSP DESCRIBE for the common
paths of each make (200: the right address, 401: right but it wants a login, 404: wrong). The first that answers
200 is opened as the anonymizer would (OpenCV/FFmpeg) and one frame read, for its size.

Only addresses on a local network (private, link-local or loopback): it is not a scanner of the internet. The
login, when given, is used for the asks only and is never written, logged or returned: what comes back names each
address without it.
"""

import hashlib
import ipaddress
import re
import socket
import threading
import urllib.error
import urllib.request
from urllib.parse import quote

PORTS = (554, 8554, 80, 8080, 443, 8000, 37777, 34567)
RTSP_PORTS = (554, 8554, 10554)
TIMEOUT_S = 2.5
FRAME_TIMEOUT_S = 12.0
# (make, path): the common ones first; {ch} is the channel (1)
PATHS = (
    ("GeoVision", "/CH001.sdp"),
    ("Hikvision", "/Streaming/Channels/101"),
    ("Hikvision", "/Streaming/Channels/1"),
    ("Dahua", "/cam/realmonitor?channel=1&subtype=0"),
    ("Axis", "/axis-media/media.amp"),
    ("Reolink", "/h264Preview_01_main"),
    ("Uniview", "/unicast/c1/s0/live"),
    ("Hanwha", "/profile2/media.smp"),
    ("Vivotek", "/live.sdp"),
    ("Bosch", "/rtsp_tunnel"),
    ("Foscam", "/videoMain"),
    ("XMeye", "/user=admin_password=_channel=1_stream=0.sdp"),
    ("ONVIF", "/onvif1"),
    ("ONVIF", "/onvif/profile1/media.smp"),
    ("ONVIF", "/MediaInput/h264"),
    ("Allmän", "/live"),
    ("Allmän", "/live/ch00_0"),
    ("Allmän", "/live/main"),
    ("Allmän", "/stream1"),
    ("Allmän", "/stream0"),
    ("Allmän", "/h264"),
    ("Allmän", "/11"),
    ("Allmän", "/1"),
    ("Allmän", "/0"),
    ("Allmän", "/ch0_0.h264"),
    ("Allmän", "/av0_0"),
    ("Allmän", "/media/video1"),
    ("Allmän", "/video1"),
    ("Allmän", "/cam1/h264"),
    ("Allmän", "/"),
)
MAKES = ("geovision", "hikvision", "dahua", "axis", "reolink", "uniview", "hanwha", "wisenet", "vivotek", "bosch",
         "foscam", "xmeye", "milesight", "avigilon", "mobotix", "ubiquiti", "tp-link", "tapo", "ezviz", "imou")
# many cameras ask for the login before they look at the path: without one, 401 does not say the path is right
MEANING = {200: "OK: rätt adress", 401: "Vill ha användare och lösenord",
           403: "Nekad: fel användare eller lösenord", 404: "Finns inte", 400: "Förstår inte adressen",
           454: "Finns inte", 451: "Förstår inte adressen"}


class NotLocal(ValueError):
    pass


def local_host(host: str) -> str:
    """The IP address of `host` when it is on a local network; NotLocal otherwise."""
    host = host.strip().strip("[]")
    if not re.fullmatch(r"[0-9A-Za-z.:-]{1,253}", host):
        raise NotLocal("Skriv kamerans IP-adress, till exempel 192.168.0.64")
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except (OSError, ValueError):
        raise NotLocal(f"Hittar inte {host}") from None
    if not (ip.is_private or ip.is_link_local or ip.is_loopback):
        raise NotLocal("Bara kameror i det egna nätverket (till exempel 192.168.x.x) kan testas")
    return str(ip)


def open_ports(ip: str, ports=None, timeout_s: float = TIMEOUT_S) -> list[int]:
    ports, found, lock = ports or PORTS, [], threading.Lock()

    def one(p):
        try:
            with socket.create_connection((ip, p), timeout_s):
                with lock:
                    found.append(p)
        except OSError:
            pass

    threads = [threading.Thread(target=one, args=(p,), daemon=True) for p in ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout_s + 1)
    return sorted(found)


def web_make(ip: str, port: int = 80, timeout_s: float = TIMEOUT_S) -> dict:
    """What the camera's web page says: its Server header, page title and a make named in either."""
    try:
        with urllib.request.urlopen(f"http://{ip}:{port}/", timeout=timeout_s) as r:
            server, body = r.headers.get("Server", ""), r.read(65536).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:  # a login page answers 401: its headers still name the make
        server, body = e.headers.get("Server", "") or "", ""
        try:
            body = e.read(65536).decode("utf-8", "replace")
        except Exception:
            pass
    except Exception:
        return {}
    m = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
    title = re.sub(r"\s+", " ", m.group(1)).strip()[:80] if m else ""
    text = f"{server} {title} {body[:20000]}".lower()
    return {"server": server[:80], "title": title, "make": next((k for k in MAKES if k in text), None)}


def _read_reply(s: socket.socket) -> tuple[int, dict]:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 65536:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    head = data.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
    m = re.match(r"RTSP/1\.\d (\d{3})", head[0] if head else "")
    if not m:
        raise OSError("inget RTSP-svar")
    headers = {}
    for line in head[1:]:
        k, _, v = line.partition(":")
        headers.setdefault(k.strip().lower(), v.strip())
    return int(m.group(1)), headers


def _digest(user: str, pw: str, method: str, uri: str, challenge: str) -> str:
    f = dict(re.findall(r'(\w+)="?([^",]*)"?', challenge))
    realm, nonce = f.get("realm", ""), f.get("nonce", "")

    def md5(s: str) -> str:  # RTSP's digest is MD5
        return hashlib.md5(s.encode()).hexdigest()

    resp = md5(md5(f"{user}:{realm}:{pw}") + f":{nonce}:" + md5(f"{method}:{uri}"))
    return f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{resp}"'


def describe(ip: str, port: int, path: str, user: str = "", pw: str = "", timeout_s: float = TIMEOUT_S) -> int:
    """The camera's answer to DESCRIBE rtsp://ip:port/path (with the login when it asks and one is given)."""
    uri = f"rtsp://{ip}:{port}{path}"
    with socket.create_connection((ip, port), timeout_s) as s:
        s.settimeout(timeout_s)
        ask = f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\nUser-Agent: Hyllora\r\n"
        s.sendall((ask + "\r\n").encode())
        code, h = _read_reply(s)
        if code == 401 and user:
            ch = h.get("www-authenticate", "")
            if ch.lower().startswith("digest"):
                auth = _digest(user, pw, "DESCRIBE", uri, ch)
            else:
                import base64

                auth = "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()
            s.sendall((ask.replace("CSeq: 1", "CSeq: 2") + f"Authorization: {auth}\r\n\r\n").encode())
            code, _ = _read_reply(s)
        return code


def url(ip: str, port: int, path: str, user: str = "", pw: str = "") -> str:
    login = f"{quote(user, safe='')}:{quote(pw, safe='')}@" if user else ""
    return f"rtsp://{login}{ip}:{port}{path}"


def frame(address: str, timeout_s: float = FRAME_TIMEOUT_S) -> dict:
    """One frame from the address as the anonymizer opens a camera: its size, or why not."""
    import cv2

    out = {}

    def read():
        cap = cv2.VideoCapture(address, cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                out["error"] = "OpenCV kunde inte öppna strömmen"
                return
            ok, img = cap.read()
            if ok and img is not None:
                out.update(width=int(img.shape[1]), height=int(img.shape[0]), fps=round(cap.get(cv2.CAP_PROP_FPS) or 0, 1))
            else:
                out["error"] = "Strömmen öppnades men gav ingen bild"
        finally:
            cap.release()

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(timeout_s)
    return out or {"error": f"Ingen bild på {timeout_s:.0f} s"}


def probe(host: str, user: str = "", pw: str = "") -> dict:
    """Everything the page shows for one camera. Never contains the password."""
    ip = local_host(host)
    ports = open_ports(ip)
    web = web_make(ip, 80) if 80 in ports else web_make(ip, 8080) if 8080 in ports else {}
    rtsp_ports = [p for p in ports if p in RTSP_PORTS] or [p for p in RTSP_PORTS if p in open_ports(ip, RTSP_PORTS)]
    tried = []
    for port in rtsp_ports:
        mk = web.get("make") or ""
        order = sorted(PATHS, key=lambda mp: not (mk and mp[0].lower() in (mk, "wisenet" if mk == "hanwha" else mk)))
        for make, path in order:
            code = None
            try:
                code = describe(ip, port, path, user, pw)
                tried.append({"make": make, "port": port, "path": path, "code": code,
                              "meaning": MEANING.get(code, f"Svarade {code}"), "address": url(ip, port, path)})
            except OSError as e:
                tried.append({"make": make, "port": port, "path": path, "code": None,
                              "meaning": f"Inget svar ({type(e).__name__})", "address": url(ip, port, path)})
            if code == 200:
                break
        if any(t["code"] == 200 for t in tried):
            break
    best = next((t for t in tried if t["code"] == 200), None)
    picture = frame(url(ip, best["port"], best["path"], user, pw)) if best else None
    if not ports:
        advice = "Kameran svarar inte alls: fel IP-adress, eller så är den inte i samma nätverk som den här datorn."
    elif not rtsp_ports:
        advice = "Kameran svarar, men inte på någon RTSP-port (554, 8554, 10554). RTSP kan vara avstängt i kamerans inställningar."
    elif best and picture and "error" not in picture:
        advice = "Den här adressen fungerar. Tryck Använd och sedan Starta."
    elif best:
        advice = "Adressen svarar men ingen bild kom. Prova den igen, eller den mindre strömmen."
    elif any(t["code"] == 401 for t in tried):
        advice = "Rätt adress finns, men kameran vill ha användare och lösenord. Fyll i dem och testa igen."
    elif any(t["code"] == 403 for t in tried):
        advice = "Fel användare eller lösenord."
    else:
        advice = "Ingen av de vanliga adresserna fungerade. Leta upp kamerans RTSP-adress i dess manual eller med ONVIF Device Manager."
    return {"ip": ip, "ports": ports, "web": web, "rtsp_ports": rtsp_ports, "tried": tried,
            "best": best, "frame": picture, "advice": advice}
