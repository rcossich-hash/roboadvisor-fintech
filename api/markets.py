import json
from http.server import BaseHTTPRequestHandler

SYMBOLS = {
    # Índices USA
    "^GSPC":  {"label": "S&P 500",      "group": "indices"},
    "^NDX":   {"label": "Nasdaq 100",   "group": "indices"},
    "^DJI":   {"label": "Dow Jones",    "group": "indices"},
    "^RUT":   {"label": "Russell 2000", "group": "indices"},
    # Sectores ETFs
    "XLK":    {"label": "Tecnología",   "group": "sectores"},
    "XLE":    {"label": "Energía",      "group": "sectores"},
    "XLF":    {"label": "Financiero",   "group": "sectores"},
    "XLV":    {"label": "Salud",        "group": "sectores"},
    "XLI":    {"label": "Industrial",   "group": "sectores"},
    "XLC":    {"label": "Comunicaciones","group": "sectores"},
    "XLY":    {"label": "Consumo disc.", "group": "sectores"},
    "XLP":    {"label": "Consumo básico","group": "sectores"},
    # Monedas
    "USDCLP=X": {"label": "USD/CLP",    "group": "monedas"},
    "USDGTQ=X": {"label": "USD/GTQ",    "group": "monedas"},
    "EURUSD=X": {"label": "EUR/USD",    "group": "monedas"},
    "DX-Y.NYB": {"label": "DXY",        "group": "monedas"},
    # Commodities
    "GC=F":   {"label": "Oro (XAU)",    "group": "commodities"},
    "CL=F":   {"label": "WTI Crude",    "group": "commodities"},
    "BTC-USD":{"label": "Bitcoin",      "group": "commodities"},
}


def fetch_quotes(symbols):
    try:
        import urllib.request
        syms_str = "%2C".join(symbols)
        url = (
            "https://query1.finance.yahoo.com/v7/finance/quote"
            f"?symbols={syms_str}"
            "&fields=regularMarketPrice,regularMarketChange,regularMarketChangePercent,regularMarketPreviousClose"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = {}
        for q in data.get("quoteResponse", {}).get("result", []):
            sym = q.get("symbol", "")
            price = q.get("regularMarketPrice")
            change = q.get("regularMarketChange", 0)
            pct = q.get("regularMarketChangePercent", 0)
            if price is not None:
                results[sym] = {
                    "price": round(float(price), 4),
                    "change": round(float(change), 4),
                    "pct": round(float(pct), 2),
                }
        return results
    except Exception as e:
        return {"_error": str(e)}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            syms = list(SYMBOLS.keys())
            quotes = fetch_quotes(syms)

            out = {"groups": {}, "timestamp": None}
            import datetime
            out["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

            for sym, meta in SYMBOLS.items():
                g = meta["group"]
                if g not in out["groups"]:
                    out["groups"][g] = []
                q = quotes.get(sym, {})
                out["groups"][g].append({
                    "symbol": sym,
                    "label":  meta["label"],
                    "price":  q.get("price"),
                    "change": q.get("change"),
                    "pct":    q.get("pct"),
                    "error":  q.get("_error"),
                })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
