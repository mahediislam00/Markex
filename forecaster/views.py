import json
import logging
import yfinance as yf
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.core.cache import cache

from .ml_engine import forecast_with_fallback as forecast, SYMBOL_MAP

logger = logging.getLogger(__name__)

INSTRUMENTS = [
    {"id": "BTC",    "name": "Bitcoin",        "cat": "crypto",      "price": 67420.50, "change": 2.34,  "vol": "38.2B", "mcap": "1.32T", "icon": "₿"},
    {"id": "ETH",    "name": "Ethereum",        "cat": "crypto",      "price": 3842.10,  "change": -1.12, "vol": "18.7B", "mcap": "462B",  "icon": "Ξ"},
    {"id": "SOL",    "name": "Solana",           "cat": "crypto",      "price": 182.45,   "change": 5.67,  "vol": "4.1B",  "mcap": "85B",   "icon": "◎"},
    {"id": "BNB",    "name": "BNB",              "cat": "crypto",      "price": 598.30,   "change": 0.88,  "vol": "1.9B",  "mcap": "92B",   "icon": "B"},
    {"id": "XRP",    "name": "Ripple",           "cat": "crypto",      "price": 0.6210,   "change": -0.43, "vol": "1.2B",  "mcap": "34B",   "icon": "✕"},
    {"id": "AAPL",   "name": "Apple Inc.",       "cat": "stocks",      "price": 228.35,   "change": 1.45,  "vol": "52M",   "mcap": "3.52T", "icon": "A"},
    {"id": "NVDA",   "name": "NVIDIA Corp.",     "cat": "stocks",      "price": 875.20,   "change": 3.21,  "vol": "41M",   "mcap": "2.16T", "icon": "N"},
    {"id": "MSFT",   "name": "Microsoft",        "cat": "stocks",      "price": 418.90,   "change": 0.67,  "vol": "22M",   "mcap": "3.11T", "icon": "M"},
    {"id": "TSLA",   "name": "Tesla Inc.",       "cat": "stocks",      "price": 242.10,   "change": -2.88, "vol": "95M",   "mcap": "775B",  "icon": "T"},
    {"id": "AMZN",   "name": "Amazon.com",       "cat": "stocks",      "price": 196.45,   "change": 1.12,  "vol": "38M",   "mcap": "2.07T", "icon": "A"},
    {"id": "XAU",    "name": "Gold",             "cat": "commodities", "price": 2312.40,  "change": 0.34,  "vol": "142B",  "mcap": "—",     "icon": "Au"},
    {"id": "XAG",    "name": "Silver",           "cat": "commodities", "price": 27.84,    "change": -0.21, "vol": "9.4B",  "mcap": "—",     "icon": "Ag"},
    {"id": "OIL",    "name": "Crude Oil",        "cat": "commodities", "price": 78.92,    "change": 1.56,  "vol": "29B",   "mcap": "—",     "icon": "🛢"},
    {"id": "NG",     "name": "Natural Gas",      "cat": "commodities", "price": 2.145,    "change": -3.22, "vol": "8.1B",  "mcap": "—",     "icon": "⛽"},
    {"id": "SPY",    "name": "SPDR S&P 500",     "cat": "etfs",        "price": 542.10,   "change": 0.88,  "vol": "72M",   "mcap": "498B",  "icon": "S"},
    {"id": "QQQ",    "name": "Invesco QQQ",      "cat": "etfs",        "price": 468.35,   "change": 1.22,  "vol": "45M",   "mcap": "241B",  "icon": "Q"},
    {"id": "VTI",    "name": "Vanguard Total",   "cat": "etfs",        "price": 248.90,   "change": 0.55,  "vol": "3.2M",  "mcap": "398B",  "icon": "V"},
    {"id": "EURUSD", "name": "EUR/USD",          "cat": "forex",       "price": 1.0842,   "change": 0.12,  "vol": "6.8T",  "mcap": "—",     "icon": "€"},
    {"id": "GBPUSD", "name": "GBP/USD",          "cat": "forex",       "price": 1.2710,   "change": -0.08, "vol": "3.1T",  "mcap": "—",     "icon": "£"},
    {"id": "USDJPY", "name": "USD/JPY",          "cat": "forex",       "price": 151.84,   "change": 0.33,  "vol": "4.4T",  "mcap": "—",     "icon": "¥"},
]

CATEGORIES = [
    {"id": "all",         "label": "ALL"},
    {"id": "crypto",      "label": "CRYPTO"},
    {"id": "stocks",      "label": "STOCKS"},
    {"id": "commodities", "label": "COMMODITIES"},
    {"id": "etfs",        "label": "ETFS"},
    {"id": "forex",       "label": "FOREX"},
]

def index(request):
    context = {
        "instruments_json": json.dumps(INSTRUMENTS),
        "categories": CATEGORIES,
    }
    return render(request, "forecaster/index.html", context)

@require_GET
def api_forecast(request, symbol):
    symbol = symbol.upper()
    if symbol not in SYMBOL_MAP:
        return JsonResponse({"error": f"Unknown symbol: {symbol}"}, status=404)
    cache_key = f"forecast_{symbol}"
    cached = cache.get(cache_key)
    if cached:
        return JsonResponse(cached)
    try:
        result = forecast(symbol)
        cache.set(cache_key, result, timeout=600)
        return JsonResponse(result)
    except Exception as e:
        logger.exception(f"Forecast failed for {symbol}")
        return JsonResponse({"error": str(e)}, status=500)

@require_GET
def api_instruments(request):
    return JsonResponse({"instruments": INSTRUMENTS})


@require_GET
def api_price(request, symbol):
    symbol = symbol.upper()
    if symbol not in SYMBOL_MAP:
        return JsonResponse({"error": "Unknown symbol"}, status=404)
    try:
        ticker = yf.Ticker(SYMBOL_MAP[symbol])
        info = ticker.fast_info          # lighter than .info, no rate-limit issues
        price = info.last_price
        return JsonResponse({"symbol": symbol, "price": round(price, 6)})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)