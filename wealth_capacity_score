import math
from dataclasses import dataclass

@dataclass
class DonorData:
    largest_gift: float
    total_giving: float
    avg_gift: float
    gifts_per_year: float
    months_since_last_gift: float
    one_time_gifts: int
    total_gifts: int
    home_value: float
    national_median: float
    zip_median: float


@dataclass
class Weights:
    w1: float = 0.15  # D_max
    w2: float = 0.12  # D_total
    w3: float = 0.08  # D_avg
    w4: float = 0.08  # D_freq
    w5: float = 0.10  # D_recency
    w6: float = 0.07  # D_onetime
    w7: float = 0.25  # H_nat
    w8: float = 0.15  # H_local


# ---------- Helper Functions (Normalization) ---------- #

def log_normalize(amount, cap):
    """
    Log normalization: maps [0, cap] => [0, 100]
    """
    if amount <= 0:
        return 0.0
    return 100 * min(
        math.log10(1 + amount) / math.log10(1 + cap),
        1
    )


def log_freq(gifts_per_year):
    """
    Log scaling for yearly gift frequency.
    Cap = 52 gifts/year (weekly)
    """
    if gifts_per_year <= 0:
        return 0.0
    return 100 * min(
        math.log2(1 + gifts_per_year) / math.log2(1 + 52),
        1
    )


def recency_score(months, tau=12):
    """
    Exponential decay: recent gifts matter more.
    """
    if months < 0:
        months = 0
    return 100 * math.exp(-months / tau)


def pct_one_time(one_time, total):
    if total == 0:
        return 0
    return 100 * (one_time / total)


def home_value_log_ratio(value, median, cap_ratio=2):
    """
    Log ratio home value component.
    Compares home value to median, capped at 2× median.
    Maps to [0, 50].
    """
    if value <= 0 or median <= 0:
        return 0.0

    ratio = value / median
    return 50 * min(
        math.log10(1 + ratio) / math.log10(1 + cap_ratio),
        1
    )


# ---------- Main WCS Function ---------- #

def compute_wcs(d: DonorData, w: Weights = Weights()) -> float:
    """
    Computes the Wealth Capacity Score (0–100).
    """

    # Donation components using logarithmic scaling
    D_max    = log_normalize(d.largest_gift, cap=10_000_000)
    D_total  = log_normalize(d.total_giving, cap=10_000_000)
    D_avg    = log_normalize(d.avg_gift, cap=10_000_000)

    # Frequency
    D_freq = log_freq(d.gifts_per_year)

    # Recency
    D_recency = recency_score(d.months_since_last_gift)

    # One-time %
    D_onetime = pct_one_time(d.one_time_gifts, d.total_gifts)

    # Home values
    H_nat   = home_value_log_ratio(d.home_value, d.national_median)
    H_local = home_value_log_ratio(d.home_value, d.zip_median)

    # Final WCS
    WCS = (
        w.w1 * D_max +
        w.w2 * D_total +
        w.w3 * D_avg +
        w.w4 * D_freq +
        w.w5 * D_recency +
        w.w6 * D_onetime +
        w.w7 * H_nat +
        w.w8 * H_local
    )

    return round(WCS, 2)


# ---------- Example Usage ---------- #

if __name__ == "__main__":
    donor = DonorData(
        largest_gift=250000,
        total_giving=1200000,
        avg_gift=5000,
        gifts_per_year=6,
        months_since_last_gift=4,
        one_time_gifts=12,
        total_gifts=20,
        home_value=850000,
        national_median=420000,
        zip_median=600000
    )

    print("Wealth Capacity Score:", compute_wcs(donor))
