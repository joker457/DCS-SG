from enum import IntEnum


class ModulationType(IntEnum):
    OOK = 0
    ASK4 = 1
    ASK8 = 2
    BPSK = 3
    QPSK = 4
    PSK8 = 5
    PSK16 = 6
    PSK32 = 7
    APSK16 = 8
    APSK32 = 9
    APSK64 = 10
    APSK128 = 11
    QAM16 = 12
    QAM32 = 13
    QAM64 = 14
    QAM128 = 15
    QAM256 = 16
    AM_SSB_WC = 17
    AM_SSB_SC = 18
    AM_DSB_WC = 19
    AM_DSB_SC = 20
    FM = 21
    GMSK = 22
    OQPSK = 23


MOD_NAMES = [
    "OOK", "4ASK", "8ASK",
    "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]

LMAX = 2048
SPS = 8
ROLLOFF = 0.35
NUM_TAPS = 6
RRC_BUFFER_SYMBOLS = 6
OBS_LEVELS = [2048, 1024, 512, 256, 128, 64]
SNR_VALUES = list(range(-20, 31, 2))
DEMAND_DIMS = ["snr", "obs", "chan", "off", "gra", "mul", "dep", "lab"]
NUM_DEMAND_DIMS = 8
SNR_LEVELS = [
    (12.0, 20.0),
    (6.0, 12.0),
    (0.0, 6.0),
    (-6.0, 0.0),
    (-12.0, -6.0),
    (-20.0, -12.0),
]
_SNR_LEVEL_MAP = {}
for _level in range(len(SNR_LEVELS) - 1, -1, -1):
    _lo, _hi = SNR_LEVELS[_level]
    for _val in SNR_VALUES:
        if _lo <= _val <= _hi:
            _SNR_LEVEL_MAP[_val] = _level


def snr_db_to_level(snr_db: float) -> int:
    snr_db = float(snr_db)
    if snr_db >= 12.0:
        return 0
    if snr_db >= 6.0:
        return 1
    if snr_db >= 0.0:
        return 2
    if snr_db >= -6.0:
        return 3
    if snr_db >= -12.0:
        return 4
    return 5


GRANULARITY_LEVEL_MODS = {
    0: [
        "BPSK", "QPSK", "16QAM", "FM",
    ],
    1: [
        "OOK", "4ASK", "BPSK", "QPSK",
        "8PSK", "16QAM", "FM", "GMSK",
    ],
    2: [
        "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK",
        "16PSK", "16QAM", "32QAM", "FM", "GMSK", "OQPSK",
    ],
    3: [
        "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK",
        "16PSK", "32PSK", "16APSK", "32APSK", "16QAM", "32QAM",
        "64QAM", "FM", "GMSK", "OQPSK",
    ],
    4: [
        "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK",
        "16PSK", "32PSK", "16APSK", "32APSK", "64APSK", "128APSK",
        "16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "FM",
        "GMSK", "OQPSK",
    ],
    5: [
        "OOK", "4ASK", "8ASK",
        "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
        "16APSK", "32APSK", "64APSK", "128APSK",
        "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
        "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
        "FM", "GMSK", "OQPSK",
    ],
}


ANALOG_TYPES = {ModulationType.AM_SSB_WC, ModulationType.AM_SSB_SC, ModulationType.AM_DSB_WC, ModulationType.AM_DSB_SC, ModulationType.FM}
CPM_TYPES = {ModulationType.GMSK}
OQPSK_TYPES = {ModulationType.OQPSK}
