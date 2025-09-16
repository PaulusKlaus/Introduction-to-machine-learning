import numpy as np

def parity_from_float(
    s,
    method="round",
    decimals=0,      # brukes av 'scale_floor' og 'last_decimal'
    even_value=1,    # hva skal vi returnere ved "even": 1 (default) eller 0
):
    """
    Returnerer en binær vektor der even -> even_value og odd -> 1-even_value.

    Parametre
    ---------
    s : array-like
        Tallene som skal behandles.
    method : str
        'round'         -> runde til nærmeste heltall
        'floor'         -> heltallsdel med gulv (alltid nedover)
        'ceil'          -> heltallsdel med tak (alltid oppover)
        'scale_floor'   -> multipliser med 10**decimals og ta floor av resultatet
        'last_decimal'  -> se på pariteten til desimalsifferet i posisjon 'decimals'
                           (decimals=1 betyr første siffer etter komma)
    decimals : int
        - For 'scale_floor': hvor mange desimaler du vil “flytte” før heltallsdel.
        - For 'last_decimal': hvilket desimalsiffer du vil inspisere (>=1).
    even_value : int
        Verdien som skal brukes for "even" (0 eller 1). Odd blir 1-even_value.
    """
    s = np.asarray(s, dtype=float)

    if method == "round":
        ints = np.rint(s).astype(np.int64)

    elif method == "floor":
        ints = np.floor(s).astype(np.int64)

    elif method == "ceil":
        ints = np.ceil(s).astype(np.int64)

    elif method == "scale_floor":
        if decimals < 0:
            raise ValueError("decimals må være >= 0 for 'scale_floor'.")
        factor = 10 ** decimals
        # Merk: floor på negative tall går "nedover" (mer negativt). Det er som definert for gulv.
        ints = np.floor(s * factor).astype(np.int64)

    elif method == "last_decimal":
        if decimals < 1:
            raise ValueError("decimals må være >= 1 for 'last_decimal'.")
        factor = 10 ** decimals
        # Finn sifferet i den gitte desimalposisjonen uavhengig av fortegn:
        # 1) skaler opp
        # 2) floor absoluttverdien for å få stabil sifferhenting
        # 3) ta modulo 10 for å hente akkurat dette sifferet
        digits = (np.floor(np.abs(s) * factor).astype(np.int64)) % 10
        is_even = (digits % 2 == 0).astype(int)
        return np.where(is_even == 1, even_value, 1 - even_value).astype(int)

    else:
        raise ValueError(
            "Ukjent method. Bruk 'round' | 'floor' | 'ceil' | 'scale_floor' | 'last_decimal'."
        )

    # Standard paritet på heltall:
    is_even = (ints % 2 == 0).astype(int)
    return np.where(is_even == 1, even_value, 1 - even_value).astype(int)
