def _request(
    method: str,
    table: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | list[dict[str, Any]] | None = None,
    prefer: str | None = None,
) -> list[dict[str, Any]]:
    """
    Send a request to Supabase with automatic retries for temporary failures.

    Retries:
    - 429 Too Many Requests
    - 500 Internal Server Error
    - 502 Bad Gateway
    - 503 Service Unavailable
    - 504 Gateway Timeout
    - connection errors
    - request timeouts
    """

    url, _ = config()

    if not configured():
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_KEY are not configured"
        )

    endpoint = f"{url}/rest/v1/{table}"

    # Clean the payload before sending it to Supabase.
    if isinstance(payload, dict):
        body: Any = _clean_dict(payload)

    elif isinstance(payload, list):
        body = [_clean_dict(x) for x in payload]

    else:
        body = None

    method_upper = method.upper()

    retryable_statuses = {
        429,
        500,
        502,
        503,
        504,
    }

    max_attempts = 5
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):

        try:
            response = requests.request(
                method_upper,
                endpoint,
                headers=_headers(prefer),
                params=params,
                json=body,
                timeout=30,
            )

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------

            if response.ok:

                if not response.text.strip():
                    return []

                try:
                    data = response.json()

                except Exception as exc:
                    raise RuntimeError(
                        f"Supabase {method_upper} {table} "
                        f"returned invalid JSON: {exc}"
                    )

                if isinstance(data, list):
                    return data

                if isinstance(data, dict):
                    return [data]

                return []

            # -------------------------------------------------
            # TEMPORARY / RETRYABLE HTTP ERROR
            # -------------------------------------------------

            if response.status_code in retryable_statuses:

                text = response.text[:800]

                last_error = RuntimeError(
                    f"Supabase {method_upper} {table} failed "
                    f"({response.status_code}): {text}"
                )

                if attempt < max_attempts:

                    wait_seconds = min(
                        2 ** (attempt - 1),
                        10,
                    )

                    print(
                        f"WARNING: Supabase {method_upper} "
                        f"{table} returned "
                        f"{response.status_code}. "
                        f"Retrying in {wait_seconds}s "
                        f"(attempt {attempt}/{max_attempts})..."
                    )

                    time.sleep(wait_seconds)

                    continue

            # -------------------------------------------------
            # NON-RETRYABLE HTTP ERROR
            # -------------------------------------------------

            text = response.text[:800]

            raise RuntimeError(
                f"Supabase {method_upper} {table} failed "
                f"({response.status_code}): {text}"
            )

        # -----------------------------------------------------
        # TIMEOUT
        # -----------------------------------------------------

        except requests.exceptions.Timeout as exc:

            last_error = exc

            if attempt < max_attempts:

                wait_seconds = min(
                    2 ** (attempt - 1),
                    10,
                )

                print(
                    f"WARNING: Supabase timeout for "
                    f"{method_upper} {table}. "
                    f"Retrying in {wait_seconds}s "
                    f"(attempt {attempt}/{max_attempts})..."
                )

                time.sleep(wait_seconds)

                continue

        # -----------------------------------------------------
        # CONNECTION ERROR
        # -----------------------------------------------------

        except requests.exceptions.ConnectionError as exc:

            last_error = exc

            if attempt < max_attempts:

                wait_seconds = min(
                    2 ** (attempt - 1),
                    10,
                )

                print(
                    f"WARNING: Supabase connection error for "
                    f"{method_upper} {table}: {exc}. "
                    f"Retrying in {wait_seconds}s "
                    f"(attempt {attempt}/{max_attempts})..."
                )

                time.sleep(wait_seconds)

                continue

        # -----------------------------------------------------
        # OTHER REQUEST ERROR
        # -----------------------------------------------------

        except requests.exceptions.RequestException as exc:

            last_error = exc

            if attempt < max_attempts:

                wait_seconds = min(
                    2 ** (attempt - 1),
                    10,
                )

                print(
                    f"WARNING: Supabase request error for "
                    f"{method_upper} {table}: {exc}. "
                    f"Retrying in {wait_seconds}s "
                    f"(attempt {attempt}/{max_attempts})..."
                )

                time.sleep(wait_seconds)

                continue

        # -----------------------------------------------------
        # REAL APPLICATION / SUPABASE ERROR
        # -----------------------------------------------------

        except RuntimeError:
            raise

    # ---------------------------------------------------------
    # ALL RETRIES FAILED
    # ---------------------------------------------------------

    raise RuntimeError(
        f"Supabase {method_upper} {table} failed after "
        f"{max_attempts} attempts. "
        f"Last error: {last_error}"
    )
