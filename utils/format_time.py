def format_time(seconds):
    """
    Format durations as hours:minutes:seconds.

    Args:
        seconds: Number of seconds to format

    Returns:
        String in format "HH:MM:SS"
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
