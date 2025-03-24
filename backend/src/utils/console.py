CONSOLE_STYLE = {
    "ERROR": "\033[91m",
    "SECONDARY_ERROR": "\033[31m",
    "WARNING": "\033[93m",
    "SECONDARY_WARNING": "\033[33m",
    "SUCCESS": "\033[92m",
    "SECONDARY_SUCCESS": "\033[32m",
    "INFO": "\033[94m",
    "SECONDARY_INFO": "\033[96m",
    "END": "\033[0m"
}

class Style:
    """
    A class to represent a styled console message.
    """

    def __init__(self, style, message):
        if style not in CONSOLE_STYLE:
            raise ValueError(f"Invalid style: {style}, must be one of:\n{CONSOLE_STYLE.keys()}")
        self.style = style
        self.message = message

    def __repr__(self):
        return f"{CONSOLE_STYLE[self.style]}{self.message}{CONSOLE_STYLE['END']}"
    
    def __str__(self):
        return self.__repr__()