import json
import warnings

class Message:
    """
        Protocol message class to communicate between the server and the client.
    """

    def __init__(self, type, message):
        self.type = type
        self.content = message

    def __repr__(self):
        message = str(self.content)[:min(50, len(str(self.content)))]
        return f"[Message<{self.type}>]: {message}"
    
    def to_json(self):
          return json.dumps({"type": self.type, "data": {"message": self.content}})
    
    @staticmethod
    def from_json(json_str):
        """
            Dynamically create a message from a json string.
            Can return a different protocole type, but always a protocol class.
        """

        data = json.loads(json_str)

        if "type" not in data or "data" not in data:
            # warning in yellow
            warnings.warn(f"\033[93mInvalid message: {data}\033[0m", stacklevel=2)
            return Error("Invalid message")

        if data["type"] not in TYPES_MAP or TYPES_MAP[data["type"]] == Message:
            return Message(data["type"], message=data["data"])
        return TYPES_MAP[data["type"]].from_json(json_str)
    
class PopUp(Message):
    def __init__(self, message, is_open=True):
        super().__init__("pop-up", message)

        self.is_open = is_open
        self.action = None

    def to_json(self):
        return json.dumps({
            "type": self.type,
            "data": {
                "message": self.content,
                "is_open": self.is_open,
                "action": self.action
            }
        })

    @staticmethod
    def from_json(data):
        pop_up = PopUp(data["data"]["message"], is_open=data["data"]["is_open"])
        pop_up.action = data["data"]["action"]

class Toast(Message):
    def __init__(self, message, duration=5000, is_done=False):
        super().__init__("toast", message)
        self.duration = duration
        self.is_done = is_done

    def to_json(self):
        return json.dumps({
            "type": self.type,
            "data": {
                "message": self.content,
                "duration": self.duration,
                "is_done": self.is_done
            }
        })

    @staticmethod
    def from_json(data):
        toast = Toast(data["data"]["message"])
        toast.duration = data["data"]["duration"]
        toast.is_done = data["data"]["is_done"]
        return toast
    
class Notification(Message):
    def __init__(self, message, is_done=False):
        super().__init__("notification", message)
        self.is_done = is_done

    def to_json(self):
        return json.dumps({
            "type": self.type,
            "data": {
                "message": self.content,
                "is_done": self.is_done
            }
        })

    @staticmethod
    def from_json(data):
        notification = Notification(data["data"]["message"])
        notification.is_done = data["data"]["is_done"]
        return notification
    
class Error(Message):
    def __init__(self, message):
        super().__init__("error", message)

    def to_json(self):
        return json.dumps({
            "type": self.type,
            "data": {
                "message": self.content
            }
        })

    @staticmethod
    def from_json(data):
        return Error(data["data"]["message"])

class MapMessage(Message):
    def __init__(self, map):
        super().__init__("map", map.to_json())

         
TYPES_MAP = {
    # Fondamental types
    "error": Error,
    "message": Message,

    # Basic types
    "pop-up": PopUp,
    "toast": Toast,
    "notification": Notification,
    
    # Complex types
    "map": MapMessage,
}