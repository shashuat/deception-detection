import asyncio
import json
import traceback
import websockets

from .message import *
from .console import Style

class ServerSocket:
    """
    A class to manage a WebSocket server.

    Attributes
    ----------
    host : str
        The host of the server.

    port : int
        The port of the server.

    running : bool
        Whether the server is running or not.

    _print : bool
        Whether to print information or not.

    server : websockets.server.WebSocketServer
        The server object.

    clients : set[websockets.server.WebSocketServerProtocol]
        The set of connected clients.

    loop : asyncio.AbstractEventLoop
        The asyncio event loop.

    messages : dict[str, list]
        The messages received from the clients.

    How to use:
    ----------

    --- synchrone --------------------
    >>> server = ServerSocket() # Create a server

    --- asynchrone --------------------
    >>> await server.start() # Start the server
    >>> await server.wait_for_clients(1) # Wait for a client to connect
    >>> await server.broadcast("Hello, clients!") # Broadcast a message to all clients
    >>> await server.stop() # Stop the server
    """

    class EVENTS_TYPES:
        on_client_connect = "on_client_connect"
        """
        Event triggered when a client connects to the server
        Listener arguments: client
        """

        on_client_disconnect = "on_client_disconnect"
        """
        Event triggered when a client disconnects from the server
        Listener arguments: client
        """

        on_message = "on_message"
        """
        Event triggered when a message is received from a client
        Listener arguments: client, message
        """

        on_server_stop = "on_server_stop"
        """
        Event triggered when the server is stopped
        Listener arguments: None
        """

        @staticmethod
        def all():
            return [
                event for event in ServerSocket.EVENTS_TYPES.__dict__.values()
                if type(event) is str and not event.startswith("__")
            ]

    HISTORY_LIMIT = 3
    def __init__(self, host="127.0.0.1", port=9000, _print=False):
        self.host = host
        self.port = port
        self.running = False
        self._print = _print

        self.server = None
        self.clients = set()  # To keep track of connected clients

        self.loop = asyncio.get_event_loop()

        self.messages: dict[str, list] = {}

        self._events_listeners = {event: {} for event in self.EVENTS_TYPES.all()}

    def _update_history(self, client, message):
        if client.remote_address not in self.messages:
            self.messages[client.remote_address] = []
        self.messages[client.remote_address].append(message)
        if len(self.messages[client.remote_address]) > self.HISTORY_LIMIT:
            self.messages[client.remote_address].pop(0)

        if self._print:
            print("[info]\t\t", Style("INFO", f"Client {client.remote_address}: {message}"))

    async def _execute_event(self, event_type, *args):
        listeners_output = []
        for listener in self._events_listeners[event_type].values():
            try:
                listeners_output.append(listener(*args))
            except TypeError as e:
                # if the listener does not have the right number of arguments
                warnings.warn(Style("WARNING", f"Error occurred in {event_type} event.\nListener does not have the right number of arguments: {e}\nThe listener will not be executed."), stacklevel=2)
                traceback.print_exc()

            except Exception as e:
                warnings.warn(f"Error occurred in {event_type} event: {e}", stacklevel=2)
                traceback.print_exc()

        # s'il y a des éléments dans listeners_output que l'on doit await, alors les await
        listeners_output = [output for output in listeners_output if asyncio.iscoroutine(output)]
        # if listeners_output:
        #     await asyncio.wait(listeners_output, timeout=3)

        if listeners_output:
            # Turn each coroutine into a Task
            tasks = [asyncio.create_task(coro) for coro in listeners_output]
            
            # Now pass tasks (not raw coroutines) to asyncio.wait
            done, pending = await asyncio.wait(tasks, timeout=3)


    async def handler(self, websocket, path=None):
        """Register client and manage communication."""
        # Register the client
        self.clients.add(websocket)
        self._update_history(websocket, Message("network", "Client connected"))

        if self._print:
            print("[network]\t", Style("SECONDARY_SUCCESS", f"Client connected: {websocket.remote_address}"))

        # execute the on_client_connect event
        await self._execute_event(self.EVENTS_TYPES.on_client_connect, websocket)

        # keep the connection alive
        while True:
            try:
                # Wait for a message from the client
                message = await websocket.recv()
                message = Message.from_json(message)

                # execute the on_message event
                await self._execute_event(self.EVENTS_TYPES.on_message, websocket, message)

                self._update_history(websocket, message)

            # If the client disconnects, remove it from the list of clients
            except websockets.ConnectionClosed:
                message = Message("network", "Client disconnected")

                # execute the on_client_disconnect event
                await self._execute_event(self.EVENTS_TYPES.on_client_disconnect, websocket)

                self._update_history(websocket, message)
                self.clients.remove(websocket)

                if self._print:
                    print("[network]\t", Style("SECONDARY_WARNING", f"Client disconnected: {websocket.remote_address}"))
                break

            # if error occurs, remove the client
            except Exception as e:
                message = Error(str(e))
                warnings.warn(Style("ERROR", f"Error occurred: {e}"), stacklevel=2)
                traceback.print_exc()
                self._update_history(websocket, message)
                self.clients.remove(websocket)
                break

    async def start(self):
        """Start the server."""
        if self.running:
            raise Exception("Server is already running")
        
        self.running = True
        self.server = await websockets.serve(self.handler, self.host, self.port)
        if self._print:
            print("[server]\t", Style("SUCCESS", f"Server started at ws://{self.host}:{self.port}"))

    async def stop(self):
        """Stop the server."""
        # execute the on_server_stop event
        await self._execute_event(self.EVENTS_TYPES.on_server_stop)

        # close all clients
        closing_tasks = [asyncio.create_task(client.close()) for client in self.clients]
        if closing_tasks:
            await asyncio.wait(closing_tasks, timeout=3)
        
        # check if some clients are still connected
        if len(self.clients) > 0:
            warnings.warn(f"Failed to close {len(self.clients)} clients; closing forcefully")

        self.server.close()
        self.running = False

        if self._print:
            print("[server]\t", Style("SECONDARY_ERROR", "Server stopped"))

    async def broadcast(self, message):
        """Broadcast a message to all connected clients."""
        if type(message) is not str:
            print(Style("ERROR", message))
            raise ValueError("Message must be a string")
        if not self.running:
            raise Exception("Server is not running")
        for client in self.clients:
            await client.send(message)

    async def send(self, client, message):
        """Send a message to a specific client."""
        if not self.running:
            raise Exception("Server is not running")
        await client.send(message)

    async def wait_for_clients(self, num_clients):
        """Wait until the specified number of clients are connected."""
        if self._print:
            print("[server]\t", Style("SECONDARY_INFO", f"Waiting for {num_clients} clients to be connected"))
        while len(self.clients) < num_clients:
            await asyncio.sleep(1)

    def on(self, event_type, listener_id, listener):
        """Add an event listener."""
        if event_type not in self._events_listeners:
            raise ValueError(f"Invalid event type: {event_type}")
        if listener_id in self._events_listeners[event_type]:
            raise ValueError(f"Listener with id {listener_id} already exists")
        
        self._events_listeners[event_type][listener_id] = listener
        return listener_id
    
    def remove_listener(self, event_type, listener_id):
        """Remove an event listener."""
        if event_type not in self._events_listeners:
            raise ValueError(f"Invalid event type: {event_type}")
        if listener_id not in self._events_listeners[event_type]:
            raise ValueError(f"Listener with id {listener_id} does not exist")
        
        del self._events_listeners[event_type][listener_id]
        return listener_id


async def main():
    SERVER = ServerSocket(_print=True)
    await SERVER.start()

    SERVER.on(ServerSocket.EVENTS_TYPES.on_client_connect, "hellow", lambda client: print(f"Client connected"))

    await SERVER.wait_for_clients(1)

    await asyncio.sleep(1)
    await SERVER.broadcast(json.dumps({"type": "message", "data": {"message": "Hello, clients!"}}))
    await SERVER.broadcast(PopUp("Hello, clients!").to_json())
    await SERVER.stop()

if __name__ == "__main__":
    asyncio.run(main())