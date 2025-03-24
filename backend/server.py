
import base64
import traceback
from src.utils.socket_server import ServerSocket
import src.utils.message as protocol

import asyncio

class Server:
    """
    Server class that handles the app.
    """

    def __init__(self):
        self.server = ServerSocket(_print=True)
        self.chunks = []
        self.loading_video = False

    async def run(self):

        await self.server.start()

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_client_connect,
            "client-connect",
            lambda client: asyncio.create_task(self.server.send(client, protocol.Message("confirm-connection", "Connection established").to_json()))
        )

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_message,
            "start_chunks",
            lambda client, message: self.setup_load_chunks() if message.type == "video_chunk_start" else None
        )

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_message,
            "load_chunks",
            lambda client, message: self.load_chunks(message) if message.type == "video_chunk" else None
        )

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_message,
            "check_chunks",
            lambda client, message: self.check_chunks(message) if message.type == "video_chunk_end" else None
        )

        # Main loop
        while self.server.running:
            await asyncio.sleep(2)

    def setup_load_chunks(self):
        if self.loading_video: raise Exception("Already loading a video")
        self.loading_video = True
        self.chunks = []

    def load_chunks(self, message):
        if not self.loading_video: raise Exception("Not loading a video")
        self.chunks.append(message.content['chunk'])

    def check_chunks(self, message):
        if not self.loading_video: raise Exception("Not loading a video")

        # Check if all chunks are present
        if len(self.chunks) != message.content['total_chunks']:
            raise Exception("Missing chunks")
        
        print("All chunks received")
        # Save the chunks to a file
        filename = "tmp.webm"
        try:
            self.save_webm(filename)
            print(f"File saved as {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
            traceback.print_exc()
        finally:
            self.loading_video = False
            self.chunks = []

    def save_webm(self, filename="output.webm"):
        decoded_data = bytearray()
        for chunk in self.chunks:
            # Remove the Data URL header if present
            if chunk.startswith("data:"):
                try:
                    header, b64data = chunk.split(",", 1)
                except ValueError:
                    print("Chunk format error, skipping:", chunk[:30])
                    continue
            else:
                b64data = chunk
            try:
                decoded_data.extend(base64.b64decode(b64data))
            except Exception as e:
                print("Error decoding chunk:", e)
        with open(filename, "wb") as f:
            f.write(decoded_data)
        print("File saved as", filename)


if __name__ == "__main__":
    Server = Server()
    asyncio.run(Server.run())

