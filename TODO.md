Features:
- Add moar APIs
- Multithread the server [done]
- Make port configurable [done]
- Add proper debug logging
- Add TLS for the socket

Bugs:
- Server exits when the client disconnects [done]
- Client dlsym resolution needs to switch to a hashmap or trie-based lookup table instead of linear strcmp [done]
