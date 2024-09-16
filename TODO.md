Features:
- Add moar APIs
- Multithread the server
- Make port configurable

Bugs:
- Server exits when the client disconnects [done]
- Client dlsym resolution needs to switch to a hashmap or trie-based lookup table instead of linear strcmp