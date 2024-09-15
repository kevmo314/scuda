Features:
- Multithread the server

Bugs:
- Server exits when the client disconnects
- Client dlsym resolution needs to switch to a hashmap or trie-based lookup table instead of linear strcmp